#!/usr/bin/env python
"""Phase 1 entry point.

Reads a JSON config, runs the compatibility gate on each (base, expert) pair,
computes the per-module conflict score, standardises it, aggregates to
per-position P_π, runs the weight-norm saliency baseline, and writes all
results plus a markdown summary.

Usage:
    python scripts/run_phase1.py --config configs/qwen2.5_1.5b.json
    python scripts/run_phase1.py --config configs/qwen2.5_1.5b.json --dry-run
    python scripts/run_phase1.py --config configs/qwen2.5_1.5b.json \\
        --max-layers 4  # smoke test: only score the first 4 layers
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Allow running as `python scripts/run_phase1.py` from the code/ directory.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

import torch  # noqa: E402

from cusp.aggregate import (  # noqa: E402
    aggregate_pairwise_cosines,
    insertion_plan,
    pearson,
    per_position_score,
    position_score_curve,
    spearman,
    standardise_within_scope,
    topk_overlap,
)
from cusp.compat import check_compatibility, render_compat_markdown  # noqa: E402
from cusp.conflict import (  # noqa: E402
    compute_module_conflict,
    expert_full_model_norms,
    pairwise_cosines,
)
from cusp.io import (  # noqa: E402
    Config,
    load_config,
    load_model_state,
    module_tensor_key,
)
from cusp.saliency import (  # noqa: E402
    saliency_position_scores,
    weight_norm_saliency,
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _ensure_outdir(cfg: Config) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg.output_dir


def _dump_json(obj: Any, path: Path) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    _log(f"wrote {path}")


# ---------------------------------------------------------------- dry run --


def _dry_run(cfg: Config) -> int:
    print("CUSP Phase 1 dry-run")
    print(f"  config name : {cfg.name}")
    print(f"  output dir  : {cfg.output_dir}")
    print(f"  base        : {cfg.base.repo}@{cfg.base.revision}")
    for i, e in enumerate(cfg.experts):
        print(f"  expert {i}    : {e.role:<10} {e.repo}@{e.revision}")
    print(f"  modules     : {cfg.modules}")
    print(f"  score       : tau_mag={cfg.score.tau_mag_percentile} "
          f"norm={cfg.score.expert_norm} zscope={cfg.score.module_zscore_scope}")
    print(f"  position    : min_gap={cfg.position.min_gap} k={cfg.position.default_k}")
    return 0


# ------------------------------------------------------------------- main --


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CUSP Phase 1")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and exit without downloading anything.",
    )
    ap.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Limit scoring to the first N layers (smoke test). "
             "Compatibility checks still run on the whole state dict.",
    )
    ap.add_argument(
        "--skip-compat",
        action="store_true",
        help="Skip the compatibility gate (use only if you have already "
             "verified it once and want to iterate on the score).",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.dry_run:
        return _dry_run(cfg)

    outdir = _ensure_outdir(cfg)

    # ---- 1. download + open all checkpoints ----
    _log(f"opening base {cfg.base.repo}@{cfg.base.revision}")
    base_state = load_model_state(cfg.base.repo, revision=cfg.base.revision)
    _log(f"  num_hidden_layers = {base_state.num_layers}")

    expert_states = []
    for ecfg in cfg.experts:
        _log(f"opening expert {ecfg.role}: {ecfg.repo}@{ecfg.revision}")
        expert_states.append(load_model_state(ecfg.repo, revision=ecfg.revision))

    # ---- 2. compatibility gate ----
    compat_reports = []
    if not args.skip_compat:
        for ecfg, est in zip(cfg.experts, expert_states):
            _log(f"compat check: {ecfg.role}")
            report = check_compatibility(base_state, est, ecfg.role)
            compat_reports.append(report)
            _log(f"  {'PASS' if report.passed else 'FAIL'}")
        _dump_json(
            [r.to_dict() for r in compat_reports],
            outdir / "compat_report.json",
        )
        (outdir / "compat_report.md").write_text(
            render_compat_markdown(compat_reports)
        )
        _log(f"wrote {outdir / 'compat_report.md'}")
        failed = [r for r in compat_reports if not r.passed]
        if failed:
            _log(
                f"WARNING: {len(failed)} expert(s) failed the compatibility "
                f"gate; their deltas will still be scored but should not be "
                f"trusted. Roles: {[r.expert_role for r in failed]}"
            )

    # ---- 3. expert full-model norms (for unit-norm normalisation) ----
    if cfg.score.expert_norm == "full_model_unit":
        _log("computing full-model delta norms for each expert ...")
        full_norms = expert_full_model_norms(
            expert_states, base_state, modules=cfg.modules
        )
        for ecfg, n in zip(cfg.experts, full_norms):
            _log(f"  {ecfg.role}: ||θ_i - θ_0||_F = {n:.4f}")
    elif cfg.score.expert_norm == "none":
        full_norms = [1.0 for _ in expert_states]
    else:
        raise ValueError(
            f"unsupported expert_norm: {cfg.score.expert_norm!r} "
            f"(use 'full_model_unit' or 'none')"
        )

    # ---- 4. per-module conflict scoring ----
    L = base_state.num_layers
    if args.max_layers is not None:
        L = min(L, args.max_layers)
        _log(f"smoke test: scoring only the first {L} layers")

    module_conflicts: dict[tuple[int, str], Any] = {}
    cosines_by_module: dict[tuple[int, str], dict[str, float]] = {}
    _log(f"scoring {L} layers × {len(cfg.modules)} modules "
         f"= {L * len(cfg.modules)} (layer, module) pairs")
    for layer in range(L):
        for m in cfg.modules:
            key = module_tensor_key(layer, m)
            base_t = base_state.get(key)  # fp32
            deltas: list[torch.Tensor] = []
            for est, n in zip(expert_states, full_norms):
                e_t = est.get(key)
                d = (e_t - base_t).float()
                if n > 0:
                    d = d / n
                deltas.append(d)
                del e_t
            mc = compute_module_conflict(
                deltas,
                layer=layer,
                module=m,
                tau_mag_percentile=cfg.score.tau_mag_percentile,
                eps=cfg.score.eps,
            )
            module_conflicts[(layer, m)] = mc
            # Same deltas, free diagnostic.
            cosines_by_module[(layer, m)] = pairwise_cosines(deltas)
            del base_t, deltas
        if layer % 4 == 0:
            _log(f"  scored layer {layer}/{L - 1}")

    _dump_json(
        {f"{l}/{m}": mc.to_dict() for (l, m), mc in module_conflicts.items()},
        outdir / "module_conflicts.json",
    )

    # Pairwise cosine diagnostic — "is any expert just noise?"
    cos_summary = aggregate_pairwise_cosines(
        cosines_by_module,
        expert_roles=[e.role for e in cfg.experts],
        modules=cfg.modules,
        num_layers=L,
        module_families=cfg.module_families,
    )
    _dump_json(
        {
            "per_module_raw": {
                f"{l}/{m}": v for (l, m), v in cosines_by_module.items()
            },
            "per_pair_global_mean": cos_summary["per_pair_global_mean"],
            "per_pair_per_layer": cos_summary["per_pair_per_layer"],
            "per_pair_per_family": cos_summary["per_pair_per_family"],
        },
        outdir / "pairwise_cosines.json",
    )

    # ---- 5. standardise + aggregate to per-position ----
    C = standardise_within_scope(
        module_conflicts,
        modules=cfg.modules,
        scope=cfg.score.module_zscore_scope,
        module_families=cfg.module_families,
        eps=cfg.score.eps,
    )
    P = per_position_score(C, num_layers=L, modules=cfg.modules)

    position_payload = {
        "num_layers": L,
        "modules": cfg.modules,
        "zscore_scope": cfg.score.module_zscore_scope,
        "P_pi": P,
        "C_per_module": {f"{l}/{m}": v for (l, m), v in C.items()},
        "rank_by_score": sorted(
            range(len(P)), key=lambda i: -P[i]
        ),
    }
    _dump_json(position_payload, outdir / "position_scores.json")

    # ---- 6. insertion plans for the configured k values ----
    plans = {}
    for k in cfg.position.default_k:
        chosen = insertion_plan(P, k=k, min_gap=cfg.position.min_gap)
        plans[str(k)] = {
            "k": k,
            "min_gap": cfg.position.min_gap,
            "positions": chosen,
            "scores": [P[i] for i in chosen],
        }
    _dump_json(plans, outdir / "insertion_plans.json")

    # ---- 7. weight-norm saliency baseline ----
    _log("computing weight-norm saliency baseline")
    raw_wn = weight_norm_saliency(base_state, modules=cfg.modules)
    raw_wn = {k: v for k, v in raw_wn.items() if k[0] < L}
    P_wn = saliency_position_scores(
        raw_wn,
        num_layers=L,
        modules=cfg.modules,
        module_families=cfg.module_families,
        scope=cfg.score.module_zscore_scope,
    )
    _dump_json(
        {
            "raw_per_module": {f"{l}/{m}": v for (l, m), v in raw_wn.items()},
            "P_pi": P_wn,
        },
        outdir / "weight_norm_saliency.json",
    )

    # ---- 8. agreement diagnostics: how much of CUSP is explained by WN? ----
    _log("computing CUSP-vs-weight-norm agreement diagnostics")
    diag = {
        "spearman_full": spearman(P, P_wn),
        "pearson_full": pearson(P, P_wn),
        # Boundary positions are forced low by P_π construction and high by
        # weight-norm because edge modules tend to have larger weights. The
        # interior slice removes that effect and gives a fairer view of how
        # redundant the two signals are where insertions actually happen.
        "spearman_interior": spearman(P[1:-1], P_wn[1:-1]),
        "pearson_interior": pearson(P[1:-1], P_wn[1:-1]),
        "topk_overlap": {
            str(k): topk_overlap(P, P_wn, k) for k in cfg.position.default_k
        },
        "topk_overlap_interior": {
            str(k): topk_overlap(P[1:-1], P_wn[1:-1], k) for k in cfg.position.default_k
        },
    }
    _dump_json(diag, outdir / "agreement_vs_weight_norm.json")

    # ---- 9. markdown summary ----
    summary_path = outdir / "summary.md"
    summary_path.write_text(
        _render_summary(
            cfg=cfg,
            L=L,
            P=P,
            P_wn=P_wn,
            plans=plans,
            compat_reports=compat_reports,
            full_norms=full_norms,
            module_conflicts=module_conflicts,
            diag=diag,
            cos_summary=cos_summary,
        )
    )
    _log(f"wrote {summary_path}")
    _log("done")
    return 0


# ---------------------------------------------------- markdown summary --


def _render_summary(
    *,
    cfg: Config,
    L: int,
    P: list[float],
    P_wn: list[float],
    plans: dict,
    compat_reports: list,
    full_norms: list[float],
    module_conflicts: dict[tuple[int, str], Any],
    diag: dict,
    cos_summary: dict,
) -> str:
    out: list[str] = []
    out.append(f"# CUSP Phase 1 — `{cfg.name}`")
    out.append("")
    out.append(f"- base: `{cfg.base.repo}@{cfg.base.revision}`")
    for ecfg, n in zip(cfg.experts, full_norms):
        out.append(f"- expert `{ecfg.role}`: `{ecfg.repo}@{ecfg.revision}` — "
                   f"full-model ‖Δ‖ = {n:.4f}")
    out.append(f"- layers scored: {L}")
    out.append(f"- candidate insertion positions: {L + 1}")
    out.append("")

    # Compatibility summary
    out.append("## Compatibility")
    out.append("")
    if compat_reports:
        for r in compat_reports:
            mark = "✅" if r.passed else "❌"
            warn = " ⚠️" if r.has_warnings else ""
            out.append(f"- {mark}{warn} {r.expert_role}")
        if any(r.has_warnings for r in compat_reports):
            out.append("")
            out.append("Warnings are non-fatal — they cover config fields like "
                       "`max_position_embeddings`, `rope_theta`, and "
                       "`rms_norm_eps` that change inference behavior but not "
                       "weight geometry, plus tokenizer added-token differences. "
                       "See `compat_report.md` for the per-field breakdown.")
    else:
        out.append("- (skipped)")
    out.append("")

    # Pairwise expert-delta cosines.
    out.append("## Pairwise expert cosines — is any expert noise?")
    out.append("")
    out.append("After unit-norm normalisation, the cosine between two "
               "experts' deltas at a (layer, module) site tells us whether "
               "their body-module changes have shared direction (>0), "
               "opposite direction (<0), or random / orthogonal (~0). An "
               "expert whose pairwise cosines hover around zero everywhere "
               "is essentially noise; one with systematic non-zero values "
               "contributes real geometric structure, even if its raw ‖Δ‖ "
               "is small.")
    out.append("")
    out.append("| pair | global mean cos | attention | ffn |")
    out.append("|------|-----------------|-----------|-----|")
    fam_means = cos_summary.get("per_pair_per_family", {})
    for label, v in cos_summary["per_pair_global_mean"].items():
        fam = fam_means.get(label, {})
        attn = fam.get("attention", float("nan"))
        ffn = fam.get("ffn", float("nan"))
        out.append(f"| {label} | {v:+.4f} | {attn:+.4f} | {ffn:+.4f} |")
    out.append("")
    out.append("Per-layer mean cosine sparklines (one row per pair):")
    out.append("")
    out.append("```")
    for label, per_layer in cos_summary["per_pair_per_layer"].items():
        sparkline = position_score_curve(per_layer)
        first = per_layer[0] if per_layer else 0.0
        last = per_layer[-1] if per_layer else 0.0
        out.append(f"{label:>20s} : {sparkline}   "
                   f"[L0={first:+.3f}, L{len(per_layer)-1}={last:+.3f}]")
    out.append("```")
    out.append("")
    out.append("Reading: pairs with |global mean cos| ≥ 0.05 carry real "
               "structure; pairs with |global mean cos| < 0.02 across both "
               "module families are statistically consistent with noise. "
               "Negative values mean the two experts pull the same module "
               "in opposite directions — that is the geometric signature of "
               "parameter conflict.")
    out.append("")

    # Agreement vs weight-norm baseline.
    out.append("## Agreement with weight-norm baseline")
    out.append("")
    out.append("How much of the conflict score is already explained by raw "
               "weight magnitude on the base? Spearman is rank-based and ignores "
               "the scale difference between the two signals.")
    out.append("")
    out.append("| slice | Spearman | Pearson |")
    out.append("|-------|----------|---------|")
    out.append(f"| full (incl. boundary)   | {diag['spearman_full']:+.3f} | "
               f"{diag['pearson_full']:+.3f} |")
    out.append(f"| interior (drop π=0, π=L) | {diag['spearman_interior']:+.3f} | "
               f"{diag['pearson_interior']:+.3f} |")
    out.append("")
    out.append("Reading: Spearman close to +1 means weight-norm already picks the "
               "same positions; close to 0 means the two signals are essentially "
               "independent; negative means CUSP systematically prefers what WN "
               "deprioritises (this is what we want at the boundaries).")
    out.append("")
    for k_str, ov in diag["topk_overlap"].items():
        out.append(f"- top-{k_str} (full): "
                   f"{ov['shared_count']} / {ov['k']} shared, "
                   f"intersection={ov['intersection']}, "
                   f"CUSP-only={ov['x_only']}, WN-only={ov['y_only']}")
    for k_str, ov in diag["topk_overlap_interior"].items():
        out.append(f"- top-{k_str} (interior): "
                   f"{ov['shared_count']} / {ov['k']} shared, "
                   f"intersection={ov['intersection']}, "
                   f"CUSP-only={ov['x_only']}, WN-only={ov['y_only']}")
    out.append("")

    # Position score curve
    out.append("## Position scores")
    out.append("")
    out.append(f"```")
    out.append(f"P_π : {position_score_curve(P)}")
    out.append(f"WN  : {position_score_curve(P_wn)}")
    out.append(f"```")
    out.append("")
    out.append("| π | P_π (CUSP) | P_π (weight-norm) | rank |")
    out.append("|---|------------|--------------------|------|")
    rank = {idx: r for r, idx in enumerate(sorted(range(len(P)), key=lambda i: -P[i]))}
    for pi, (p, wn) in enumerate(zip(P, P_wn)):
        out.append(f"| {pi} | {p:+.4f} | {wn:+.4f} | {rank[pi]+1} |")
    out.append("")

    # Insertion plans
    out.append("## Proposed insertion plans")
    out.append("")
    for k_str, plan in plans.items():
        positions = plan["positions"]
        out.append(f"### k = {k_str} (min_gap = {plan['min_gap']})")
        out.append("")
        out.append(f"- positions: {positions}")
        out.append(f"- P_π at chosen positions: "
                   f"{[round(s, 4) for s in plan['scores']]}")
        out.append("")

    # Top contributing modules at the top positions
    out.append("## Top per-module conflict at the top-scoring positions")
    out.append("")
    top_positions = sorted(range(len(P)), key=lambda i: -P[i])[:3]
    for pi in top_positions:
        left = pi - 1 if pi >= 1 else None
        right = pi if pi <= L - 1 else None
        out.append(f"### π = {pi}  (left layer = {left}, right layer = {right})")
        out.append("")
        out.append("| layer | module | D | S | energy |")
        out.append("|-------|--------|---|---|--------|")
        rows: list[tuple[int, str, float, float, float]] = []
        for layer in (left, right):
            if layer is None:
                continue
            for m in cfg.modules:
                mc = module_conflicts.get((layer, m))
                if mc is None:
                    continue
                rows.append((layer, m, mc.D, mc.S, mc.energy))
        # sort by D*S as a quick proxy without re-standardising
        rows.sort(key=lambda r: -(r[2] * r[3]))
        for row in rows[:6]:
            out.append(
                f"| {row[0]} | `{row[1]}` | {row[2]:.4f} | {row[3]:.4f} | {row[4]:.2e} |"
            )
        out.append("")

    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
