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
    insertion_plan,
    per_position_score,
    position_score_curve,
    standardise_within_scope,
)
from cusp.compat import check_compatibility, render_compat_markdown  # noqa: E402
from cusp.conflict import (  # noqa: E402
    compute_module_conflict,
    expert_full_model_norms,
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
            del base_t, deltas
        if layer % 4 == 0:
            _log(f"  scored layer {layer}/{L - 1}")

    _dump_json(
        {f"{l}/{m}": mc.to_dict() for (l, m), mc in module_conflicts.items()},
        outdir / "module_conflicts.json",
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

    # ---- 8. markdown summary ----
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
            out.append(f"- {mark} {r.expert_role}")
    else:
        out.append("- (skipped)")
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
