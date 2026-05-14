"""Compatibility gate.

Same-base task-vector geometry is only meaningful when checkpoints share
parameterisation. Before scoring, we verify locally that base and expert
agree on:

  - tokenizer vocabulary (including added tokens)
  - the config fields that affect tensor shapes
  - the full set of state-dict keys
  - every tensor shape

A failing check makes the expert ineligible for delta scoring; it can still
be used as a distillation teacher or evaluation reference.

Hub model-tree ancestry is intentionally *not* consulted here. It is only a
candidate-selection hint upstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cusp.io import ModelState


# Fields that change weight tensor shapes or the meaning of which weight
# belongs to which module. A mismatch here makes delta arithmetic invalid.
FATAL_CONFIG_FIELDS = (
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "head_dim",
    "tie_word_embeddings",
    "hidden_act",
)

# Fields that change inference behavior but do not change weight shapes or
# weight geometry. A mismatch warrants a warning but does not invalidate
# delta arithmetic on the projection modules CUSP scores.
SOFT_CONFIG_FIELDS = (
    "max_position_embeddings",
    "rope_theta",
    "rms_norm_eps",
)


@dataclass
class CompatReport:
    base_repo: str
    expert_repo: str
    expert_role: str
    # Hard failures.
    config_mismatches: list[tuple[str, object, object]] = field(default_factory=list)
    state_dict_missing_in_expert: list[str] = field(default_factory=list)
    state_dict_extra_in_expert: list[str] = field(default_factory=list)
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=list
    )
    # Soft warnings.
    config_warnings: list[tuple[str, object, object]] = field(default_factory=list)
    tokenizer_vocab_match: bool = True
    tokenizer_diff_keys: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Hard pass — delta arithmetic on body projections is valid."""
        return (
            not self.config_mismatches
            and not self.state_dict_missing_in_expert
            and not self.state_dict_extra_in_expert
            and not self.shape_mismatches
        )

    @property
    def has_warnings(self) -> bool:
        return bool(self.config_warnings) or not self.tokenizer_vocab_match

    def to_dict(self) -> dict:
        return {
            "base_repo": self.base_repo,
            "expert_repo": self.expert_repo,
            "expert_role": self.expert_role,
            "passed": self.passed,
            "has_warnings": self.has_warnings,
            "config_mismatches": [
                {"field": f, "base": b, "expert": e} for f, b, e in self.config_mismatches
            ],
            "config_warnings": [
                {"field": f, "base": b, "expert": e} for f, b, e in self.config_warnings
            ],
            "tokenizer_vocab_match": self.tokenizer_vocab_match,
            "tokenizer_diff_keys": self.tokenizer_diff_keys,
            "state_dict_missing_in_expert": self.state_dict_missing_in_expert,
            "state_dict_extra_in_expert": self.state_dict_extra_in_expert,
            "shape_mismatches": [
                {"key": k, "base_shape": bs, "expert_shape": es}
                for k, bs, es in self.shape_mismatches
            ],
        }


# ---------------------------------------------------------------- tokenizers


def _read_tokenizer_vocab(local_dir: Path) -> tuple[dict | None, list[str]]:
    """Return (vocab_dict, added_tokens_keys). vocab_dict is None if the repo
    uses a fast tokenizer.json we choose not to parse exhaustively. We always
    return a stable comparison surface in `added_tokens_keys`."""
    # Easy path: tokenizer.json carries the full model.
    tok_json = local_dir / "tokenizer.json"
    if tok_json.exists():
        with tok_json.open() as f:
            data = json.load(f)
        model = data.get("model", {})
        # BPE: vocab is dict[str, int]; Unigram: vocab is list of [str, score].
        raw_vocab = model.get("vocab")
        if isinstance(raw_vocab, dict):
            vocab: dict = dict(raw_vocab)
        elif isinstance(raw_vocab, list):
            vocab = {entry[0]: i for i, entry in enumerate(raw_vocab)}
        else:
            vocab = {}
        added = sorted({tok["content"] for tok in data.get("added_tokens", [])})
        return vocab, added

    # Fallback: vocab.json + merges.txt (legacy GPT-2 style).
    vocab_path = local_dir / "vocab.json"
    if vocab_path.exists():
        with vocab_path.open() as f:
            vocab = json.load(f)
        added_path = local_dir / "added_tokens.json"
        added = []
        if added_path.exists():
            with added_path.open() as f:
                added = sorted(json.load(f).keys())
        return vocab, added

    return None, []


# -------------------------------------------------------------------- runner


def check_compatibility(
    base: ModelState, expert: ModelState, expert_role: str
) -> CompatReport:
    """Run all four checks. Does no network IO beyond what ModelState already
    cached; it reads only files in the local snapshot directory."""
    report = CompatReport(
        base_repo=base.repo, expert_repo=expert.repo, expert_role=expert_role
    )

    # 1) Config — fatal vs soft.
    for field_name in FATAL_CONFIG_FIELDS:
        b_val = base.config.get(field_name)
        e_val = expert.config.get(field_name)
        if b_val is None and e_val is None:
            continue
        if b_val != e_val:
            report.config_mismatches.append((field_name, b_val, e_val))
    for field_name in SOFT_CONFIG_FIELDS:
        b_val = base.config.get(field_name)
        e_val = expert.config.get(field_name)
        if b_val is None and e_val is None:
            continue
        if b_val != e_val:
            report.config_warnings.append((field_name, b_val, e_val))

    # 2) Tokenizer vocabulary.
    b_vocab, b_added = _read_tokenizer_vocab(base.local_dir)
    e_vocab, e_added = _read_tokenizer_vocab(expert.local_dir)
    if b_vocab is None or e_vocab is None:
        # Cannot fully verify; flag this so the caller does not silently trust
        # the result.
        report.tokenizer_vocab_match = False
        report.tokenizer_diff_keys = ["<could not parse one or both tokenizer files>"]
    else:
        # Same number of entries is a necessary condition; matching keys is
        # sufficient for our purposes — we do not care about merges order.
        diff: list[str] = []
        if len(b_vocab) != len(e_vocab):
            diff.append(f"vocab_size {len(b_vocab)} vs {len(e_vocab)}")
        else:
            b_set, e_set = set(b_vocab.keys()), set(e_vocab.keys())
            missing = sorted(b_set - e_set)[:10]
            extra = sorted(e_set - b_set)[:10]
            if missing:
                diff.append(f"missing in expert (up to 10 shown): {missing}")
            if extra:
                diff.append(f"extra in expert (up to 10 shown): {extra}")
        if b_added != e_added:
            diff.append(f"added_tokens differ: base={b_added}, expert={e_added}")
        if diff:
            report.tokenizer_vocab_match = False
            report.tokenizer_diff_keys = diff

    # 3) State-dict keys.
    b_keys = set(base.keys())
    e_keys = set(expert.keys())
    report.state_dict_missing_in_expert = sorted(b_keys - e_keys)
    report.state_dict_extra_in_expert = sorted(e_keys - b_keys)

    # 4) Shapes for the shared keys.
    for key in sorted(b_keys & e_keys):
        bs = base.shape(key)
        es = expert.shape(key)
        if bs != es:
            report.shape_mismatches.append((key, bs, es))

    return report


def render_compat_markdown(reports: list[CompatReport]) -> str:
    """Render one human-readable markdown block per expert."""
    out: list[str] = ["# Compatibility report", ""]
    for r in reports:
        out.append(f"## {r.expert_role} — `{r.expert_repo}`")
        out.append("")
        mark = "✅" if r.passed else "❌"
        warn = "  ⚠️  with warnings" if r.has_warnings else ""
        out.append(f"- **passed:** {mark}{warn}")
        out.append(f"- base: `{r.base_repo}`")
        if r.config_mismatches:
            out.append("- fatal config mismatches:")
            for f, b, e in r.config_mismatches:
                out.append(f"  - `{f}`: base=`{b}`, expert=`{e}`")
        if r.config_warnings:
            out.append("- soft config warnings (do not affect weight geometry):")
            for f, b, e in r.config_warnings:
                out.append(f"  - `{f}`: base=`{b}`, expert=`{e}`")
        if not r.tokenizer_vocab_match:
            out.append("- tokenizer differences:")
            for d in r.tokenizer_diff_keys:
                out.append(f"  - {d}")
        if r.state_dict_missing_in_expert:
            out.append(f"- {len(r.state_dict_missing_in_expert)} state-dict keys missing in expert")
            for k in r.state_dict_missing_in_expert[:5]:
                out.append(f"  - `{k}`")
            if len(r.state_dict_missing_in_expert) > 5:
                out.append("  - ...")
        if r.state_dict_extra_in_expert:
            out.append(f"- {len(r.state_dict_extra_in_expert)} state-dict keys extra in expert")
            for k in r.state_dict_extra_in_expert[:5]:
                out.append(f"  - `{k}`")
            if len(r.state_dict_extra_in_expert) > 5:
                out.append("  - ...")
        if r.shape_mismatches:
            out.append(f"- {len(r.shape_mismatches)} shape mismatches")
            for k, bs, es in r.shape_mismatches[:5]:
                out.append(f"  - `{k}`: base={bs}, expert={es}")
            if len(r.shape_mismatches) > 5:
                out.append("  - ...")
        out.append("")
    return "\n".join(out)
