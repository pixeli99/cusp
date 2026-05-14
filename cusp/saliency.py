"""Saliency baselines that can be computed offline from weights alone.

This file covers the cheap baselines that need no calibration corpus and no
GPU. The four that *do* need a forward / backward pass (gradient norm,
empirical Fisher, activation norm, inverse-pruning sensitivity) live in
`scripts/run_saliency_gpu.py` so they can be run separately once a corpus is
ready.

Implemented here:
  - weight_norm: per-module ||W||_F on the base model.

Each saliency produces a value per (layer, module) on the same modules as
the conflict score, then aggregates to per-position with the same
flank-averaging rule (so all baselines and the headline are compared on the
same positional grid).
"""

from __future__ import annotations

from typing import Iterator

import torch

from cusp.io import ModelState, module_tensor_key
from cusp.aggregate import per_position_score, standardise_within_scope
from cusp.conflict import ModuleConflict


def weight_norm_saliency(
    base: ModelState, modules: list[str]
) -> dict[tuple[int, str], float]:
    """Raw ||W||_F per (layer, module) on the base. The conflict pipeline
    then standardises this on the same scope as the headline score."""
    out: dict[tuple[int, str], float] = {}
    for layer in range(base.num_layers):
        for m in modules:
            w = base.get(module_tensor_key(layer, m))
            out[(layer, m)] = float(w.float().norm().item())
    return out


def saliency_to_pseudo_conflict(
    raw: dict[tuple[int, str], float]
) -> dict[tuple[int, str], ModuleConflict]:
    """Wrap a scalar saliency into a ModuleConflict-shaped dict so we can
    reuse per_position_score / standardise_within_scope unchanged.

    We map the scalar to both `D` and `S` so that `z(D)·z(S)` becomes the
    z-scored saliency squared with a sign flip when the value is below
    mean — equivalent up to a monotone transform to z-scoring the saliency
    directly. Callers that want a different aggregation should bypass this
    helper and standardise their own scalar.
    """
    out: dict[tuple[int, str], ModuleConflict] = {}
    for (l, m), v in raw.items():
        out[(l, m)] = ModuleConflict(
            layer=l,
            module=m,
            D=float(v),
            S=float(v),
            energy=float(v),
            per_expert_norm=[],
            n_above_threshold=0,
            tau_mag=0.0,
        )
    return out


def saliency_position_scores(
    raw: dict[tuple[int, str], float],
    num_layers: int,
    modules: list[str],
    module_families: dict[str, list[str]],
    scope: str = "module",
) -> list[float]:
    """End-to-end: raw scalar saliency -> standardised C -> P_π."""
    mc = saliency_to_pseudo_conflict(raw)
    C = standardise_within_scope(mc, modules=modules, scope=scope, module_families=module_families)
    return per_position_score(C, num_layers, modules)
