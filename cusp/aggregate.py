"""Per-position aggregation and the greedy top-k insertion plan.

Module conflicts live on (layer, module) pairs. The depth-up-scaling operator
inserts new transformer blocks between consecutive layers, so the relevant
unit is an insertion *position* π ∈ {0, 1, ..., L}: π = 0 inserts before
layer 0, π = L inserts after the final layer L-1. The per-position score
aggregates module conflicts on the two layers flanking π, following the
formula in Section 4.1 of the paper:

    P_π = 0.5 · Σ_m ( 1[π ≥ 1] · C_{π-1,m}  +  1[π ≤ L-1] · C_{π,m} )

Boundary positions are systematically lower because only one flank exists,
which matches the intuition that inserting at the very top or very bottom of
the stack discards information from the missing flank.
"""

from __future__ import annotations

import math
from typing import Mapping

from cusp.conflict import ModuleConflict


# ----------------------------------------------------------- standardisation


def _zscore(values: list[float], eps: float = 1e-12) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(len(values), 1)
    std = math.sqrt(var)
    return [(v - mean) / (std + eps) for v in values]


def standardise_within_scope(
    conflicts: Mapping[tuple[int, str], ModuleConflict],
    modules: list[str],
    scope: str,
    module_families: Mapping[str, list[str]] | None = None,
    eps: float = 1e-12,
) -> dict[tuple[int, str], float]:
    """Return C_{ℓ,m} = z(D_{ℓ,m}) * z(S_{ℓ,m}) standardised within the
    requested scope.

    scope ∈ {module, family, global}:
      - module: standardise across all layers of one module name (most
        conservative — corrects for module-name norm differences).
      - family: standardise across all modules in one family (attention vs FFN).
      - global: one mean / std over the whole network.

    The default in the paper is `module`.
    """
    # Index modules into scope groups.
    if scope == "module":
        groups: dict[str, list[tuple[int, str]]] = {m: [] for m in modules}
        for (l, m), _ in conflicts.items():
            if m in groups:
                groups[m].append((l, m))
    elif scope == "family":
        if module_families is None:
            raise ValueError("scope=family requires module_families")
        groups = {fam: [] for fam in module_families}
        rev = {m: fam for fam, ms in module_families.items() for m in ms}
        for (l, m), _ in conflicts.items():
            fam = rev.get(m)
            if fam is not None:
                groups[fam].append((l, m))
    elif scope == "global":
        groups = {"all": list(conflicts.keys())}
    else:
        raise ValueError(f"unknown standardisation scope: {scope!r}")

    C: dict[tuple[int, str], float] = {}
    for keys in groups.values():
        if not keys:
            continue
        D_vals = [conflicts[k].D for k in keys]
        S_vals = [conflicts[k].S for k in keys]
        D_z = _zscore(D_vals, eps=eps)
        S_z = _zscore(S_vals, eps=eps)
        for k, dz, sz in zip(keys, D_z, S_z):
            C[k] = dz * sz
    return C


# ----------------------------------------------------------- per-position P_π


def per_position_score(
    C: Mapping[tuple[int, str], float],
    num_layers: int,
    modules: list[str],
) -> list[float]:
    """Aggregate per-module C to per-position P_π for π in [0, L].

    Returns a list of length L+1.
    """
    P: list[float] = []
    for pi in range(num_layers + 1):
        s = 0.0
        if pi >= 1:  # left flank: layer pi - 1
            for m in modules:
                s += C.get((pi - 1, m), 0.0)
        if pi <= num_layers - 1:  # right flank: layer pi
            for m in modules:
                s += C.get((pi, m), 0.0)
        P.append(0.5 * s)
    return P


# ---------------------------------------------------------- insertion plan


def insertion_plan(scores: list[float], k: int, min_gap: int) -> list[int]:
    """Greedy top-k under a minimum-gap constraint.

    Walks positions in descending score order; accepts a position iff it is
    at least `min_gap` away from every already-accepted position. Returns the
    selected positions in ascending order.

    If `min_gap` is too aggressive to admit `k` positions, returns as many as
    fit and lets the caller decide whether to relax the constraint.
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    chosen: list[int] = []
    for idx in order:
        if all(abs(idx - j) >= min_gap for j in chosen):
            chosen.append(idx)
        if len(chosen) >= k:
            break
    return sorted(chosen)


def position_score_curve(P: list[float]) -> str:
    """ASCII sparkline of P_π for the markdown summary. Just makes the curve
    eye-checkable next to the per-position table."""
    if not P:
        return ""
    lo, hi = min(P), max(P)
    span = hi - lo
    if span <= 0:
        return "─" * len(P)
    ramp = " ▁▂▃▄▅▆▇█"
    out = []
    for v in P:
        bucket = int(((v - lo) / span) * (len(ramp) - 1))
        out.append(ramp[bucket])
    return "".join(out)
