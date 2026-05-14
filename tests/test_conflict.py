"""Unit tests for the conflict score and per-position aggregation.

These run on synthetic tensors so they need no HF download. They cover the
properties the score is supposed to have: when all experts agree, D and S are
0; when experts disagree at every coordinate, S is ~0.5; the aggregation
flank-averages two layers and halves boundary positions.

Run from the code/ directory:
    PYTHONPATH=. python tests/test_conflict.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Make `cusp.*` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from cusp.aggregate import (  # noqa: E402
    insertion_plan,
    pearson,
    per_position_score,
    spearman,
    standardise_within_scope,
    topk_overlap,
)
from cusp.conflict import ModuleConflict, compute_module_conflict  # noqa: E402


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return math.isclose(a, b, abs_tol=tol)


# -------------------------------------------------------------------- D, S


def test_all_agree_gives_zero_D_and_low_S() -> None:
    # Three identical deltas: residual energy is exactly zero, sign
    # disagreement is zero.
    d = torch.randn(64, 32)
    mc = compute_module_conflict([d.clone(), d.clone(), d.clone()], layer=0, module="q_proj")
    assert approx(mc.D, 0.0), f"D = {mc.D} should be 0"
    assert approx(mc.S, 0.0), f"S = {mc.S} should be 0"


def test_opposite_signs_give_high_S() -> None:
    # Two experts with deltas that are exact negatives of each other. After
    # subtracting their mean (which is zero), the residual is the original
    # tensor itself, so D = 1.0. Every coordinate disagrees on sign, so S =
    # 0.5 (since K=2: one expert says +, the other says -, max majority = 1/2,
    # 1 - 1/2 = 0.5).
    d = torch.randn(64, 32).abs() + 0.01  # avoid zeros on the boundary
    mc = compute_module_conflict([d, -d], layer=0, module="q_proj")
    assert approx(mc.D, 1.0, tol=1e-4), f"D = {mc.D} should be ~1.0"
    assert approx(mc.S, 0.5, tol=1e-3), f"S = {mc.S} should be ~0.5"


def test_orthogonal_deltas_give_high_D() -> None:
    # K experts whose non-zero supports are disjoint coordinates each of unit
    # norm. Exact algebra: D = (K - 1) / K. So K = 3 gives 2/3, K = 4 gives
    # 3/4. We test with K = 3.
    K = 3
    rows = 64
    cols = 32
    deltas = []
    for k in range(K):
        d = torch.zeros(rows, cols)
        d[k, :] = 1.0  # each expert lives in a disjoint row
        deltas.append(d)
    mc = compute_module_conflict(deltas, layer=0, module="q_proj",
                                  tau_mag_percentile=0.5)
    expected_D = (K - 1) / K
    assert approx(mc.D, expected_D, tol=1e-4), \
        f"D = {mc.D}, expected ~{expected_D} for K={K} orthogonal supports"


def test_normalization_invariance_of_D() -> None:
    # Scaling all deltas by the same constant does not change D, because the
    # numerator and denominator scale together.
    deltas = [torch.randn(32, 16) for _ in range(3)]
    mc1 = compute_module_conflict(deltas, layer=0, module="up_proj")
    scaled = [d * 7.5 for d in deltas]
    mc2 = compute_module_conflict(scaled, layer=0, module="up_proj")
    assert approx(mc1.D, mc2.D, tol=1e-5), f"D not scale-invariant: {mc1.D} vs {mc2.D}"


# ----------------------------------------------------- standardise + P_π


def _toy_conflicts() -> dict[tuple[int, str], ModuleConflict]:
    """A 4-layer toy: high conflict at layer 1, low elsewhere."""
    mc: dict[tuple[int, str], ModuleConflict] = {}
    for layer in range(4):
        for m in ["q_proj", "up_proj"]:
            if layer == 1:
                d_val, s_val = 0.9, 0.4
            else:
                d_val, s_val = 0.1, 0.05
            mc[(layer, m)] = ModuleConflict(
                layer=layer,
                module=m,
                D=d_val,
                S=s_val,
                energy=1.0,
                per_expert_norm=[1.0, 1.0, 1.0],
                n_above_threshold=10,
                tau_mag=0.0,
            )
    return mc


def test_standardisation_puts_outlier_layer_on_top() -> None:
    mc = _toy_conflicts()
    C = standardise_within_scope(
        mc,
        modules=["q_proj", "up_proj"],
        scope="module",
    )
    # Layer 1 should have the largest C because its D and S are both far above
    # the within-module mean. With one outlier and three identical low values,
    # z-scores are (-0.577, +1.732, -0.577, -0.577) for the D row when the
    # outlier is at position 1.
    layer1_C = sum(C[(1, m)] for m in ["q_proj", "up_proj"])
    other_C = sum(C[(l, m)] for l in [0, 2, 3] for m in ["q_proj", "up_proj"])
    assert layer1_C > other_C, "outlier layer should dominate after standardisation"


def test_per_position_aggregation_boundary_halving() -> None:
    """Boundary positions are systematically half-magnitude because only one
    flank exists. Interior positions average both flanks."""
    # 4 layers, 1 module, every C = 2.0.
    C = {(layer, "q_proj"): 2.0 for layer in range(4)}
    P = per_position_score(C, num_layers=4, modules=["q_proj"])
    # π=0: only right flank (layer 0), so 0.5 * 2.0 = 1.0
    # π=4: only left flank (layer 3), so 0.5 * 2.0 = 1.0
    # π in 1..3: both flanks contribute 2.0+2.0=4.0, halved to 2.0.
    assert P == [1.0, 2.0, 2.0, 2.0, 1.0], f"unexpected P_π: {P}"


def test_insertion_plan_respects_min_gap() -> None:
    scores = [5.0, 4.9, 4.8, 0.1, 4.7]
    chosen = insertion_plan(scores, k=3, min_gap=2)
    # Greedy: pick 0 (5.0), then 2 (4.8, gap 2 ok), then 4 (4.7, gap 2 ok).
    # Skips 1 (4.9, gap 1 to 0) and 3 (gap 1 to 2 and 4).
    assert chosen == [0, 2, 4], f"got {chosen}"


def test_insertion_plan_underfills_when_gap_too_tight() -> None:
    scores = [5.0, 4.5, 4.0]
    # min_gap=5 with only 3 positions means we can fit at most 1.
    chosen = insertion_plan(scores, k=3, min_gap=5)
    assert chosen == [0]


# -------------------------------------------------- correlations + overlap


def test_pearson_identical_is_one() -> None:
    x = [1.0, 2.0, 3.0, 4.5, 7.1]
    assert approx(pearson(x, x), 1.0, tol=1e-9)


def test_pearson_negated_is_minus_one() -> None:
    x = [1.0, 2.0, 3.0, 4.5, 7.1]
    y = [-v for v in x]
    assert approx(pearson(x, y), -1.0, tol=1e-9)


def test_pearson_constant_is_nan() -> None:
    x = [1.0, 2.0, 3.0]
    y = [5.0, 5.0, 5.0]
    assert math.isnan(pearson(x, y))


def test_spearman_monotone_nonlinear_is_one() -> None:
    # x and x ** 3 share the same rank order, so Spearman = 1 even though
    # Pearson is < 1.
    x = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    y = [xi ** 3 for xi in x]
    assert approx(spearman(x, y), 1.0, tol=1e-9)
    assert pearson(x, y) < 1.0


def test_spearman_ties_share_average_rank() -> None:
    # Two ties at rank 2.5 each.
    x = [1.0, 2.0, 2.0, 4.0]
    y = [10.0, 20.0, 20.0, 40.0]
    # Same tied structure -> Spearman = 1.
    assert approx(spearman(x, y), 1.0, tol=1e-9)


def test_topk_overlap_full_match() -> None:
    x = [5.0, 4.0, 3.0, 2.0, 1.0]
    y = [50.0, 40.0, 30.0, 20.0, 10.0]
    r = topk_overlap(x, y, k=3)
    assert r["intersection"] == [0, 1, 2]
    assert r["x_only"] == [] and r["y_only"] == []
    assert r["jaccard"] == 1.0


def test_topk_overlap_disjoint() -> None:
    x = [5.0, 4.0, 0.0, 0.0]
    y = [0.0, 0.0, 5.0, 4.0]
    r = topk_overlap(x, y, k=2)
    assert r["intersection"] == []
    assert r["x_only"] == [0, 1]
    assert r["y_only"] == [2, 3]
    assert r["jaccard"] == 0.0


def test_topk_overlap_partial() -> None:
    x = [5.0, 4.0, 3.0, 2.0, 1.0]
    y = [5.0, 1.0, 2.0, 3.0, 4.0]  # top-3 of y is [0, 4, 3]
    r = topk_overlap(x, y, k=3)
    assert r["intersection"] == [0]
    assert sorted(r["x_only"]) == [1, 2]
    assert sorted(r["y_only"]) == [3, 4]
    assert approx(r["jaccard"], 1 / 5, tol=1e-9)


# -------------------------------------------------------------------- main


def main() -> int:
    tests = [
        test_all_agree_gives_zero_D_and_low_S,
        test_opposite_signs_give_high_S,
        test_orthogonal_deltas_give_high_D,
        test_normalization_invariance_of_D,
        test_standardisation_puts_outlier_layer_on_top,
        test_per_position_aggregation_boundary_halving,
        test_insertion_plan_respects_min_gap,
        test_insertion_plan_underfills_when_gap_too_tight,
        test_pearson_identical_is_one,
        test_pearson_negated_is_minus_one,
        test_pearson_constant_is_nan,
        test_spearman_monotone_nonlinear_is_one,
        test_spearman_ties_share_average_rank,
        test_topk_overlap_full_match,
        test_topk_overlap_disjoint,
        test_topk_overlap_partial,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failures.append((t.__name__, str(e)))
        except Exception as e:
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")
            failures.append((t.__name__, repr(e)))
    print()
    if failures:
        print(f"{len(failures)} failed / {len(tests)} total")
        return 1
    print(f"all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
