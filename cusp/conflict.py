"""Per-module conflict score.

Given K expert deltas Δ_i (each a fp32 flattened tensor of the same length),
we compute:

  - residual energy fraction D, the share of delta energy left after removing
    the consensus update;
  - sign disagreement S over coordinates above a magnitude percentile, the
    fraction of non-trivial coordinates on which the experts split sign.

Both are scalars in [0, 1]. The product after within-family standardisation is
the module conflict C in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass
class ModuleConflict:
    layer: int
    module: str
    D: float            # residual energy fraction
    S: float            # sign disagreement on the magnitude-thresholded mask
    energy: float       # sum_i ||Δ_i||² (for diagnostics)
    per_expert_norm: list[float]  # ||Δ_i|| per expert (after normalisation)
    n_above_threshold: int        # |M_{ℓ,m}|, number of coords used for S
    tau_mag: float                # the magnitude threshold actually used

    def to_dict(self) -> dict:
        return asdict(self)


def compute_module_conflict(
    deltas: list[torch.Tensor],
    layer: int,
    module: str,
    tau_mag_percentile: float = 0.5,
    eps: float = 1e-12,
) -> ModuleConflict:
    """Compute D, S, and diagnostics for one (layer, module) pair.

    deltas: list of K tensors, each fp32 on CPU, identical shape. These are
    the per-expert deltas *after* whatever normalisation policy was selected
    upstream (e.g. unit full-model norm). This function does not re-normalise.

    The choice of fp32 is load-bearing: bf16 norms saturate well below the
    energies we care about for popular fine-tunes.
    """
    K = len(deltas)
    if K < 2:
        raise ValueError("need at least 2 expert deltas to score conflict")

    stacked = torch.stack([d.reshape(-1) for d in deltas], dim=0)  # (K, N)
    if stacked.dtype != torch.float32:
        stacked = stacked.float()
    N = stacked.shape[1]

    # ---------- residual energy ----------
    mean = stacked.mean(dim=0)  # (N,)
    residual = stacked - mean   # (K, N)
    res_energy = (residual * residual).sum().item()
    total_energy = (stacked * stacked).sum().item()
    D = float(res_energy / (total_energy + eps))

    per_expert_norm = stacked.norm(dim=1).tolist()  # K-vector

    # ---------- sign disagreement on the magnitude-thresholded mask ----------
    abs_max_per_coord = stacked.abs().max(dim=0).values  # (N,)
    # quantile is exact on CPU; on huge tensors torch may complain, so do an
    # explicit clamp for safety.
    q = float(torch.quantile(abs_max_per_coord, tau_mag_percentile).item())
    mask = abs_max_per_coord >= q
    n_above = int(mask.sum().item())
    if n_above == 0 or q == 0.0:
        S = 0.0
    else:
        sub = stacked[:, mask]  # (K, n_above)
        signs = sub.sign()      # values in {-1, 0, +1}
        n_pos = (signs > 0).sum(dim=0)
        n_neg = (signs < 0).sum(dim=0)
        max_agree = torch.maximum(n_pos, n_neg).float() / K  # (n_above,)
        S = float((1.0 - max_agree).mean().item())

    return ModuleConflict(
        layer=layer,
        module=module,
        D=D,
        S=S,
        energy=float(total_energy),
        per_expert_norm=[float(x) for x in per_expert_norm],
        n_above_threshold=n_above,
        tau_mag=q,
    )


# ---------------------------------------------- pairwise cosine diagnostic


def pairwise_cosines(deltas: list[torch.Tensor]) -> dict[str, float]:
    """Cosine similarity between every ordered pair (i, j) with i < j.

    Keyed as "{i}_{j}" using positional indices into `deltas`. Callers map
    positional indices back to expert roles upstream.

    A near-zero pairwise cosine over many (layer, module) pairs is the
    signature of "this expert is noise" — an expert whose body-module
    direction is essentially random under the unit-norm normalisation will
    show cos ≈ 0 with the other experts at almost every site. Systematic
    non-zero values (positive or negative) indicate real geometric
    structure, even if the absolute delta magnitude is small.
    """
    K = len(deltas)
    out: dict[str, float] = {}
    if K < 2:
        return out
    flat = [d.reshape(-1).float() for d in deltas]
    # Inline norms in fp32 for the same reason we compute D in fp32 — bf16
    # underflow on small deltas (Instruct) makes pairwise cosines unreliable.
    norms = [float((f * f).sum().sqrt().item()) for f in flat]
    for i in range(K):
        for j in range(i + 1, K):
            denom = norms[i] * norms[j]
            if denom > 0.0:
                dot = float((flat[i] * flat[j]).sum().item())
                out[f"{i}_{j}"] = dot / denom
            else:
                out[f"{i}_{j}"] = 0.0
    return out


# --------------------------------------------------------------------- helper


def expert_full_model_norms(
    experts_state: list,           # list[ModelState]
    base_state,                    # ModelState
    modules: list[str],
) -> list[float]:
    """Compute the Frobenius norm of each expert's full-model delta (against
    the base) summed across the listed modules. Used for unit-norm
    normalisation upstream of conflict scoring.

    Iterates module-by-module and adds squared norms, so peak memory is the
    size of one tensor per checkpoint at a time.
    """
    from cusp.io import module_tensor_key

    sq = [0.0 for _ in experts_state]
    for layer in range(base_state.num_layers):
        for m in modules:
            key = module_tensor_key(layer, m)
            base_t = base_state.get(key)
            for i, e in enumerate(experts_state):
                e_t = e.get(key)
                d = (e_t - base_t).float()
                sq[i] += float((d * d).sum().item())
                del e_t, d
            del base_t
    return [s ** 0.5 for s in sq]
