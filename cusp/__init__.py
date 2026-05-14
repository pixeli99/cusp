"""CUSP — Conflict-Aware Up-Scaling Placement.

Phase 1 library: compatibility gate, per-module conflict score, per-position
aggregation, insertion plan, weight-norm saliency. No training, no forward
pass; pure weight-space arithmetic.
"""

from cusp.compat import CompatReport, check_compatibility
from cusp.conflict import ModuleConflict, compute_module_conflict
from cusp.aggregate import per_position_score, insertion_plan
from cusp.io import Config, load_config, load_model_state, ModuleTensor
from cusp.saliency import weight_norm_saliency

__all__ = [
    "CompatReport",
    "check_compatibility",
    "ModuleConflict",
    "compute_module_conflict",
    "per_position_score",
    "insertion_plan",
    "Config",
    "load_config",
    "load_model_state",
    "ModuleTensor",
    "weight_norm_saliency",
]
