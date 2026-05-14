"""Config loading and lazy state-dict access.

We do not call AutoModelForCausalLM.from_pretrained for Phase 1 because we
never need the full nn.Module — just per-tensor weights for the seven
projection modules in each decoder layer. The HF cache + safetensors lazy
open keeps peak memory at one model's worth of bytes even when scoring four
checkpoints together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


# ---------------------------------------------------------------------- config


@dataclass
class CheckpointSpec:
    role: str
    repo: str
    revision: str = "main"


@dataclass
class ScoreCfg:
    tau_mag_percentile: float = 0.5
    expert_norm: str = "full_model_unit"  # full_model_unit | none
    module_zscore_scope: str = "module"   # module | family | global
    eps: float = 1e-12


@dataclass
class PositionCfg:
    min_gap: int = 2
    default_k: list[int] = field(default_factory=lambda: [7, 14])


@dataclass
class Config:
    name: str
    base: CheckpointSpec
    experts: list[CheckpointSpec]
    modules: list[str]
    module_families: dict[str, list[str]]
    score: ScoreCfg
    position: PositionCfg
    output_dir: Path


def load_config(path: str | Path) -> Config:
    """Parse the JSON config and resolve relative output_dir against the
    config file's directory."""
    path = Path(path)
    with path.open() as f:
        raw = json.load(f)
    base = CheckpointSpec(role="base", **raw["base"])
    experts = [CheckpointSpec(**e) for e in raw["experts"]]
    score = ScoreCfg(**raw.get("score", {}))
    position = PositionCfg(**raw.get("position", {}))
    output_dir = Path(raw["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (path.parent.parent / output_dir).resolve()
    return Config(
        name=raw["name"],
        base=base,
        experts=experts,
        modules=raw["modules"],
        module_families=raw["module_families"],
        score=score,
        position=position,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------- model layout


# Qwen2.5 / Llama-style decoder block tensor naming. The pipeline trusts these
# names; if a future architecture uses different keys, override at the call
# site rather than guessing.
def module_tensor_key(layer: int, module: str) -> str:
    if module in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return f"model.layers.{layer}.self_attn.{module}.weight"
    if module in {"up_proj", "gate_proj", "down_proj"}:
        return f"model.layers.{layer}.mlp.{module}.weight"
    raise KeyError(f"unknown module name: {module!r}")


def module_bias_key(layer: int, module: str) -> str:
    """Qwen2.5 uses q/k/v biases but no o_proj bias and no MLP bias. We never
    score biases in the conflict score, but the compatibility gate checks them
    when both base and expert have them."""
    if module in {"q_proj", "k_proj", "v_proj"}:
        return f"model.layers.{layer}.self_attn.{module}.bias"
    return ""  # no bias on this module


# ---------------------------------------------------------------- model state


@dataclass
class ModuleTensor:
    """A view of one module's weight tensor on a particular checkpoint."""
    repo: str
    layer: int
    module: str
    tensor: torch.Tensor       # always materialised in fp32 here
    dtype_orig: torch.dtype    # the dtype on disk (bf16 / fp16 / fp32)


class ModelState:
    """Lazy-open the safetensors shards of one HF repo and serve per-tensor
    accessors. Loads metadata immediately, materialises tensors only when
    asked for. Reusable across many module queries."""

    def __init__(self, repo: str, revision: str = "main", cache_dir: str | None = None):
        self.repo = repo
        self.revision = revision
        # Pull every safetensors shard + config + tokenizer once; cached afterwards.
        self.local_dir = Path(
            snapshot_download(
                repo_id=repo,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=[
                    "*.safetensors",
                    "*.safetensors.index.json",
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "special_tokens_map.json",
                    "added_tokens.json",
                ],
            )
        )
        with (self.local_dir / "config.json").open() as f:
            self.config = json.load(f)
        index_path = self.local_dir / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open() as f:
                self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        else:
            # Single-shard model. Find the one safetensors file.
            shards = list(self.local_dir.glob("*.safetensors"))
            if len(shards) != 1:
                raise RuntimeError(
                    f"{repo} has neither an index file nor exactly one .safetensors shard"
                )
            single = shards[0].name
            self.weight_map = {}  # filled on first access
            self._single_shard = single
        # Open the shards once; safetensors closes file handles automatically.
        self._open_files: dict[str, "safe_open"] = {}

    @property
    def num_layers(self) -> int:
        return int(self.config["num_hidden_layers"])

    def keys(self) -> list[str]:
        if self.weight_map:
            return list(self.weight_map.keys())
        # Single-shard path: open and read keys.
        with safe_open(self.local_dir / self._single_shard, framework="pt", device="cpu") as f:
            return list(f.keys())

    def shape(self, key: str) -> tuple[int, ...] | None:
        """Return tensor shape without materialising it. Returns None if the
        key is absent (used by the compatibility gate)."""
        shard = self._shard_for(key)
        if shard is None:
            return None
        with safe_open(shard, framework="pt", device="cpu") as f:
            tensor_slice = f.get_slice(key)
            return tuple(tensor_slice.get_shape())

    def get(self, key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Materialise one tensor. Returns it as fp32 on CPU by default — we
        always compute conflict in fp32 to avoid bf16 underflow on norms."""
        shard = self._shard_for(key)
        if shard is None:
            raise KeyError(f"{key} not found in {self.repo}")
        with safe_open(shard, framework="pt", device="cpu") as f:
            t = f.get_tensor(key)
        return t.to(dtype)

    def get_module(self, layer: int, module: str) -> ModuleTensor:
        key = module_tensor_key(layer, module)
        t = self.get(key, dtype=torch.float32)
        # Check disk dtype for diagnostic; safetensors stores it in metadata.
        shard = self._shard_for(key)
        with safe_open(shard, framework="pt", device="cpu") as f:
            sl = f.get_slice(key)
            dtype_orig = sl.get_dtype()  # returns a string like "BF16"
        return ModuleTensor(
            repo=self.repo,
            layer=layer,
            module=module,
            tensor=t,
            dtype_orig=_safetensors_dtype_to_torch(dtype_orig),
        )

    # ------- internals -------

    def _shard_for(self, key: str) -> Path | None:
        if self.weight_map:
            shard_name = self.weight_map.get(key)
            if shard_name is None:
                return None
            return self.local_dir / shard_name
        # Single-shard repo: assume the key is in the one shard if it exists.
        path = self.local_dir / self._single_shard
        with safe_open(path, framework="pt", device="cpu") as f:
            if key not in f.keys():
                return None
        return path


def load_model_state(repo: str, revision: str = "main") -> ModelState:
    """Convenience helper. Downloads on first call, returns a ModelState
    bound to the local cache afterwards."""
    return ModelState(repo, revision=revision)


def iter_module_tensors(
    state: ModelState, modules: list[str]
) -> Iterator[tuple[int, str, torch.Tensor]]:
    """Yield (layer_idx, module_name, fp32_tensor) for every (layer, module)
    pair in the decoder. Useful for the full-model delta norm pass."""
    for layer in range(state.num_layers):
        for m in modules:
            yield layer, m, state.get(module_tensor_key(layer, m))


# ----- safetensors dtype mapping -----


_SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _safetensors_dtype_to_torch(name: str) -> torch.dtype:
    return _SAFETENSORS_DTYPES.get(str(name), torch.float32)
