# CUSP — Phase 1 (offline scoring) implementation

Computes the layer-conflict score described in the CUSP paper from a base
checkpoint and same-base fine-tuned experts. No training and no forward pass
required at this stage — all work is per-tensor weight arithmetic, runs fine on
CPU. Outputs JSON tables and a markdown report under `--output-dir`.

This is what the paper calls **阶段 1** in `EXPERIMENTS.md`: the no-training
diagnostic that has to land before any GPU is burned.

## What it does

1. **Compatibility gate.** For each (base, expert) pair, locally verifies tokenizer
   vocab, the relevant `config.json` fields, the full set of state-dict keys,
   and every tensor shape. Writes a per-expert pass/fail report.
2. **Per-module conflict score.** For each `(layer, module)` pair on the seven
   projection modules in a Qwen2.5/Llama decoder block, computes residual
   energy `D` and sign disagreement `S`, standardises within module family, and
   produces the per-module conflict `C = z(D) · z(S)`.
3. **Per-position aggregation.** Aggregates module conflicts to insertion
   positions `P_π` matching the paper's per-position formula (Section 4.1 of
   the LaTeX).
4. **Insertion plan.** Greedy top-`k` selection over positions, with a
   minimum-gap constraint so the entire budget cannot collapse into a single
   hot spot.
5. **Weight-norm saliency baseline.** The free saliency baseline (no calibration
   corpus needed); the gradient / activation / Fisher / inverse-pruning
   baselines that need a calibration corpus on GPU live in `cusp_saliency_gpu.py`
   (separate script, run after Phase 1 if you want the full RQ1 baseline set).

## Install

```bash
cd code
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

You also need network access to download base + expert checkpoints from
Hugging Face, and roughly 20 GB of disk for the Qwen2.5-1.5B family. The
1.5B family fits comfortably in 16 GB RAM at bf16.

## Run

```bash
python scripts/run_phase1.py --config configs/qwen2.5_1.5b.json
```

The script logs progress to stderr and writes the following under the configured
output directory:

```
results/qwen2.5-1.5b/
├── compat_report.md             # human-readable pass/fail per expert
├── compat_report.json           # machine-readable version
├── module_conflicts.json        # (layer, module) -> {D, S, C, energy, per-expert norms}
├── position_scores.json         # per-position P_π and the underlying C values
├── insertion_plans.json         # top-k positions for k in {7, 14}
├── weight_norm_saliency.json    # baseline #1 (free)
└── summary.md                   # one-page markdown report
```

The markdown summary is the file to look at first; it includes the per-position
score curve, the proposed insertion plan, and the top / bottom contributing
modules.

## Reproducibility

The script pins a base revision in the config. Re-running with the same config
should produce bit-identical outputs (only fp32 reductions are non-deterministic
across CUDA devices; CPU reductions are deterministic).

## Sanity test (no download)

```bash
python scripts/run_phase1.py --config configs/qwen2.5_1.5b.json --dry-run
```

Validates the config and prints what would be downloaded, without touching the
network or the disk.

## Next: GPU saliency baselines

Run separately once Phase 1 is happy:

```bash
python scripts/run_saliency_gpu.py --config configs/qwen2.5_1.5b.json \
    --calibration-corpus path/to/corpus.jsonl
```

This produces `gradient_norm_saliency.json`, `fisher_saliency.json`,
`activation_norm_saliency.json`, and `inverse_pruning_saliency.json` to drop
into the RQ1 comparison.
