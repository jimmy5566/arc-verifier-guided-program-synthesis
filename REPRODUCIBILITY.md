# Reproducibility

This document distinguishes what can be reproduced from the public source tree from historical runs whose models, datasets, or restricted artifacts are not redistributed.

## Public CPU-only checks

Python 3.11+ is required. The package declares base dependencies in `pyproject.toml`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q
```

These tests exercise deterministic components and documentation-adjacent configuration checks. They do not download data, checkpoints, or run GPU inference.

## Data, models, and provenance

ARC datasets, ARC Prize competition files, solution files, model weights, model caches, raw completions, and Kaggle runtime state are not shipped. Obtain each external dependency directly under its original terms. Keep it outside version control, for example under `data/raw/`.

Several native inference experiments use the public checkpoint identifier `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`; earlier program-synthesis experiments use public Qwen3 backends. These are external/pretrained assets, not checkpoints trained in this repository. The NVARC interface provenance and verified serialization contract are recorded in `configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json`.

## Frozen evaluation protocol

For the documented frozen studies, the following were fixed before a scored condition: task selection, method configuration/hash, model identity, sampling/decoding budget, augmentation set, candidate-selection rule, and solution-access gate.

1. Generate candidates from task training pairs and test inputs only.
2. Run deterministic train-pair execution/verification where applicable.
3. Deduplicate and select one or two test attempts using the fixed, target-blind rule.
4. Freeze candidate and selection artifacts.
5. Only then permit a separate scorer to read solution files.

The 60-task A/B manifest (`artifacts/untouched60_manifest.json`, local/private) records that selection was immutable, target-blind, and frozen before inference. Its score artifact records that solutions were loaded only after the A/B prediction structure was complete. Detailed artifacts are ignored by Git because they may contain non-public outputs; the sanitized conclusions and caveats are documented in `docs/RESEARCH_AUDIT.md`.

## Configurations and artifact map

- `configs/PUBLIC_REFERENCE_SELECTION_SEARCH_ABLATION_V1.json` — frozen30 A/B/C/D diagnostic protocol and public-reference provenance.
- `configs/ARC2_NATIVE_LIKELIHOOD_RANKER_V2_FAST.json` — eight-task untouched ranking protocol.
- `configs/ARC2_NATIVE_TTT_AND_STRONG_VERIFIER_V1.json` — TTT isolation, reset policy, and clean five-task validation protocol.
- `experiments/results/` — versioned sanitized aggregates for symbolic, program-synthesis, recognition, and system studies.
- `reports/` — human-readable reports; negative findings are retained.
- `docs/EXPERIMENT_INDEX.md` — the navigational index with scope and result status.

## GPU requirements

Historical native runs used NVIDIA L4 GPUs, `bfloat16`, PyTorch/Transformers, and complete per-worker model replicas. The production runner supports four workers, one model per L4, checkpointing, resume/no-duplicate behavior, and a dynamic queue. Model weights and competition infrastructure must be provisioned separately. Do not interpret the CPU-only scheduler replay as a substitute for a live hardware benchmark.

The Qwen3 scaling smoke and its corrected latency summary are in `experiments/results/QWEN3_8B_1_2_4_GPU_SCALING_SMOKE_CORRECTED.json`. Some historical configurations require exactly four L4 devices; tests should not attempt to recreate those runs automatically.

## Seeds and inference settings

Settings are configuration-specific. Examples recorded in frozen configurations include temperature 0, fixed seeds, prescribed augmentation counts, candidate budgets, and model serialization contracts. Use the configuration associated with the experiment, rather than transferring a setting across cohorts. Exact raw generations are not guaranteed reproducible without the same external model version, CUDA/PyTorch stack, accelerator topology, and private data mount.

## Reproducibility boundary

This repository supports deterministic code inspection, CPU tests, configuration review, and reproduction of analyses whose inputs are available. It does not claim full replayability of every GPU experiment because external models, datasets, raw predictions, and competition runtime artifacts are unavailable or intentionally withheld. Metrics are reported as historical artifacts, never regenerated or altered by this documentation package.
