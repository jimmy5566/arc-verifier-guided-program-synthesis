# Reproducibility

## Supported environment

The package metadata requires Python 3.11 or newer. Dependency lower bounds are recorded in `pyproject.toml`; no unverified lockfile is supplied. Install the normal local development environment with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest -q
```

The public release verification suite is CPU-only and should not require ARC data, a model, or a GPU.

## Data layout

No data is bundled. Obtain an ARC dataset yourself under its original terms and keep it in an ignored location such as `data/raw/`. Scripts accept data roots or use repository-relative paths where documented. Do not add solution files, competition-only data, or prediction checkpoints to commits.

## V2 configuration and deterministic controls

`configs/v2_pilot_public_protocol.json` documents the public, portable V2 pilot protocol: Qwen3-8B, local/offline Transformers, deterministic sampling (`temperature=0`, `top_p=1`, seed `0`), Macro DSL version, capability-registry hash, candidate budget, and the solution-access gate. Its `private_run_config_sha256` records the byte hash of the original environment-bound frozen configuration used for the run; that private configuration is intentionally not published because it contains a competition mount locator.

The verified protocol is: generate from challenge inputs, execute and hard-verify against train pairs, freeze predictions, then allow the gated scorer to access solutions for exact scoring. Candidate selection never accesses test solutions.

## Optional GPU/model runs

GPU runs require a separately acquired Qwen3-8B-compatible local model, PyTorch with a CUDA wheel appropriate to the host, and the optional `llm` dependencies. Set a local model locator through the runner's CLI/configuration before launching a run. No model weights, caches, or hosted runtime metadata are distributed here.

The historical V2 infrastructure used four independent GPU workers with one worker mapped to each GPU. It performs sequential artifact warm-up and staggered model initialization before opening the shared task queue, because simultaneous cold reads of the model artifact caused contention in a fresh runtime. Checkpoints are atomic and support resume/no-duplicate behavior; checkpoints themselves are intentionally ignored.

## Frozen-config philosophy

Scientific parameters are frozen before a scored condition: model identity, sampling settings, prompt/schema versions, capability registry hash, candidate budget, task-selection method, and solution-access protocol. The public snapshot keeps sanitized aggregate metrics and a portable protocol description. It does not change reported metrics or reconstruct unavailable raw outputs.
