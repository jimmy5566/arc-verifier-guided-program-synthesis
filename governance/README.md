# ARC2 Governance V1

This directory separates one immutable release identity from one reusable
experiment workbench. It does not publish data, start a GPU session, or submit
to Kaggle.

## Identities

- Frozen release: `releases/d1-submitted-v1/release.lock.json`.
- Mutable status history: the release's append-only
  `submission_events.jsonl`.
- Experiment entry: `scripts/run_experiment_workbench.py` and the locally
  staged `workbench/arc2-experiment-workbench.ipynb`.

The experiment entry rejects a truthy `KAGGLE_IS_COMPETITION_RERUN`, never
FAST_SAVEs, and never writes the official top-level `submission.json`. The
submitted release wrapper is not modified by this governance layer.

## Authoritative shared solver

- `scripts/run_d1_failsoft_4gpu.py:run_live_failsoft`
- `src/inference/d1_failsoft_runtime.py:finalize_failsoft`
- `src/inference/d1_failsoft_runtime.py`
- `src/inference/selector_d1.py`

Historical runners remain indexed under `registry/runner_index.json`; they are
not deleted or silently promoted.

See [OPERATIONS.md](OPERATIONS.md) for the controlled workflow.
