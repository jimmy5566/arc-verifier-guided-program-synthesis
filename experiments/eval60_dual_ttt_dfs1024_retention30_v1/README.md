# Eval60 dual-TTT DFS1024 Retention30 V1

This is a target-blind GPU candidate-retention experiment built from the 30
Eval60 outputs that the immutable historical fixed4+4 greedy pool could reach.

The Gold-derived cohort is frozen once in `RETENTION30_OUTPUTS.json`.  The GPU
runner receives only `TARGET_BLIND_RUN_MANIFEST.json` and the evaluation
challenge inputs.  It persists all valid deduplicated DFS candidate grids and
hashes the per-cell and source-union pools before the separate CPU scorer is
allowed to read evaluation solutions.

No DFS Top-1 or Top-2 selector is defined in this experiment.  Its primary
scientific result is exact candidate-pool oracle retention.

This directory is a review source.  It does not authorize a Kaggle push or run.

The pre-launch telemetry-only correction reserves `IDLE` for explicit queue
waiting, labels unmatched samples `UNCLASSIFIED`, phases per-cell candidate
serialization, and keeps initial GPU model preparation inside `MODEL_LOAD`.
The frozen cohort and scientific configuration are byte-identical.
