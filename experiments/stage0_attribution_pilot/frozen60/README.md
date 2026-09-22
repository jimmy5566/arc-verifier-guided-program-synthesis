# Frozen60 primary Stage-0 cohort

This directory extends, but never overwrites, the historical Frozen30 Stage-0
sanity check at the parent directory.  It is CPU-only and uses these frozen
local inputs:

- `artifacts/untouched60_manifest.json` — immutable 60-task membership;
- `artifacts/untouched60_v33_output_final/artifacts/untouched60/A_candidates_frozen.json`
  — 32-view cached candidate pools;
- `artifacts/frozen60_support_ablation/B_SUPPORT_predictions_frozen.json` —
  frozen two-attempt B-SUPPORT selection; and
- `data/raw/arc-agi_training_solutions.json` — already-local labels, opened
  only after structural validation of all frozen prediction inputs.

No model, CUDA, candidate generator, Kaggle client, or notebook path is called.
