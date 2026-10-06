# Exposure policy v2

Every puzzle receives exactly one final role.
- `HARD_EXCLUDE_FUTURE_EVAL`: official evaluation and explicit future governance references.
- `TRAIN_ELIGIBLE_REPLAY`: known SFT139 or historical-development puzzles retired from future evaluation.
- `NOVEL_TRAIN_CANDIDATE` and `NOVEL_HOLDOUT_RESERVED` require licensed provenance and a pre-training family split.
- `QUARANTINE_PROVENANCE_OR_OVERLAP_UNRESOLVED` is the default for unresolved provenance or licensing.
Filename words such as `frozen` do not determine heldout status.
