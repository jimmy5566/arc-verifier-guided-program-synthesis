# Data source catalog

- `official_arc_agi_2`: ACCEPTED — accepted: training=REPLAY, evaluation=HARD_EXCLUDE
- `miniarc`: ACCEPTED — accepted as BASE_MODEL_REPLAY
- `conceptarc`: ACCEPTED — accepted as BASE_MODEL_REPLAY
- `rearc`: ACCEPTED — accepted as BASE_MODEL_REPLAY through source-specific native pair-bank adapter; exact original NVARC 2048-pair/256-episode recipe is not reconstructable from observed 1000-pair banks
- `nvarc_source`: QUARANTINE_FOR_DATA — not a data input; retained for provenance audit
- `nvarc_augmented_puzzles`: QUARANTINE — not downloaded or trained: license unknown
- `nvarc_synthetic_puzzles`: QUARANTINE — not downloaded or trained: license unknown
- `nvarc_artifacts_puzzles`: QUARANTINE — not downloaded or treated as SFT examples
- `h_arc`: QUARANTINE — not downloaded or trained: license unknown
- `barc`: QUARANTINE — not downloaded or trained: license unknown
- `sft139_model`: ACCEPTED_IF_COMPLETE — identity-only; never loaded for GPU training
