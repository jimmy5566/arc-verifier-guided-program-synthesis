# CLEAN_HF_PARALLEL_DFS_PRODUCTION_SPEED_V1

- Decision: `PRODUCTION_PARALLEL_DFS_CENSUS_READY`
- Target blind: `True`; Gold loaded: `False`; Unsloth: `False`.
- B1 search: `14.580337106752355` nodes/s in `35.11578616127372` s.
- B2 search: `24.116524587519876` nodes/s in `21.23025638051331` s.
- Search speedup: `1.6540443757196246`; prefill-inclusive speedup: `1.6031195185046496`.
- B2 forwards/effective batch/active2: `256` / `2.0` / `1.0`.
- Old validation B1 non-model wall estimate: `271.6420153081417` s (0.886 of old total).
- Old validation B2 non-model wall estimate: `273.50247297622263` s (0.939 of old total).
- Instrumentation-dominance assessment: `CONFIRMED_LARGE_VALIDATION_OVERHEAD`.

All measurements exclude model load, adapter load, tokenizer/challenge preparation, result serialization, and file writes from the primary search metric.
