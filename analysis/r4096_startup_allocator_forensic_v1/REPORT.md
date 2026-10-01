# R4096 startup allocator forensic V1

Target-blind one-retry allocator diagnosis. No Gold, policy change, or retry loop.

- **Classification:** `R4096_LONG_HORIZON_CAPACITY_FAIL`
- **Retention30 readiness:** `DO_NOT_PROCEED`
- **Initialization hygiene gate:** `PASS`
- **M4 allocated/reserved:** `12964816384` / `13922992128`
- **M7 allocated/reserved:** `12963825152` / `13696499712`
- **Reserved-unallocated M4→M7:** `958175744` → `732674560`

## R4096 retry

- **Status:** `OOM`
- **OOM timing class:** `AFTER_SEARCH_STARTED`
- **OOM execution stage:** `DURING_MODEL_FORWARD`
- **Last dispatch:** `{'cache_length_max': 2117, 'cache_length_mean': 2117.0, 'cache_length_min': 2117, 'last_execution_stage': 'DURING_MODEL_FORWARD', 'physical_forward_index': 5635, 'scheduler_mode': 'B8_SPLIT_MODE', 'scheduler_search_attempted': True, 'selected_batch_size': 8, 'split_group': 'A'}`
