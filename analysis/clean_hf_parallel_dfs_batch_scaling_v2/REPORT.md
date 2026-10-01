# CLEAN_HF_PARALLEL_DFS_BATCH_SCALING_V2

- Raw compute/capacity benchmark only.
- Replicas are independent DynamicCache lanes, not additional ARC views.
- Target blind; Gold not loaded; no Unsloth inference; no scientific DFS search.
- Frozen B4 baseline: `49.26944269607013` lanes/s.
- Classification: `B16_OOM_FAIL`.
- Highest efficient physical batch: `12`.
- Next: `INVESTIGATE_HARDWARE_CAPACITY_BOUNDARY`.
