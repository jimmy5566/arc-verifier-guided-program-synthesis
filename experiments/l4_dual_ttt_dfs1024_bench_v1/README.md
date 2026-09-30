# L4 dual-TTT scalar DFS1024 notebook benchmark

Performance-only benchmark for the frozen workload:

- independent TTT24 plus its frozen four views;
- independent TTT48 plus its frozen four views;
- one scalar `CUMULATIVE_REGRET_r=4.00` DFS invocation per output/view;
- 1024 expanded nodes per logical cell;
- four persistent workers, one complete model per NVIDIA L4;
- no Gold access, rerun, retry, Batch2 cross-cell execution, or Batch4 Regret.

The eight-task cohort is selected from the already exposed Eval60 development
cohort solely by `SHA256(task_id)`.  This directory freezes scientific and
cohort identity; actual runtime artifacts are produced only by the Kaggle
notebook after separate user review and launch authorization.

Kaggle Notebook Version 1 is retained as an infrastructure-only failed run in
`FAILED_V1_INFRA_PROVENANCE.json`: all four models loaded, but an unpickleable
CUDA UUID prevented `MODEL_READY` delivery before any task, TTT, DFS cell, or
candidate began.  The benchmark ID and scientific configuration remain V1.
