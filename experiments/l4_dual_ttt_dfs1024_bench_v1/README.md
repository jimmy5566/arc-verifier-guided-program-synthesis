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

Kaggle Notebook Version 2 is retained separately in
`FAILED_V2_STARTUP_PROVENANCE.json`. The UUID serialization fix passed, but
only workers 0 and 1 were confirmed ready before worker 3 exited. Its caught
exception was not preserved and peers waited at the 1200-second barrier; no
task, TTT run, DFS cell, candidate, or other scientific datum was produced.

The subsequent infrastructure-only patch adds atomic worker milestones,
authoritative startup failure files, queue draining, unique ready validation,
and bounded global abort cleanup. Lifecycle differences from the known
successful four-L4 runner are recorded in
`SUCCESSFUL_RUNNER_LIFECYCLE_COMPARISON.md`. The scientific benchmark remains
`L4_DUAL_TTT_DFS1024_NOTEBOOK_BENCH_V1`.
