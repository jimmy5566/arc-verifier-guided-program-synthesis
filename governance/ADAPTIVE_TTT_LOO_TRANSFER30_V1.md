# Adaptive TTT LOO Transfer30 V1 — frozen protocol

This is a target-blind real-test transfer validation.  Before GPU work, an
exposure audit will classify every official evaluation task and select exactly
30 `ADAPTIVE_UNSEEN` tasks with at least three train pairs by ascending
`SHA256(task_id)`.  All Eval60 tasks, Step 1 tasks, and every task found in
prior Adaptive-TTT/LOO materials are excluded.  If fewer than 30 qualify, the
run stops rather than filling from exposed tasks.  A six-task sentinel subset
is selected independently by `SHA256('LOO_TRANSFER30_V1:REAL_TEST_SENTINEL6:'
+ task_id)`.

For every task and every leave-one-train-pair-out fold, the unchanged REF128,
rank-256 rsLoRA configuration runs a continuous trajectory through depth 72.
At depths 0, 12, 24, 48, and 72, all eight canonical geometry views receive
teacher-forced held-pair scores.  The primary competence score is the
arithmetic mean fold NLL per token, ascending.  Median, worst-fold,
token-weighted NLL, and variance are descriptive only.  The complete 40-cell
ranking and its Top1/2/4 prefixes freeze before generation.

For each task, a fresh all-train trajectory generates only the frozen Top4
union and three fixed baselines: depth24/identity, depth48/identity, and
depth12/transpose.  Sentinel6 additionally generates the complete 40-cell
real-test surface.  All predictions freeze before the evaluation solution file
is opened.  No aggregation tuning, new router, task exception, new depth/view,
or post-result regeneration is allowed.  Two RTX 3090 workers use atomic task
claims; each GPU executes at most one continuous TTT trajectory at a time.

After the GPU freeze, CPU-only scoring will report Transfer30 Top1, Top2,
Top4, fixed baselines, and fully solved tasks, plus Sentinel6 full oracle,
TopK retention, correct-cell rank distribution, and unique depth/view/cell
contributions.  The only allowed conclusions are `STRONG_TRANSFER`,
`PORTFOLIO_TRANSFER`, `WEAK_TRANSFER`, or `NO_TRANSFER`.
