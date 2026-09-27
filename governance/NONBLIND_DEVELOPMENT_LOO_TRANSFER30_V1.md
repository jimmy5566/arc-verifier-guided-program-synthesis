# NONBLIND_DEVELOPMENT_LOO_TRANSFER30_V1 — frozen amended protocol

The strict-blind cohort is unavailable: the prior CPU exposure audit found
`ADAPTIVE_UNSEEN = 0`.  This amendment creates a separately scoped **development
transfer / mechanism validation** only.  It is not unseen, untouched, held-out,
strict blind validation, or final generalization evidence.

The cohort is the first 30 eligible `HISTORICALLY_EXPOSED_BUT_NOT_ADAPTIVE`
official evaluation tasks in ascending `SHA256(task_id)` order, requiring at
least three train pairs.  Eval60, all prior Adaptive-TTT/LOO design tasks, and
tasks classified as Adaptive development remain excluded.  Exposure metadata
and bounded historical path evidence remain frozen for every selected task.  If
fewer than 30 are eligible, use all eligible tasks and report the reduction.
The Sentinel6 selection is independently deterministic among the frozen cohort.

Historical correctness, solutions, and outcomes are forbidden for selection,
LOO teacher forcing, competence ranking, Top1/2/4 selection, and GPU inference.
They are read only after the complete main predictions, three fixed baselines,
and Sentinel6 40-cell surfaces have frozen.

All scientific mechanics remain unchanged: all folds, REF128, rank-256 rsLoRA,
depths 0/12/24/48/72, eight canonical views, continuous 0→72 trajectories,
ascending mean fold NLL-per-token ranking, Top1/2/4 generation, fixed
depth24/identity, depth48/identity, depth12/transpose baselines, and no
target-informed regeneration or retries.  Two RTX 3090s use atomic task claims
with at most one continuous TTT trajectory per GPU.
