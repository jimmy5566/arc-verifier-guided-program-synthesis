# Adaptive TTT Research Plan V1

Status: **FROZEN BEFORE STEP 1 GPU INFERENCE**

## Definitions

- **REF128**: the fixed TTT training representation: 16 deterministic colour permutations x 8 D4 variants. It is the training-time representation only.
- **TTT_DEPTH**: an adaptation checkpoint on one continuous TTT trajectory.
- **GEN_VIEW**: a reversible, post-TTT generation-time view. It is not TTT augmentation and must never be described as such.

Historical development anchors (not inputs to Step 1 selection or routing):

| Line | Oracle |
| --- | ---: |
| TTT24 Aug8 | 24/89 |
| TTT48 Aug8 | 28/89 |
| TTT24 union TTT48 Aug8 | 31/89 |
| Production-like fixed 4+4 | 30/89 |

## Hypotheses

1. Qwen self-confidence predicts exact correctness.
2. The useful TTT depth differs by task.
3. Optimal GEN_VIEW interacts with TTT depth.
4. A target-blind confidence signal can select a near-oracle policy.

## Phase roadmap

`history audit -> Step 1 pilot -> interaction audit -> large-scale surface -> adaptive router -> early stopping -> unseen validation -> full candidate generation -> production comparison`

Only Step 1 is authorized by this plan. Later phases require a separate frozen plan and explicit launch authorization.

## Step 1: Depth x GEN_VIEW confidence surface

### Cohort and target-blind protocol

- 12 deterministic hash-selected tasks from the frozen Eval60 development cohort, restricted to tasks with at least three train pairs.
- Selection may use only task structure and task hash; it may not use historical P24/P48 outcomes.
- One deterministic train pair is held out per task. Its output is excluded from TTT and is only opened after each cell's adaptation state is frozen.
- Eval60 real-test targets remain unavailable throughout GPU execution.

### Scientific contract

- Preserve REF128, rank 256, alpha 32, rsLoRA, learning rate, seed, optimizer, parser, dtype, and reference machinery.
- `reference_schedule_total_steps = 128` for the continuous trajectory.
- Run one trajectory per task to depth 72. Do not restart per checkpoint depth.
- Checkpoints: `0, 12, 24, 48, 72`.
- At each checkpoint decode each standard GEN_VIEW once: `identity`, `rot90`, `rot180`, `rot270`, `flip_lr`, `flip_ud`, `transpose`, `anti_transpose`.
- Planned surface: 12 tasks x 5 depths x 8 views = 480 cells.
- Hard GPU wall time: 85 minutes. At deadline no new work starts; completed work is atomically frozen and reported as partial if necessary.

### Per-cell measurements

Primary self-confidence signals: mean sequence log-probability per token, sequence log-probability, mean token entropy, mean and minimum top1-top2 token margin, and generation length. Record available adaptation signals: TTT loss, loss slope/recent delta, and adapter update magnitude when already available without changing training semantics.

Retrospective LOO labels: greedy exact correctness, parse validity, teacher-forced NLL, and token accuracy. Teacher-forced metrics are called `CROSS_VALIDATION_SCORE` and do not enter primary self-confidence H1.

### Analysis gates

- H1: within-task self-confidence association with LOO greedy exactness; per-signal AUC/rank association/quartile rates and argmax-confidence router regret versus oracle.
- H2: coverage and exclusive contribution of depths.
- H3: view-by-depth interaction, joint oracle, depth-only oracle, and view-only oracle.
- H4: compare confidence-selected cell, random-cell expectation, best fixed depth+view, and per-task cell oracle. No learned multivariate router or threshold tuning is allowed in Step 1.

## Artifact contract

`artifacts/adaptive_ttt_step1_depth_view_confidence_v1/` contains the manifest, cohort, cells, trajectory log, surface, report, SHA256 manifest, and decision log update. Each completed cell is atomically checkpointed. The base model and adapter reset/integrity checks must pass.
