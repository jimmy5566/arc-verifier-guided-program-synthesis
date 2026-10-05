# Search-order node separability v1

CPU-only post-freeze development audit over frozen Phase-3 d24 raw search trees.

## Dataset

- Outputs: 89
- Unique tasks: 60
- Cells: 712
- Expanded retained nodes: 656542
- Gold-prefix positive nodes: 90395
- Positive prevalence: 0.1377
- Missing-parent feature rows: 0

A positive node is an expanded retained node whose entire prefix exactly matches the transformed Gold prefix through that node.
Gold is used for labels only.

## Target-blind model

Fixed model: StandardScaler + LogisticRegression(C=1, class_weight=balanced), 5-fold GroupKFold by task ID.

Features:
- branch_depth
- cumulative_nll
- cumulative_regret
- token_logprob
- regret_increment
- is_top1
- discrepancy_count
- mean_nll_per_depth
- mean_regret_per_depth
- frontier_floor_activated
- view=flip_lr
- view=flip_ud
- view=identity
- view=rot180
- view=rot270
- view=rot90
- view=transpose

No task/output IDs, token-prefix identity, Gold rank, Gold NLL, Gold prefix fraction, exact hit, or future rescue fields are model inputs.

## OOF results

- ROC-AUC: 0.9501
- Average precision: 0.7381
- Positive prevalence: 0.1377
- AP gain: 0.6004
- Minimum fold ROC-AUC: 0.9349
- Recall @ top 5% nodes: 0.3040
- Recall @ top 10% nodes: 0.5591
- Recall @ top 20% nodes: 0.8491

Decision: **STRONG_NODE_VALUE_SIGNAL**

Strong authorization requires all preregistered gates in FEATURE_SCHEMA.json. A weak or failed result does not authorize live P4 learned-value search.
