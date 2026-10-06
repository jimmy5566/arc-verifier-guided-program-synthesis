# Compositional-ARC systematicity audit

Pinned dataset revision: `51d0950d525a31a63c4d7444b109826b6d2ddf9b`.
Pinned generator commit: `ff0a7d16c678169e5248f46bb45d28d053d0f841`.
Official split: `split_seed_1860`.

Episodes: {'train': 82908, 'validation': 8546, 'holdout': 8546}.
Template counts: {'train': 8, 'validation': 2, 'holdout': 2}.
Episode identity overlap: {'train_validation': 0, 'train_holdout': 0, 'validation_holdout': 0}.
Higher-level composition overlap: {'train_validation': 0, 'train_holdout': 0, 'validation_holdout': 2}.

The pinned upstream protocol is exactly reproduced. It intentionally splits the two OOD composition templates by episode into validation/test, so those two higher-level templates and their 96 parameter identities occur in both evaluation partitions.
