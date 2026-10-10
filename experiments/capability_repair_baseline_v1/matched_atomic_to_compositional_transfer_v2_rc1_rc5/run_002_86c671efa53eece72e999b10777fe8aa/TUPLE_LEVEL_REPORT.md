# Tuple-level CPU closure ? Matched Transfer V2 Run 002

The exact GPU predictions and sealed synthetic-target sidecar were reused without model loading, generation, or target changes. The bootstrap used 10,000 seed-20261010 stratified replicates. Each canonical tuple retained all three roles as one inseparable resampling unit.

| Measure | V7 | Family-Balanced |
|---|---:|---:|
| Both atomic prerequisites exact | 48/48 | 48/48 |
| Composition exact, unconditional | 48/48 | 45/48 |
| Composition failure conditional on both atoms | 0/48 | 3/48 |
| Conditioned-failure bootstrap 95% CI | [0, 0] | [0, 0.125] |
| Protected recolor retention | 24/24 | 24/24 |

The paired common-support contrast (Family-Balanced minus V7 conditioned composition-failure rate) is `0.0625`, with frozen bootstrap 95% interval `[0, 0.125]`; it includes zero and is therefore **inconclusive** under the frozen rule. Four paired tuple states: both correct 45, V7-only correct 3, Family-Balanced-only correct 0, both wrong 0.

V7's bootstrap interval is degenerate because its observed failures are 0/48. It does not establish a zero population failure rate. This remains synthetic within-schema execution evidence only, not held-out-TRAIN transfer or independent ARC generalization.
