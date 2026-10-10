# Matched Atomic-to-Compositional Transfer V2 ? Run 002

Process completed under the one-shot 1,800-second no-update authorization. CPU scoring used the sealed local synthetic-target sidecar only after prediction freeze.

| Checkpoint | Select with cue | Parameterized recolor | Composition select-recolor | Protected recolor retention |
|---|---:|---:|---:|---:|
| V7 | 48/48 | 48/48 | 48/48 | 24/24 |
| Family-Balanced | 48/48 | 48/48 | 45/48 | 24/24 |

The fixed Batch1 validation matched Batch16 parsed grids on 84/84 checked rows. Across all 168 paired synthetic episodes: both correct 165, V7-only 3.

Scope: synthetic within-schema compositional execution evidence only; this is not held-out TRAIN transfer or independent ARC generalization evidence.
