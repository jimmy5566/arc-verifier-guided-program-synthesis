# Atomic-prerequisite training-signal audit

## What the frozen worker optimizes

Each four-example accumulation group uses each row's mean token loss multiplied by `supervised_tokens / group_supervised_tokens`. Each optimizer step is equally weighted. Thus equal episode counts do **not** create equal nominal objective weight.

| Family | Episodes | Transformer tokens | Supervised tokens | Nominal loss share | Steps present |
|---|---:|---:|---:|---:|---:|
| color mapping | 11 | 8838 | 4419 | 3.23% | 6 |
| complete missing structure | 11 | 8282 | 4141 | 2.90% | 5 |
| connected components | 48 | 19312 | 1092 | 1.78% | 48 |
| difference | 48 | 35900 | 17950 | 24.46% | 48 |
| inside/contains | 48 | 36240 | 18120 | 24.74% | 48 |
| orientation | 48 | 19625 | 1645 | 2.52% | 48 |
| overlay | 11 | 8030 | 4015 | 3.15% | 5 |
| propagation | 10 | 7748 | 3874 | 2.98% | 6 |
| recolor | 42 | 31040 | 15520 | 11.97% | 32 |
| rotate | 11 | 4534 | 495 | 2.11% | 5 |
| same color | 32 | 24644 | 12322 | 10.29% | 28 |
| selector prerequisites | 32 | 23052 | 11526 | 8.06% | 28 |
| width | 48 | 19578 | 1152 | 1.81% | 48 |

## Interpretation

- **Established:** the table is a deterministic property of the frozen schedule, tokenizer labels, and worker scaling.
- **Not established:** gradient conflict, gradient size, and learning efficacy were not measured. Nominal loss share is not a gradient measurement.
- **Current round:** remains unchanged. A family-balanced loss comparison is only a candidate for a separately reviewed successor after this round is evaluated.
