# Same-adapter pair overlap across TTT depth

## Measured macro medians

|Depth|Top2|Top4|eps1e-3|eps5e-3|eps1e-2|all-pair Top4 tasks|
|---:|---:|---:|---:|---:|---:|---:|
|0|0.250|0.467|0.000|0.000|0.000|4/6|
|12|0.167|0.333|0.000|0.000|0.183|5/6|
|24|0.083|0.333|0.100|0.259|0.661|4/6|
|48|0.000|0.333|0.381|0.750|0.875|5/6|
|72|0.333|0.333|0.688|0.857|1.000|5/6|

## Measured interpretation
- TTT12 is not a coarse-overlap sweet spot: macro Top2 falls from 0.250 to 0.167 and Top4 from 0.467 to 0.333.
- Strict epsilon basins move in the opposite direction after TTT12: eps1e-3 rises from 0.000 at depth12 to 0.100/0.381/0.688 at 24/48/72.
- Therefore the measured result is scale-dependent: a coarse view family is strongest initially, while sharp optima become more cross-pair aligned at deeper TTT.

## Old LOO structural comparison
- The fixed depth/view atoms match, but old LOO scores test outputs whereas this audit scores held-out train pairs. The comparisons are directional/descriptive, not a causal percentage.
- depth 0: Top4 median same-minus-old = 0.000 across 7 old-output comparisons.
- depth 12: Top4 median same-minus-old = 0.000 across 7 old-output comparisons.
- depth 24: Top4 median same-minus-old = 0.095 across 7 old-output comparisons.
- depth 48: Top4 median same-minus-old = 0.000 across 7 old-output comparisons.
- depth 72: Top4 median same-minus-old = 0.000 across 7 old-output comparisons.

## Decision
- Decision gate: F (mixed / insufficient). TTT configuration optimization should not yet be prioritized ahead of decoder research from this N=6 diagnostic alone.
