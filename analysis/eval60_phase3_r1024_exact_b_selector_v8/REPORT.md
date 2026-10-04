# Exact historical B-selector on frozen Phase-3 R1024 pool

- Cohort: 35 ORC-hit outputs / 27 tasks.
- B Top-1 / Top-2: 19/35 / 25/35.
- Full Eval60 Top-2 equivalent: 25/89.
- Historical retained 28: 22/28; DFS-new 7: 3/7.
- NLL baseline: 25/35; B fixes/harms/net: 3/3/0.
- Classification: `DFS_POOL_SELECTION_DISTRIBUTION_SHIFT`.

## Measured
- Frozen-pool B Top-2 conversion with exact historical adapter bytes.

## Not established
- Generalization outside this frozen Eval60 diagnostic cohort.
