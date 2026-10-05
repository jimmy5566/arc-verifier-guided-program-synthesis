# Search Order Micro24 V1 — Post-Freeze Gold Path Diagnostic

## 1. Freeze boundary

The authoritative source archive remains:

- commit: `f33ccc0d8bf655fa67ad35c8e2896fddf6ad202e`
- archive: `analysis/search_order_micro24_v1_e1/`
- archive hash verification: 254/254 PASS
- source generation: target-blind, Gold not loaded

This diagnostic is deliberately stored outside the source archive. The original P0/P1/P2 generation evidence must not be modified.

Frozen interpretation:

- P0 / CURRENT_DFS: SEMANTIC_PARITY_PASS
- P1 / FAIR_DFS_Q64: PARTIALLY_WALL_CENSORED; usable only with censor-aware analysis
- P2 / REGRET_BAND_FAIR_Q64: HEAVILY_WALL_CENSORED; not valid for a full policy-accuracy comparison

## 2. Diagnostic cohort and method

The diagnostic cohort is the 18 Micro24 outputs with `historical_d24_orc == false`.

Gold-path anatomy is taken from:

`analysis/eval60_phase3_r1024_miss_anatomy_v1/CELL_GOLD_PATH_ANATOMY.csv`

Filters:

- depth = 24
- canonical AUG8 views only
- exact Micro24 historical-miss output IDs

This is a post-freeze development diagnostic. It is not blind evaluation.

## 3. Main result

Across 18 outputs × 8 canonical views = 144 d24 cells:

- Gold completed: 0 / 144
- Gold successor retained but not expanded: 90 / 144 = 62.5%
- Gold path pruned: 54 / 144 = 37.5%

At output level:

- 17 / 18 have at least one retained-but-not-expanded Gold successor
- 17 / 18 have at least one pruned Gold path
- 11 / 18 have more retained-not-expanded views than pruned views

Therefore the failure population is mixed. Search-order / compute-allocation failure is a major mechanism, but pruning remains material.

## 4. Breakdown by difficulty group

| Group | Cells | Retained-not-expanded | Pruned |
|---|---:|---:|---:|
| HIGH | 48 | 38 (79.2%) | 10 (20.8%) |
| MID | 48 | 22 (45.8%) | 26 (54.2%) |
| LOW | 48 | 30 (62.5%) | 18 (37.5%) |

The HIGH group is especially search-order limited.

## 5. Two high-prefix cases

### 36a08778:o1

- historical best Gold prefix: 98.17%
- 7/8 views: Gold successor retained but not expanded
- 1/8 views: Gold path pruned
- best retained prefix: 98.17%

This is strong evidence that a near-Gold trajectory can survive pruning yet still fail because expansion priority is poor.

P1 Fair DFS produced no observed exact Gold in the frozen pool, but 2/8 P1 cells were wall-censored. Therefore this is **not a confirmed P1 miss**; it is `UNKNOWN_WALL_CENSORED`.

### 8b7bacbf:o1

- historical best Gold prefix: 88.94%
- 7/8 views: retained-not-expanded
- 1/8 views: pruned
- best retained prefix: 53.63%
- best observed near-Gold prefix: 88.78%, but that trajectory is in the pruned view

P1 Fair DFS reached R1024 on all 8 views and still produced no exact Gold. This is a **confirmed P1 miss**.

The two cases therefore have different mechanisms: 36a08778:o1 is strongly ordering-limited, while 8b7bacbf:o1 shows that fairness alone cannot recover a best trajectory that was removed by pruning.

## 6. Consequence for DFS design

The evidence supports moving beyond uniform compute allocation.

However it does **not** justify a hand-tuned Gold-specific priority function. The next low-overfit baseline should be a pre-registered generic Limited Discrepancy Search (LDS), followed only then by a learned node-value model if needed.

The current artifacts do not expose the exact per-token Gold rank sequence, so exact LDS discrepancy counts are **not observed** here. `first_divergence_token` and cumulative regret must not be treated as discrepancy count.

Decision:

- Search-order / compute-allocation failure: SUPPORTED
- Pruning failure: ALSO MATERIAL
- Simple fairness sufficient: NOT SUPPORTED
- Classic LDS: PLAUSIBLE BUT NOT PROVEN
- Next baseline: P3 CLASSIC LDS, definition frozen before live testing

## 7. Anti-overfitting rule

Gold is used only to characterize failure modes and select a generic hypothesis class. Do not add task-specific thresholds or hand-crafted conditions based on individual Micro24 Gold cases.
