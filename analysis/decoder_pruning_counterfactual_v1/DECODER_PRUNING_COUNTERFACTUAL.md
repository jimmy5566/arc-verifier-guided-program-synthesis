# Decoder pruning counterfactual audit

CPU-only diagnostic over frozen G1 incremental-KV Gold paths.

Gold survival is not an actual solve. Cost is a local Gold-prefix conditional proxy, not a reconstructed DFS tree.

## Policy summary

| Policy | Searchable outputs | Gold-surviving cells | Local multiplier | Under 2x cap |
|---|---:|---:|---:|---|
| CURRENT_V5 | 0 | 0 | 1.000 | True |
| TOPK_INSURANCE_1 | 0 | 0 | 1.000 | True |
| TOPK_INSURANCE_2 | 22 | 162 | 1.998 | True |
| RELREGRET_0.5 | 0 | 0 | 1.000 | True |
| RELREGRET_1.0 | 1 | 3 | 1.000 | True |
| RELREGRET_2.0 | 4 | 12 | 1.003 | True |
| RELREGRET_4.0 | 10 | 27 | 1.011 | True |

## Decision

- Hypothesis: SUPPORTED
- Best policy 1: TOPK_INSURANCE_2
- Best policy 2: RELREGRET_4.0
- Next: SMALL_REAL_DFS_AB_TEST

- Limitation: Local retained-successor counts on frozen Gold-prefix states; actual DFS-tree expansion is not established.
