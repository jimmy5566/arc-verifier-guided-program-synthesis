# D2 retrieval alignment (CPU-only)

D0 Gold-prefix survival was recomputed from frozen G1 incremental-KV traces and joined only to post-freeze D1 Smoke8 pools. This is nonblind mechanism evidence, not accuracy validation.

| Policy | Surviving smoke cells | Actual-hit cells | Retrieval conversion |
|---|---:|---:|---:|
| D1_V5_CAPPED_CONTROL | 0 | 0 | NA |
| TOPK_LOCAL_k=2 | 6 | 0 | 0.000 |
| AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400 | 2 | 0 | 0.000 |
| CUMULATIVE_REGRET_r=4.00 | 1 | 1 | 1.000 |

- Mechanism cohort: `6` unique cells
- `SURVIVE0_HIT1` inconsistencies: `0`
