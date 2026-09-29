# D1 fixed-budget decoder A/B: control and 8-cell cost smoke

This is a new fixed-budget decoder comparison, not historical V5 parity. Candidate retrieval is target-blind; no D1 result updates the historical 33/89 union.

- Contract SHA256: `caaa3ff7490eaa6deb3f4965ed7e382ddf5d6a1b315382bf820fc3090eeff6a1`
- Control completion: `144/144`
- Deterministic smoke cells: `8`

| Policy | Median seconds/cell | P90 seconds/cell | Median nodes | Runtime ratio vs control | Node ratio vs control |
|---|---:|---:|---:|---:|---:|
| D1_V5_CAPPED_CONTROL | 48.533 | 103.250 | 1438.0 | 1.000 | 1.000 |
| CUMULATIVE_REGRET_r=4.00 | 241.530 | 243.704 | 4096.0 | 4.977 | 2.848 |
| AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400 | 57.387 | 67.620 | 952.0 | 1.182 | 0.662 |
| TOPK_LOCAL_k=2 | 55.517 | 57.512 | 933.0 | 1.144 | 0.649 |

- Regret4 slowdown classification: `BOTH`
- Phase 4 finalist surfaces have not started.
