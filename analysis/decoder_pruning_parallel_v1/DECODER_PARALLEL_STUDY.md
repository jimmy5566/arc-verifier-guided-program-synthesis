# D0 parallel decoder-policy study

CPU-only replay over frozen G1 incremental-KV Gold-prefix traces.

**Limit:** `LOCAL_BRANCHING_PROXY` is a Gold-prefix conditional statistic. It is not actual DFS nodes, runtime, or true search complexity.

## Family bests

| Family | Config | Outputs | Local proxy ratio | Robust folds |
|---|---|---:|---:|---:|
| LENGTH_AWARE | AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400 | 15 | 1.252 | 4/4 |
| RELATIVE_REGRET | CUMULATIVE_REGRET_r=4.00 | 10 | 1.011 | 4/4 |
| RANK_MARGIN | TOPK_LOCAL_k=2 | 22 | 1.998 | 4/4 |
| HYBRID | V5_REGRET_RESERVE_r=2.00 | 4 | 1.002 | 4/4 |

## Finalists

- FINALIST_A: CUMULATIVE_REGRET_r=4.00 (highest robust fixed-grid survival with bounded local proxy)
- FINALIST_B: AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400 (highest robust fixed-grid survival with bounded local proxy)
- FINALIST_C: TOPK_LOCAL_k=2 (highest robust fixed-grid survival with bounded local proxy)

## Interpretation

- Hypothesis: STRONGLY_SUPPORTED
- Next: D1_SMALL_REAL_GPU_AB
- No candidate generation, TTT, DFS, model loading, or GPU work was run.
