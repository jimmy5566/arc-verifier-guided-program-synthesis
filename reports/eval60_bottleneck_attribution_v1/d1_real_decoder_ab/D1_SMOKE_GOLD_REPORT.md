# D1 fixed-budget smoke: post-freeze Gold scoring

This CPU-only report scores immutable, target-blind frozen candidate pools. It is D1 fixed-budget pilot evidence only and does not update the historical 33/89 union.

- Contract SHA256: `caaa3ff7490eaa6deb3f4965ed7e382ddf5d6a1b315382bf820fc3090eeff6a1`
- Unique smoke outputs: `7`

| Policy | Gold-hit cells | Gold-hit outputs | New outputs vs V5 |
|---|---:|---:|---:|
| D1_V5_CAPPED_CONTROL | 0 | 0 | 0 |
| CUMULATIVE_REGRET_r=4.00 | 1 | 1 | 1 |
| AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400 | 0 | 0 | 0 |
| TOPK_LOCAL_k=2 | 0 | 0 | 0 |

- Promote TopK2 to Full144: `NO`
- Retain Affine for later validation: `NO`
- Run Full144 Regret4: `YES`
- If no finalist rescue: `None`
