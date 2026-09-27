# Interim LOFO competence-region audit

CPU-only and target-blind. This reads only the Git-frozen checkpoint snapshot; no evaluation solution or real-test correctness data was opened.

- Frozen ranking tasks: `22`
- Frozen production tasks: `21`
- Held LOO folds: `70`
- Ranking snapshot SHA256: `d8a487319a3de392a1b107799f42e6eab0452c7e89dde024056d0c0983535bf0`

## LOFO region retrieval

| K | Exact held-fold winner inclusion | Median held-fold rank | Median NLL regret |
| --- | ---: | ---: | ---: |
| Top1 | 0.043 (3/70) | 12.50 | 0.018099 |
| Top2 | 0.114 (8/70) | 6.00 | 0.012791 |
| Top4 | 0.200 (14/70) | 4.00 | 0.006184 |

## Near-optimal competence-basin retrieval

| Epsilon | Median basin cells | Top1 | Top2 | Top4 |
| --- | ---: | ---: | ---: | ---: |
| 0.0001 | 1.00 | 0.071 | 0.143 | 0.257 |
| 0.0005 | 1.00 | 0.086 | 0.157 | 0.300 |
| 0.001 | 1.00 | 0.100 | 0.171 | 0.329 |
| 0.002 | 2.00 | 0.114 | 0.200 | 0.371 |
| 0.005 | 3.00 | 0.186 | 0.314 | 0.457 |
| 0.01 | 6.00 | 0.271 | 0.471 | 0.600 |

## Frozen production diversity

- Usable completed task predictions: `21/21`
- Top1/Top2 identical canonical-output rate: `0.143` (3/21)
- Mean unique outputs: Top1 `1.381`, Top2 `2.476`, Top4 `4.762`.

## Scope

This measures transfer of train-pair LOO NLL regions only. It does not establish real-test output correctness and cannot authorize a production decision.
