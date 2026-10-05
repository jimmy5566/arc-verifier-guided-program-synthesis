# P3 Classic LDS Micro12 — Post-Freeze Gold Scoring

## Result

P3 `LDS_UNIT_DISCREPANCY_V1` produced:

- 3 / 12 exact-Gold Any-of-AUG8 hits = 25%
- 0 / 9 historical d24 miss rescues = 0%
- 3 / 3 control retention = 100%
- 0 unknown/censored outputs

All nine development miss outputs are clean **CONFIRMED_MISS** under P3.

## By stratum

| Stratum | Hits | Total |
|---|---:|---:|
| HIGH | 0 | 3 |
| MID | 0 | 3 |
| LOW | 0 | 3 |
| CONTROL | 3 | 3 |

The three control hits appear by:

- `2d0172a1:o0`: R256, first Gold at expanded node 125
- `1818057f:o0`: R512, first Gold at expanded node 505
- `c7f57c3e:o0`: R1024, first Gold at expanded node 811

Gold-hit curve:

| Budget | Overall | Historical-miss rescue |
|---|---:|---:|
| R256 | 1/12 | 0/9 |
| R512 | 2/12 | 0/9 |
| R768 | 2/12 | 0/9 |
| R1024 | 3/12 | 0/9 |

## Comparison with P1 Fair DFS on the same Micro12 outputs

Exact-Gold outcome is unchanged:

- P1: 3 hits, 0 historical-miss rescues, 3 controls retained
- P3: 3 hits, 0 historical-miss rescues, 3 controls retained

However P1 had four non-hit outputs affected by wall censoring, while P3 has zero.

Runtime improves materially:

- P1 total wall: 5357.13 s = 89.29 min
- P3 total wall: 4109.53 s = 68.49 min
- P3 wall-time reduction: ~23.3%
- P3 speedup: ~1.30x

Replay efficiency:

- P1 replay/useful = 33.72%
- P3 replay/useful = 1.87%

Thus P3 fixes the runtime/censoring confound and makes the search substantially cheaper, but it does **not** improve Gold recall.

## Scientific interpretation

This is a clean negative result for the pre-registered Classic LDS hypothesis on Micro12.

The earlier Gold-path audit showed many Gold successors were retained but not expanded, but simply preferring lower unit-discrepancy paths is not sufficient to rescue them under R1024.

The result does not imply that search ordering is irrelevant. It rejects this specific low-discrepancy ordering as an accuracy improvement on the frozen Micro12 cohort.

The next experiment should not tune P3 after seeing these outcomes. Any change must receive a new pre-registration and policy label.

Most natural next direction:

`P4 = REGRET-GUIDED / VALUE-GUIDED SEARCH`

using target-blind node-value signals rather than unit discrepancy alone.
