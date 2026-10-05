# P3 LDS V1 — Pre-Registered Test Plan

## Gate 0 — implementation audit

Before GPU generation:

- SearchOrderName includes P3 only through the explicit frozen name `LDS_UNIT_DISCREPANCY_V1`.
- RetainedWorkItem records `local_token_rank` and `discrepancy_count`.
- discrepancy_count is created mechanically from model rank.
- retention code is byte/logically unchanged.
- P0/P1/P2 semantics are unchanged.
- no Gold/solution path is imported by generation code.

Required unit tests:

1. all-top1 chain remains D=0;
2. one non-top1 choice creates D=1;
3. two non-top1 choices create D=2;
4. P3 always pops the lowest non-empty discrepancy layer;
5. within one layer, traversal is depth-first and lower-NLL siblings go first;
6. every retained work item is eligible exactly once; membership is unchanged;
7. frontier-floor restored work uses original local model rank;
8. P0 current DFS parity tests still pass.

## Gate 1 — target-blind 3-output smoke

Use the same deterministic P0 smoke-output selector (PROFILE_S / PROFILE_M / PROFILE_L) and the same frozen model/adapters/AUG8/R1024 contract.

PASS requires:

- no Gold loaded;
- generation hashes PASS;
- no duplicate work-item expansion;
- `retention_changed == false`;
- all 24 smoke cells end as R1024_REACHED or legitimate terminal carry;
- wall_censored = 0;
- no OOM fallback;
- discrepancy telemetry reconciles with work-item lineage.

If Gate 1 fails, do not launch Micro24.

## Gate 2 — full Micro24 target-blind generation

Run P3 on all 24 frozen Micro24 outputs.

Freeze raw outputs, checkpoint curves, EOS events, work-item telemetry, discrepancy telemetry, runtime receipts and hashes before Gold access.

Primary post-freeze scientific endpoints:

- exact-Gold Any-of-AUG8 rescue count on the 18 historical d24 misses;
- exact-Gold retention on the 6 historical d24 controls;
- first-Gold expanded-node index where observable;
- confirmed hit / confirmed miss / unknown-wall-censored counts.

Efficiency endpoints:

- useful expanded nodes;
- replay model forwards;
- replay/useful ratio;
- wall seconds per output;
- wall censoring rate.

## Interpretation

- A Gold hit observed before censoring is a confirmed hit.
- A miss is confirmed only when all relevant cells are clean.
- Otherwise classify as UNKNOWN_WALL_CENSORED.
- Do not tune P3 after seeing Micro24 Gold. If a later algorithm is changed because of this result, it must receive a new policy label and a new pre-registration.
