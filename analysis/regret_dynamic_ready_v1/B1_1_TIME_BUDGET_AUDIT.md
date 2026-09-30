# REGRET_DYNAMIC_READY_B1_1_TIME_BUDGET_REPAIR

No Gold was opened during the B1.1 target-blind experiment; Dynamic Batch2 was not run.

- Root cause: MIXED
- Isolated B1 parity: 4/4
- Shared scheduler B1 parity: 1/4
- Gate: FAIL

## Prior scheduler evidence

- First seen order: anti_transpose, flip_ud, identity, transpose.
- Lexical scheduling verified: true.

## Cells

| Cell | authoritative / isolated / shared nodes | isolated exact | shared exact |
|---|---:|---|---|
| d59b0160:o0:d24:identity | 4096 / 4096 / 3899 | True | False |
| d59b0160:o0:d24:flip_ud | 4096 / 4096 / 3249 | True | False |
| d59b0160:o0:d24:transpose | 3716 / 3716 / 3899 | True | False |
| d59b0160:o0:d24:anti_transpose | 3899 / 3899 / 3899 | True | True |
