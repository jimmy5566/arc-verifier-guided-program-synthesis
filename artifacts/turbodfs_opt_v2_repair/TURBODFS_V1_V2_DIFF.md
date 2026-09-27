# TURBODFS OPT V1 to V2 diff

V2 changes one stopping-policy dimension only. V1 imposed a 256 batch-forward
cap in addition to its wall-clock limit. The public NVARC notebook's
`turbo_dfs` loop has no such forward/node cap; it uses a 540-second
time-only stop. V2 restores that public stopping rule.

| Field | V1 | V2 | Classification | Rationale |
| --- | --- | --- | --- | --- |
| `max_batch_forward_passes` | `256` | `null` | `PUBLIC_REFERENCE_PARITY` | The public loop has no forward-pass cap. |
| `max_wall_seconds` | `45.0` | `540.0` | `PUBLIC_REFERENCE_PARITY` | The public loop checks `time.time() - start_time < 540`. |

All of the following remain byte-for-byte frozen in configuration semantics:
native vocabulary, EOS token, cumulative-NLL threshold, maximum generated
length, branch ordering, candidate cap, TTT checkpoints, depth/view set and
candidate parsing.

This change is target-blind and does not use any evaluation solution.
