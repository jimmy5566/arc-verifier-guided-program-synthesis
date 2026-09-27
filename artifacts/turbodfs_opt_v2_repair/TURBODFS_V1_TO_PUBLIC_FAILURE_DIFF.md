# Failure-relevant public parity: V1 vs NVARC

Public reference: `1ytic/NVARC` commit
`846d0198efa752534594e321fc3289fc0a06c657`,
`ARC-AGI1/002_ivan_arc1.ipynb` (SHA256
`24e57f6ae39e63d0041641ae2302a40f7afb9c2973420fb8db05054028537e52`).

This is a focused source comparison; no evaluation solution was opened.

| Failure-relevant mechanism | Public notebook | V1 | Result |
| --- | --- | --- | --- |
| Completion/EOS | retain EOS only when cumulative score is below `max_score` | same | MATCH |
| Native vocabulary | digits, newline and EOS | same token IDs | MATCH |
| Max output length | formatter-derived 30x30 reply bound | 931 tokens | MATCH |
| Score pruning | `score < -log(0.2)` cumulative gate | same strict cumulative gate | MATCH |
| Per-lane candidate ordering | ascending score | ascending cumulative NLL/token | MATCH |
| Lane refill/KV handoff | recursive batched call with `past_key_values` | same public-style lane handoff | MATCH |
| Candidate retention | completed EOS suffixes returned to caller | retained up to frozen cap 4 | NO V1 FAILURE OBSERVED |
| Hard forward/node cap | none | 256 batch forwards | **MISMATCH** |
| Time stop | 540 seconds per TurboDFS call | 45 seconds | **MISMATCH** |

The frozen V1 traces show 13/19 zero-complete cells terminate specifically at
the 256-forward cap before maximum output length. There is no trace evidence
of an EOS detector, grid parser, serialization, deduplication, lane lifecycle,
or KV-cache exception. The public stopping-rule mismatch is therefore the
justified minimal V2 repair. The remaining V1 `search_exhausted` cells remain
separately classified as cumulative-score pruning/frontier exhaustion and are
not claimed fixed before V2 calibration.
