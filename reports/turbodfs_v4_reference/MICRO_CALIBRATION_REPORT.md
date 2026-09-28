# TurboDFS V4 public-reference parity micro calibration

Status: **MICRO_FAIL_STOP**. No evaluation solutions were accessed.

The initial direct public recursion was run one stored view at a time and
completed 3/6 target cells. Trace inspection identified a bounded B-class
integration discrepancy: the public worker forwards equal-length prompts as a
batch and carries its batched KV state through recursion. The only permitted
correction therefore grouped the frozen views as `(identity, flip_ud)` and
`(transpose, anti_transpose)`, without adding a view, candidate budget, or any
new pruning rule.

The corrected path completed 4/6 target cells. It therefore still missed the
required 5/6 gate. The two residual target cells (`97d7923e/o0/d12/flip_ud` and
`97d7923e/o0/d48/transpose`) had no EOS completion, reconstructible complete
search traces, valid native-token contracts, and only
`public_cumulative_nll` prunes before `search_exhausted`. They are classified
as `E_PUBLIC_REFERENCE_SEARCH_EXHAUSTED`, not a parser, token-mask, cache, or
runtime error.

Consequently the protocol stops: full 24-cell calibration and 1068-cell Eval60
TurboDFS collection were **not** started; Gold was not attached. The unchanged
authoritative Greedy run remains at
`/workspace/arc2/active_runs/eval60_authoritative_greedy_v1` with freeze hash
`54f16cce21f504c30562260ca9163725eb8d6d9e1475fc6f4459c874fb786bc2`.

Remote handoff: `/workspace/arc2/active_runs/eval60_turbodfs_v4_reference/TURBODFS_V4_HANDOFF.md`
(`8f81e23db4ae7bcbf8594dc947c894df9bd9029a20bb2ed6dd54c9bee8c5b6d1`).
