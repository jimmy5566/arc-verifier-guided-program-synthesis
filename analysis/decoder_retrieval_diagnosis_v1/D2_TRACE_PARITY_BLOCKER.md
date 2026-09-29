# D2 target-blind trace parity blocker

The required two-cell D2 instrumentation gate did not pass, so no D2
mechanism trace, Gold annotation, V6 prototype, or mechanism A/B was started.

The append-only trace implementation exactly reproduced both selected cells
for `TOPK_LOCAL_k=2`, `AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400`, and
`CUMULATIVE_REGRET_r=4.00`: candidate pool and order, nodes expanded, and
termination reason all matched their frozen D1 records.

`V5_CURRENT` did not reproduce its frozen D1 reference records:

| Cell | Frozen nodes / termination | Re-executed nodes / termination |
|---|---:|---:|
| `20270e3b:o0:d24:identity` | 1873 / `budget_exhausted` | 92 / `search_exhausted` |
| `332f06d7:o0:d24:anti_transpose` | 419 / `search_exhausted` | 419 / `search_exhausted` |

Both frozen V5 records were resolved by D1's legacy paired `raw/` fallback,
not by a per-view fixed-contract queue checkpoint. They have the current
configuration and adapter hashes but no `fixed_budget_contract_sha256` or
scheduler provenance field. Therefore the mismatch is a frozen-control source
identity problem, not evidence that append-only tracing changed a verified
fixed-contract V5 run.

The D2 protocol requires exact parity against frozen D1 records. The correct
status is `BLOCKED_PRETRACE`: retain all D1 pools immutable and do not replace
them silently. A subsequent authorized remediation must decide whether to
freeze a newly generated V5 control under the D1 contract or restrict D2 to a
separately verified control surface. Neither action has been taken here.
