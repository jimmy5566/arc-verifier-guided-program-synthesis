# MINIMAL_CALL_HISTORY_REPRO_V1

## Status

`REPRODUCTION_GATE = FAIL` for the new RTX 3090 runtime.  No decoder,
scheduler, cache, or model change was implemented.

The valid `exact_request` run used the Dynamic-READY coroutine's first legal
request, not an argmax approximation:

| Field | Value |
| --- | --- |
| Source commit | `2b67720281f1dd6cac21f7d995de23d961f4be78` |
| Task/output/depth | `d59b0160:o0:d24` |
| View/request | `flip_ud`, token `0`, position `1928`, ordinal `0` |
| Policy/caps | Regret4, 4096 expanded nodes / 32 completed candidates |
| Gold access | `NO` |
| Search/DFS | `NOT RUN` |

The immediate A/B replay was exact.  The fresh-model C/D/E condition then
performed one ordinary `anti_transpose` incremental forward (token `7`,
position `1928`) before replaying that same frozen `flip_ud` request.  It was
also exact:

```
immediate_replay_exact             = true
foreign_call_reproduces_drift      = false
foreign_call_changes_arc_ranking   = false
max_abs_arc_logit_delta            = 0.0
```

This does **not** contradict the historical partially frozen drift.  It means
that the newly rebuilt runtime did not satisfy the historical symptom's
necessary execution condition, so it cannot be used to validate a repair.

## Concrete runtime finding

The GPU bootstrap independently passed a BF16 xFormers dispatch smoke with
`fa2F@2.5.7-pt`, but the model actually loaded for both minimal conditions
reported:

```
model.config._attn_implementation = eager
Unsloth startup: Xformers present; FA2 = False
```

Therefore, direct availability of an xFormers operator is **not** evidence
that the Qwen forward which produced the historical artifact used the same
attention route.  This is a confirmed execution-backend mismatch in the new
reproduction environment.  It is the concrete reason the call-history
diagnostic cannot yet attribute the historical effect to a particular
fast-attention/cache/CUDA mechanism.

The historical effect's lower-level cause remains `NOT_ESTABLISHED`; neither
the current result nor the eager route supports changing Dynamic-READY.

## Runtime identity observed

* GPU: RTX 3090, sm86, BF16; CUDA runtime 12.8.
* Torch: `2.8.0+cu128`; Transformers: `4.55.4`; Unsloth: `2025.9.7`.
* xFormers: `0.0.33+5d4b92a.d20260925`.
* The frozen bootstrap also emitted the pre-existing TorchAO compatibility
  warning for `torchao 0.14.1+cu128` under Torch 2.8.  It was recorded, not
  treated as a root cause.

## Invalid preliminary precheck

`invalid_argmax_precheck/` is deliberately retained only for provenance.  It
used the raw root-logit argmax rather than `start_ready_cell(...).request`, so
it is invalid for any call-history conclusion.  The valid result is solely
`exact_request/`.

## Frozen artifacts

`exact_request/CONTRACT.json`, `IMMEDIATE_REPLAY.json`,
`FOREIGN_CALL_REPLAY.json`, `DECISION.json`, and `HASHES.json` were committed
from the Linux producer.  `HASHES.json` contains the producer's SHA256 values.
Verify those bytes from a Git blob or Linux checkout: a Windows working tree
with autocrlf can change the byte representation of JSON line endings.

## Decision

1. Do not implement the proposed runtime-state repair.
2. Do not promote Dynamic B2.
3. To continue root-cause work, first recover or construct a runtime in which
   the Qwen model forward itself uses the historically relevant attention path
   and then rerun exactly this five-forward gate.  Only a reproduced drift can
   justify a mechanism-level intervention.
