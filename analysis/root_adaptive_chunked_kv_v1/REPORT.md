# Root-adaptive Chunked KV V1

Target-blind engineering validation.  No evaluation solutions or Gold were loaded.

## Cohort root lengths

- min / median / p90 / max: `605 / 2198 / 3996.0 / 8400`
- S / M / L / XL / XXL / OVERSIZE: `37 / 20 / 29 / 1 / 0 / 0`

## Decision

- Classification: `ROOT_ADAPTIVE_CHUNKED_KV_PARTIAL_PASS`
- Retention30: `DO_NOT_PROCEED`
- Canary R128 parity: `PASS`
- Canary R256: `COMPLETE`
- Canary R512: `COMPLETE`
- Canary R4096: `COMPLETE`

## Non-S profile smokes

- PROFILE_M: `WORKER_FAILED_NO_RESULT`
- PROFILE_L: `WORKER_FAILED_NO_RESULT`
- PROFILE_XL: `WORKER_FAILED_NO_RESULT`
- PROFILE_XXL: `PROFILE_UNAVAILABLE`
