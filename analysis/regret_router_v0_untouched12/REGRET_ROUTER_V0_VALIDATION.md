# Regret Router-v0 untouched12 validation

## Static-depth execution amendment

The original exact-resume route is unavailable because the frozen decoder has no external serializable search-state/KV-cache resume interface. The amended execution contract is frozen before GPU work: d12/d48 run at 1024 and d24 runs directly at 4096; separate d12/d48 4096 runs provide the shadow control. This changes execution only, not the Router-v0 depth decision.

- `RESUME_PARITY = NOT_APPLICABLE_STATIC_EXECUTION`
- `GPU_USED = NO` before the subsequent target-blind run
- Historical union remains `33/89 unchanged`.
