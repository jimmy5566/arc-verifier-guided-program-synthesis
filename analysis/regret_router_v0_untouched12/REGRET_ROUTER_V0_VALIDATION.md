# Regret Router-v0 untouched12 validation

## Controlled prelaunch stop

The Router-v0 contract and leakage-free 12-output cohort are frozen. GPU validation did not start: the frozen decoder has no external serializable search-state/KV-cache resume interface. The required 1024 checkpoint to 4096 exact-resume parity therefore cannot be run. Restarting at node zero would violate the registered protocol.

- `RESUME_PARITY = NOT_RUN`
- `GPU_USED = NO`
- Historical union remains `33/89 unchanged`.
