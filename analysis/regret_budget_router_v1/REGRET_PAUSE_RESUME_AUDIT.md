# Regret pause/resume static audit

## Result

- PAUSE_RESUME_FEASIBLE: NO for the current public D1 TurboDFS interface.
- PAUSE_RESUME_PARITY: NOT_RUN (no GPU execution authorized or needed for this static result).

`inference_d1_turbo_dfs` creates a new local state dictionary and root KV cache for every call. The recursive `d1_turbo_dfs` owns the frontier candidate lists, prefixes, scores/regrets, parent nodes, `past_key_values`, counters, dedup/trace-pending state, and clock state only as call-local values. There is no public checkpoint/resume object or continuation API.

An implementation would need to retain, in one live process: frontier paths and order, candidate pool/dedup state, prefix tokens, path NLL/regret, parent node IDs, counters, termination/budget state, trace state, and the current KV cache. Disk serialization of KV is not required for an immediate in-process router decision, but parity must be proven before deployment.
