# REGRET_DYNAMIC_READY_KV_ISOLATION_AUDIT_V1

Target-blind diagnostic. Gold was not opened and Dynamic Batch2 was not instantiated.

- Root storage aliases across views: False
- Cross-prefill mutation: False
- Idle-root mutation during 1/8/32/128 incremental forwards: False
- Model mutable-state entries changed: 0
- Root-clone shared B1 parity: 1/4
- First scheduled cell exact in both construction orders: True
- Root-cause classification: MIXED_OR_MODEL_STATE_INTERACTION

## Scope

- Cell family: d59b0160:o0:d24 {identity, flip_ud, transpose, anti_transpose}
- Decoder: fixed CUMULATIVE_REGRET_r=4.00, 4096 nodes, 32 candidates
- Dynamic Batch2: not run
- Gold: not accessed
- Untouched12/Untouched24: not used
