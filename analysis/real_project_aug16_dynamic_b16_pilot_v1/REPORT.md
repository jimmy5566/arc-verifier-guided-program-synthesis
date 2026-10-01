# REAL PROJECT AUG16 Dynamic-B16 pilot — controlled capacity stop

Target-blind hardware/scheduler/semantic pilot. Gold was never opened. The project-defined frozen AUG16 set is not the stochastic public NVARC AUG16.

## Frozen contract

- Step-0 commit: `461e61121777d251e6b96d26d6511cdd698e27e5`
- Native source/task: `d59b0160:o0:d24`
- Nominal AUG16 cells: `16`
- Unique prompt states / equivalence classes: `16 / 16` (all singleton)
- Root compatibility: `1` group, `16` compatible lanes, prompt length `1928`

## R128 — PASS

- Logical nodes: `2048`; physical forwards: `128`
- Mean / median / p25 / p75 physical batch: `16.0 / 16.0 / 16.0 / 16.0`
- B16 / B12+ / B8+ forward fractions: `1.000 / 1.000 / 1.000`
- Logical nodes/s: `114.609449`; raw-B16 reference: `138.915`; realized efficiency: `0.825033`
- Search wall / prefill / model / pack / adoption / scheduler seconds: `17.869382 / 6.174434 / 10.916295 / 0.773047 / 2.883965 / 0.003816`
- Peak allocated / reserved: `22958445568` / `24958205952` bytes
- Semantic gate: `PASS`; all 16 cells at exact 128-node accounting.

## R256 — controlled stop

- R256 was started only because R128 passed.
- Fixed B16 R256 terminated with CUDA OOM before an R256 result could be frozen; see `R256_OOM_EVENT.json`.
- No same-configuration retry was made; no physical batch, cache policy, decoder policy, adapter, prompt, target set, or scientific threshold was altered.
- R256 occupancy, throughput, semantic gate, and final B16-ready classification: `NOT_ESTABLISHED`.

## Decision

- Classification: `REAL_AUG16_R256_CAPACITY_STOP`.
- Large 3090 census: `DO_NOT_PROCEED`.
- This is a capacity outcome, not a Gold or solution-quality result.
