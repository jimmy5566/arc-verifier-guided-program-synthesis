# Runtime backend audit — Regret Dynamic-READY

## Scope and status

This is Phase 0 of `MINIMAL_CALL_HISTORY_REPRO_V1`.  It is a source and
environment-identity audit only.  It does not access Gold, generate a
candidate, or change a decoder/search policy.

* Authoritative source commit: `2acb367eae8e71ecd9e9affb938409a010ac5bc0`.
* Frozen experiment condition: Qwen3-4B, BF16, `d59b0160:o0:d24`, Regret4,
  `max_expanded_nodes=4096`, candidate cap 32.
* `MECHANICALLY_OBVIOUS_BUG = NO`.
* `GPU_MINIMAL_REPRO_REQUIRED = YES`.

The existing partial evidence establishes a **runtime call-history numerical
effect**, not its lower-level implementation cause: an otherwise identical
incremental `flip_ud` request changes after one ordinary `anti_transpose`
incremental call, although immediate replay is bit-identical and the recorded
token, position and root-cache content are unchanged.  The prior KV-isolation
audit did not establish root-cache storage aliasing, cross-prefill mutation,
idle-root mutation, or audited model-metadata mutation.

## Frozen environment identity

The pinned Ampere runtime contract in `requirements-turbodfs-v5.lock` and
`scripts/bootstrap_3090_ampere_env_v2.sh` is:

| Component | Frozen identity |
| --- | --- |
| Python | 3.11 |
| GPU contract | RTX 3090, sm86, BF16 |
| Torch | `2.8.0+cu128` |
| Transformers | `4.55.4` |
| PEFT | `0.17.1` |
| TorchAO | `0.14.1+cu128` |
| Triton | `3.4.0` |
| Unsloth | `2025.9.7` |
| Unsloth Zoo | `2025.9.9` |
| xFormers | `0.0.33+5d4b92a.d20260925` |
| xFormers wheel SHA256 | `3f4609e50df81543e545f1e925a1bdce61dca9fe419aeb22b95d81d3dc1c123c` |

The bootstrap rejects a non-sm86 device and dispatches a BF16 xFormers
forward/backward smoke through `xformers.ops.fmha.dispatch._dispatch_fw`.  It
requires a backend name beginning `fa`; a generic PyTorch fallback is not
accepted.  Exact function-level inspection of the patched Unsloth/Qwen module
is deliberately deferred until the local, executable Ampere venv has passed
that contract.

## Static execution-path audit

`scripts/run_eval60_authoritative_greedy_v1.py` loads the model through
`FastLanguageModel.from_pretrained(..., full_finetuning=False,
load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False)`;
then attaches a PEFT adapter with rank 256, alpha 32, zero dropout,
`random_state=42`, and `use_rslora=True`.  Float32 parameters are cast to BF16.

`src/inference/nvarc_turbodfs_dynamic_ready.py` has the following relevant
semantics:

1. `start_ready_cell` performs a scalar root prefill with `use_cache=True` and
   exposes `outputs.past_key_values` (optionally legacy-normalized) to the
   ready-cell generator.
2. A single ready request calls the same model object with a one-token
   `input_ids`, explicit one-token `position_ids`, the request
   `past_key_values`, `return_dict=True`, and `use_cache=True`.
3. The B1 branch does **not** pass an explicit `attention_mask` or
   `cache_position`.  This is also the established historical B1 invocation,
   so changing it would be a semantic change until parity is proven.
4. The B2 branch concatenates/splits caches, but the observed partial history
   contains 50 B1 calls and no actual coroutine interleaving.  B2 machinery is
   therefore not needed to reproduce the known symptom.

The same explicit-position incremental convention appears in the scalar
TurboDFS code paths.  It is a plausible boundary for the minimal diagnostic,
but not a mechanically proven fault: the immediate same-request replay is
exact under that convention.

## Candidate causes ranked by current evidence

| Candidate | Status | Why |
| --- | --- | --- |
| Fast-attention/Unsloth/CUDA call-history numerical state | Plausible, unproven | The logical request and recorded root-cache content match, yet logits drift after a foreign incremental forward. |
| Hidden cache or model metadata not covered by prior recursive tensor/content hashing | Plausible, unproven | Prior checks rule out listed tensors/aliases, not every backend-private property or opaque cache field. |
| CUDA allocator/stream/workspace lifecycle effect | Plausible, unproven | The effect is call-history dependent; no allocator/stream/workspace comparison has yet been recorded. |
| Python scheduler/coroutine interleaving | Less likely for the captured symptom | The recorded first 50 calls are serial B1 calls of one cell, and an isolated foreign call was sufficient to produce drift. |
| Direct root-cache storage aliasing | Not supported | The frozen KV-isolation audit did not find it; deep-cloning roots still yielded only 1/4 shared parity. |

## Required minimal live discriminator

On the verified frozen runtime, use a fresh model and one frozen `flip_ud`
request:

1. Record a baseline incremental forward.
2. Immediately replay the same request and require exact full-logit equality.
3. On a second fresh model, record the same baseline, execute exactly one
   ordinary `anti_transpose` incremental forward, then replay `flip_ud`.

For every forward, record token, position, input dtype/device/shape/stride,
all accessible cache-tensor shape/dtype/device/stride/storage-offset/
contiguity/content SHA, attention-mask/cache-position/position-ID arguments,
and backend/model flags.  The third step is a gate: do not implement or
promote a fix unless it reproduces the previously observed drift on the frozen
runtime.

## Decision

No source-level change is justified from static evidence.  The smallest safe
next action is the three-forward GPU discriminator above; it separates a
reproducible backend/runtime effect from a stale or incomplete historical
capture without changing Dynamic-READY semantics.
