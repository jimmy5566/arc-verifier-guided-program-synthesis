# Historical runtime reconstruction — Dynamic-READY call-history drift

## Scope

This is a repository-only forensic reconstruction for the historical
`d59b0160:o0:d24:flip_ud` call-history drift.  It does not access Gold, load a
model, execute GPU work, modify the decoder, or run DFS.

## What is established

The historical artifact records an immediate identical replay followed by one
foreign `anti_transpose` incremental forward.  On the same frozen `flip_ud`
token (`0`), position (`1928`), and cache-content SHA
`34b3869cd02f64d1e36b3e400eb083cb6822de918ebb5c5ab31c41c5f8438d6c`,
the foreign-call replay had a full-logit delta of `38.375` and an ARC-logit
delta of `50.79766845703125`.  It changed ARC ranking and the Regret retained
set.  This is measured evidence of a call-history effect at the model-call
boundary, not evidence of a particular lower-level kernel defect.

The current valid five-forward negative control used the actual
`start_ready_cell(...)` request, not an argmax reconstruction.  It has the
same task/output/depth/views/adapter contract and source-path blobs, but its
post-foreign replay was exact (`max_abs_arc_logit_delta=0.0`).

## Reconstruction table

| Field | Historical failing runtime | Status | Evidence / limit |
| --- | --- | --- | --- |
| Exact request token / position | `0` / `1928` | MEASURED | `CALL_HISTORY_LOGIT_DRIFT.csv` |
| Historical effect | full delta `38.375`; ARC delta `50.79766845703125` | MEASURED | `CALL_HISTORY_LOGIT_DRIFT.csv` |
| Scheduler interleaving | No; first 50 B1 calls were serial `anti_transpose` | MEASURED | `SCHEDULER_SEQUENCE_SUMMARY.json` |
| Python | 3.11 family | INFERRED | The frozen Ampere lock/bootstrap used Python 3.11; no historical process dump exists |
| Torch / CUDA | `2.8.0+cu128` / 12.8 | INFERRED | Frozen Ampere lock, not a failing-process `pip freeze` |
| Transformers / PEFT / TorchAO / Triton | `4.55.4` / `0.17.1` / `0.14.1+cu128` / `3.4.0` | INFERRED | Frozen Ampere lock, not a failing-process package dump |
| Unsloth / Zoo / xFormers | `2025.9.7` / `2025.9.9` / `0.0.33+5d4b92a.d20260925` | INFERRED | Frozen Ampere lock, not a failing-process package dump |
| bitsandbytes | `0.50.2` | INFERRED | Frozen Ampere lock; it is not established as used by this BF16 load path |
| Flash Attention Python package | UNKNOWN | No historical package dump records an installed package or version |
| CUDA driver | UNKNOWN | A related runtime record has `580.126.20`; it is not a dump from the failing five-forward process |
| GPU / precision | RTX 3090 sm86 / BF16 | INFERRED | Frozen Ampere contract and B1 audit contract |
| Model class | PEFT causal-LM wrapper | INFERRED | Historical loader source is identical to current, but the failing process did not record `type(model)` |
| Model-loading flags | `full_finetuning=False`, `load_in_4bit=False`, local files, no gradient checkpointing | MEASURED | Identical historical/current `run_eval60_authoritative_greedy_v1.py` Git blob |
| `FastLanguageModel.for_inference` | Called after prompt construction | MEASURED | Identical historical/current audit source Git blob |
| Qwen attention implementation | UNKNOWN | No failing-process model-config or module-class record exists |
| Qwen attention class / forward source / hash | UNKNOWN | No failing-process module identity exists |
| Unsloth patch identity | UNKNOWN | The callable actually executed was not recorded in the failing process |
| torch.compile / CUDA Graph state | UNKNOWN | No failing-process runtime flags were captured |
| Cache / RoPE / causal-mask internals | UNKNOWN | Only request/cache content and invocation arguments were captured |

## Source-path comparison

These science-path files are byte-identical Git blobs in historical source
commit `dfea27d923651be09caa31e8ac041df1706ebcdc` and current preflight commit
`0293382173000c1a31eaee52320211fc941db3de`:

| Path | Git blob |
| --- | --- |
| `src/inference/nvarc_turbodfs_dynamic_ready.py` | `fc898729407198eb3cadf1016e9e4e5f9e2cee69` |
| `scripts/run_eval60_authoritative_greedy_v1.py` | `4f0c94a042125648ee4586f45f261468097f03b7` |
| `scripts/run_eval60_adaptive_inference_joint_v2.py` | `aeddc1337897fed161f6122722eedb2d2dca6419` |
| `scripts/run_regret_dynamic_ready_v1.py` | `018d0d62d4ed12aa2e4b10e54da29155e49e8716` |
| `scripts/run_d1_real_decoder_ab.py` | `54953263452ba8ac11c11e248e4165dfde2897d5` |

Thus the repository evidence does not support a decoder, prompt, adapter-load,
or Dynamic-READY source difference as the historical execution condition.

## Current negative control (not a repair)

The only measured live identity is the valid `exact_request` negative control:
RTX 3090/sm86, driver `595.71.05`, Python `3.11.13`, Torch `2.8.0+cu128`, CUDA
12.8, Transformers `4.55.4`, Unsloth `2025.9.7`, Zoo `2025.9.9`, xFormers
`0.0.33+5d4b92a.d20260925`, and `model.config._attn_implementation="eager"`.
Its model wrapper forward is `unsloth.models.llama.PeftModel_fast_forward`.
Its generic first-attention probe was null because the PEFT/Unsloth wrapper
does not expose Qwen blocks at that attempted path; this is not evidence about
the historical attention class.

The xFormers BF16 dispatch smoke selected `fa2F@2.5.7-pt`, but that only proves
that an xFormers operator exists.  It does not prove that Qwen used it.

## Gate decision

There is no recovered historical container, process-level package dump,
attention-module identity, or driver identity sufficient to construct a
materially different runtime that is supported by evidence.  Reinstalling an
arbitrary non-eager backend would therefore be a hypothesis experiment, not a
historical reconstruction.  Per the protocol stop condition, Phase 3 and all
later GPU phases are **NOT_RUN_DUE_TO_FAILED_REPRO_GATE**.

Decision: `HISTORICAL_RUNTIME_NOT_RECONSTRUCTED`.
