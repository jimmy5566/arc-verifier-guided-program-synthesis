# Local CPU analysis of the partial shared-B1 call-history audit

## Scope and status

This analysis uses only the target-blind files frozen in
`e45b2d6c2212e69e028ff5b3926deb48d028879b`, the preceding B1.1/KV-isolation
artifacts, and static source inspection. No model was loaded and no GPU or
Gold was used locally.

The remote GPU run was stopped before the forced-first, prefill-after-history,
and first-per-forward trace phases. Consequently this report establishes the
root cause at the system boundary but does not identify the exact internal
CUDA/Unsloth/xFormers implementation defect.

## Facts rechecked locally

1. `SCHEDULER_SEQUENCE.csv` contains 50 scalar selections, all for
   `d59b0160:o0:d24:anti_transpose`. The decoder's scheduler sorts ready cells
   by `(cell_key, request.ordinal)`, so this is consistent with serial lexical
   servicing, not coroutine alternation.
2. `SAME_REQUEST_REPLAY.csv` and its summary show two immediate replays of the
   same deep-cloned `flip_ud` request with equal full-logits hashes and zero
   full/ARC-logit/ARC-logprob deltas. The request itself is reproducible.
3. The history probe at zero anti forwards is also bit-identical.
4. After exactly one ordinary `anti_transpose` incremental model forward, the
   later replay uses the same recorded flip token (`0`), position (`1928`), and
   cache content SHA (`34b386…f8438d6c`) but differs in full-logits SHA,
   `max_abs_full_logit_delta=38.375`, and
   `max_abs_arc_logit_delta=50.79766845703125`.
5. The prior KV-isolation audit had already found no root storage alias, no
   cross-prefill mutation, no idle-root mutation, and no audited model-metadata
   mutation. Deep root cloning had not repaired shared B1 parity.

## Static code boundary

`start_ready_cell` creates the root prefill and yields a `ReadyForwardRequest`.
The B1 scheduler then calls the same `model(...)` object with the request's
token, position, and supplied `past_key_values`. It does not intentionally
write a different cell's request/cache. The inspected runtime enables Unsloth
inference on BF16 RTX 3090 with xFormers, and does not install a
deterministic-algorithms guard around those forwards.

## Root-cause classification

**MODEL_RUNTIME_CALL_HISTORY_NUMERICAL_EFFECT — supported at the model-call
boundary.** A prior unrelated incremental forward changes output logits for a
replayed, content-identical request, while immediate replays are deterministic.
This explains why the serial shared B1 run can drift after `anti_transpose`
completes even though it does not interleave coroutines.

The following more specific attribution is **not established** from the frozen
evidence: xFormers kernel state, Unsloth patch/cache state, CUDA runtime state,
or an unmeasured `past_key_values` metadata/property. The stopped forced-first
and prefill-after-history controls are the minimal live tests that would split
those explanations.

## Consequence

Do not use shared B1/ready-cell multi-view execution as parity evidence under
this runtime. Existing isolated-cell behavior remains the validated path.
The next live diagnostic, if authorized later, should run only the missing
forced-first and newly-prefilled-after-anti controls with per-forward request
logging; it should not change Regret policy, precision, or decoder budgets.
