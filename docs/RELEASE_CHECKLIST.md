# ARC2 production release checklist

This checklist governs packaging and submission, not normal research. It is
the required gate before a Kaggle competition submission. A visible Save
Version is not evidence that a hidden competition execution used the same
inputs, branch, or environment.

## Release contract

Freeze the **scientific configuration**, not a visible task cohort. The
configuration may include model identity and revision, tokenizer, TTT method,
steps, LoRA rank/alpha/target modules, Augmentation-8 definition and order,
search, selector/B-SUPPORT logic, parser/inverse transforms, deduplication,
and seeds.

At production runtime, build the production manifest from the mounted
challenge file. Never reuse task IDs, task count, test-output count, or
multi-test distribution from Eval60, Frozen60, a visible competition run, a
cached manifest, or a historical submission. Values such as 240 tasks or 259
outputs are observations only after reading the runtime input.

The required identity chain is:

```text
runtime challenge hash + task/test-index structure
  -> manifest -> queue -> worker checkpoint -> candidate pool
  -> selector -> finalizer -> submission
```

A checkpoint/cache identity must contain the runtime challenge content hash,
task ID, test-index structure, model identity, scientific-config hash, and
relevant parser/generation configuration. Same task ID with a changed input
must invalidate prior state. Ambiguous file discovery such as choosing the
first `rglob()` match is forbidden: zero or multiple plausible files must
fail explicitly.

`TASK_PROCESSED` is not `VALID_PREDICTION_EXISTS`. Track expected, processed,
valid-candidate, zero-candidate, selected, failed, timed-out, unfinished, and
fallback task counts separately.

## Required CPU behavioral tests

Run the production-path mutation suite; static inspection and `py_compile`
alone do not satisfy this gate.

| Test | Mutation and required outcome |
| --- | --- |
| A — task-ID replacement | Replace every challenge ID. Queue, checkpoints, candidates, and final submission must contain only new IDs. |
| B — multi-test mutation | Change the count/distribution of test examples. Manifest and finalizer must emit exactly the runtime-required outputs. |
| C — same ID, changed input | Mutate content while retaining task ID. Runtime input hash must change and stale checkpoint reuse must be rejected. |
| D — stale injection | Supply prior manifest/checkpoints. Identity mismatch must fail explicitly; no silent reuse. |
| E — sample/challenge mismatch | Fail explicitly or perform documented deterministic reconciliation; never guess. |
| F — test-index mapping | Trace task ID plus test index through manifest, worker, candidate, selector, and finalizer. |
| G — zero candidates | Exercise a completed task with no valid candidates and verify the documented release policy. |
| H — interrupted task | Exercise incomplete worker/checkpoint state and verify that it cannot silently become a healthy release. |

The tests must use the real production manifest/finalizer/checkpoint code with
synthetic runtime inputs, not only string assertions. An expected failure is
evidence of a release blocker, not a passed release check.

## Delivery and failure contract

Before release, document and test the exact policy for zero valid candidates,
model exceptions, timeout, incomplete/corrupt checkpoints, invalid grids,
duplicate/partial outputs, old `submission.json`, and output overwrite.

- A new release must deterministically remove or replace prior output; an old
  submission cannot silently survive.
- Dummy artifacts must never masquerade as production output.
- Exceptions cannot be swallowed and structural JSON validity alone cannot
  claim production health.
- A fallback is permitted only when it is explicit in the frozen approved
  production contract. This checklist does not invent or alter fallback
  policy; ambiguity is a release blocker.

## Packaging and environment identity

Record and verify: source commit and package hash, scientific-config hash,
model source/revision, tokenizer source, Python/Torch/Transformers/Unsloth/
Triton versions as applicable, ptxas path/version, CUDA-visible GPU
assumptions, mounted datasets/models, Internet setting, notebook slug/version,
and the challenge path and hash actually used.

## Validation reuse matrix

| Reusable only when configuration identity matches | Must be revalidated for production packaging |
| --- | --- |
| Model load, rank-256 LoRA construction, TTT forward/backward, Aug8 generation mechanics, parser basics, 4xL4 worker architecture, approximate VRAM envelope | Runtime challenge binding, dynamic manifest, task/test-index mapping, stale-state rejection, finalizer, submission schema, no-candidate/failure policy, package/environment paths, Save-Version versus competition behavior, and competition attachments |

An unrun production-specific validation is not PASS.

## Rerun and evidence rules

- Save Version status is not competition-submission status.
- A visible notebook completion does not prove hidden execution behavior; a
  public score does not expose hidden logs or outputs.
- Do not rely on undocumented rerun environment variables or reintroduce
  `KAGGLE_IS_COMPETITION_RERUN` gating without explicit user instruction.

## Resource discipline

- CPU-answerable investigation stays CPU-only; frozen candidates must be
  reused for scorer, report, import, and mapping diagnosis.
- A development GPU smoke is at most 30 tasks unless the user explicitly
  authorizes a larger run. No experiment expands automatically.
- GPU launch, Kaggle Save Version, push, and competition submission each need
  explicit user authorization. For this governance round: all are forbidden.

Use these labels in reports:

- **CONFIRMED** — directly observed in accessible logs/artifacts.
- **LOCALLY_REPRODUCED** — reproduced locally, not directly observed hidden.
- **DEVELOPMENT_EVIDENCE** — Frozen/Evaluation cohort observation.
- **PUBLIC_LB_EVIDENCE** — submission status/score observation.
- **HIDDEN_HYPOTHESIS** — plausible but unobservable hidden-run explanation.

## Current known status

The current reference-TTT production path has CPU-reproduced release blockers:

1. **CONFIRMED:** its builder creates a manifest from local visible challenge
   files and enforces a 240-task contract before runtime.
2. **CONFIRMED:** its checkpoint identity is derived from task-ID hash and
   config, not challenge content/test-index identity.
3. **CONFIRMED:** strict finalization fails closed for a zero-candidate task;
   that is its current documented behavior, but its production suitability
   still requires an explicit, approved release contract.

No hidden-run cause is asserted by these CPU findings.

## Binary gate

Mark every item truthfully:

- [ ] Scientific config frozen
- [ ] Runtime challenge identity verified
- [ ] Dynamic production manifest verified
- [ ] Mutation/behavioral CPU tests PASS
- [ ] Cache/checkpoint identity PASS
- [ ] Packaging/environment identity PASS
- [ ] No-candidate/failure contract explicit
- [ ] Finalizer behavior PASS
- [ ] Submission schema PASS
- [ ] Output overwrite hygiene PASS
- [ ] No dummy/stale files
- [ ] Production-specific tests actually run
- [ ] Remaining blockers = NONE

Final status is exactly one of:

```text
RELEASE_READY
RELEASE_BLOCKED
```

Competition submission is allowed only for `RELEASE_READY`; otherwise stop.
