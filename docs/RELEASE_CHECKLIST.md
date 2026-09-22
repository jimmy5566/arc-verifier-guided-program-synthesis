# ARC2 production release checklist

This is the mandatory, binary pre-submission gate. It governs packaging and
competition submission, not ordinary research. A visible Save Version does not
prove that a hidden competition execution used the same inputs, branch, or
environment.

## 1. Freeze science, not production task IDs

Scientific configuration may be frozen before release: model identity/revision,
tokenizer, TTT method/steps/LoRA rank/alpha/target modules, Aug8 definition and
order, search, selector/B-SUPPORT, seeds, parser/inverse transforms, and
deduplication behavior.

Production task identity must not be inherited from Eval60, Frozen60, visible
competition runs, historical manifests, cached IDs, or an assumed output count.
At runtime, build the production manifest from the **actual mounted challenge**.
Task count, task IDs, and multi-test distribution are observations recorded only
after that file is read.

The following must derive from one runtime identity:

```text
runtime challenge content hash + task/test-index structure
  -> production manifest -> task queue -> worker assignment/checkpoint
  -> candidate pool -> selector -> finalizer -> submission
```

Checkpoint/cache identity must include challenge content hash, task ID,
test-index structure, model identity, scientific-config hash, and relevant
parser/generation configuration. A same-ID changed input must invalidate old
state. Deterministic file discovery is mandatory: zero or multiple plausible
challenge/model/source files must fail; `next(rglob(...))` is not a valid
selection policy.

## 2. Delivery and failure contract

`TASK_PROCESSED != VALID_PREDICTION_EXISTS`. Track expected, processed,
valid-candidate, zero-candidate, selected, failed, timed-out, unfinished, and
fallback task counts separately.

Before release, document and exercise the production policy for every case:

| Case | Required release evidence |
| --- | --- |
| Zero valid candidates | Exact documented outcome; ambiguity is a blocker. |
| Model exception / timeout | Explicit status, no swallowed exception, and documented finalizer result. |
| Incomplete/corrupt checkpoint | Identity validation rejects it; no silent reuse. |
| Invalid grid / duplicate or partial output | Rejected before finalization. |
| Existing `submission.json` / overwrite | Removed or replaced deterministically at run start; an old output cannot survive unnoticed. |
| Dummy file | Cannot masquerade as production output. |
| Fallback | Permitted only if it is an explicit frozen approved contract; structural JSON alone is never production health. |

Do not invent or change fallback semantics as part of a release audit. If the
current code has no unambiguous policy, record `RELEASE_BLOCKER`.

## 3. Required CPU behavioral release tests

Run the real production manifest/checkpoint/finalizer path using synthetic
runtime inputs. Grep, `py_compile`, and static notebook inspection alone do not
satisfy this section.

| Test | Mutation | Required outcome |
| --- | --- | --- |
| A — entire task-ID replacement | Replace all runtime IDs. | Queue, checkpoint, candidate, and submission IDs contain only replacements. |
| B — multi-test count mutation | Change test-output count/distribution. | Manifest/finalizer emits exactly runtime-required outputs. |
| C — same ID, changed input | Retain ID, mutate input. | Runtime hash changes; old cache is rejected. |
| D — stale manifest/checkpoint injection | Provide a prior identity. | Explicit identity mismatch; no silent reuse. |
| E — sample/challenge mismatch | Alter sample structure. | Explicit failure or documented deterministic reconciliation; never guess. |
| F — multi-test mapping | Trace `task_id + test_index` end-to-end. | Mapping remains exact through finalizer. |
| G — no valid candidate | Supply completed zero-candidate task. | Exact documented release policy occurs. |
| H — interrupted task | Supply incomplete worker/checkpoint state. | Cannot silently become a healthy release. |

An expected failure is evidence of a release blocker, not a passed gate.

## 4. Package and environment identity

Record and verify source commit/package hash; scientific-config hash; model and
tokenizer source/revision; Python, Torch, Transformers, Unsloth, Triton, and
ptxas versions where applicable; CUDA-visible GPU assumptions; attached
datasets/models; Internet setting; notebook slug/version; and actual challenge
path/hash.

## 5. Validation reuse matrix

| Reusable only with identical configuration | Must be rerun for production packaging |
| --- | --- |
| Model load, rank-256 LoRA construction, TTT forward/backward, Aug8 mechanics, parser basics, four-L4 worker architecture, approximate VRAM envelope | Runtime challenge binding, runtime manifest, task/test-index mapping, cache rejection, finalizer, submission schema, no-candidate/failure contract, package/environment paths, Save-Version vs competition behavior, competition attachments |

An unrun production-specific validation is not PASS.

## 6. Rerun, evidence, and resources

- Save Version status is not competition-submission status. A public score does
  not expose hidden logs or outputs.
- Do not rely on undocumented rerun environment variables. Do not reintroduce
  `KAGGLE_IS_COMPETITION_RERUN` gating without explicit user instruction.
- CPU-answerable work stays CPU-only. Frozen candidates are reused for scorer,
  report, import, and mapping diagnosis. A development GPU smoke is at most 30
  tasks unless explicitly approved; experiments never auto-expand.
- GPU launch, material Kaggle Save Version, push, and competition submission
  each require explicit user authorization.
- **This documentation-only governance round:** GPU launch, Kaggle Save
  Version, push, and competition submission are forbidden.

Use these labels precisely:

- **CONFIRMED** — directly observed in accessible logs/artifacts.
- **LOCALLY_REPRODUCED** — mechanism reproduced locally, not directly observed in hidden execution.
- **DEVELOPMENT_EVIDENCE** — Frozen/Evaluation cohort observation.
- **PUBLIC_LB_EVIDENCE** — submission status or score observation.
- **HIDDEN_HYPOTHESIS** — plausible but unobservable hidden-run explanation.

## 7. Current confirmed release blockers

1. **CONFIRMED:** the current reference-TTT production builder freezes the
   visible local 240-task cohort before runtime instead of deriving its
   manifest from the mounted challenge.
2. **CONFIRMED:** current checkpoint identity hashes task IDs and configuration,
   but not runtime challenge content or test-index structure; a same-ID changed
   input can therefore be treated as reusable state.
3. **CONFIRMED:** current source discovery uses ambiguous `next(rglob(...))`
   selection in the production notebook builder.
4. **CONFIRMED:** the strict finalizer fails closed on zero valid candidates;
   this behavior is documented, but suitability requires an approved explicit
   no-candidate release contract.

No hidden-run mechanism is asserted by these CPU findings.

## 8. Binary release gate

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

Final status must be exactly one of:

```text
RELEASE_READY
RELEASE_BLOCKED
```

Competition submission is allowed only for `RELEASE_READY`; otherwise stop.
