# ARC2 native TTT and strong verifier v1

## Scope and isolation

This run evaluated a generic train-pair relation verifier and a per-task native
LoRA adaptation path for `qwen3_4b_grids15_sft139`.  Candidate generation,
deduplication, likelihood scoring, relation-verifier reranking, and exact
scoring are separate phases.  All candidate artifacts were frozen before any
solution file was opened.  The verifier uses only grid shape deltas, color
sets, and quantized same-shape change rates observed in train pairs; it has no
task IDs or ARC operation rules.

The clean cohort was committed before inference in `984d176`:
`d5d6de2d`, `88a10436`, `e41c6fd3`, `25ff71a9`, `a64e4611`.  It excludes the
earlier development, legacy held-out, ranker, and frozen30 cohorts.  Its hash
is `348b108d1796a2d36dc26827ed36aa1a21fc563ffdffbd195bc866022ca76580`.

## Deterministic TTT safety gate

The isolated L4 memory/reproducibility smoke passed before this validation.
It used a short task at one step twice plus `8719f442` at one and 24 steps
twice, with four augmentations and no test-candidate generation.  Peak
allocated memory was 7,551,898,112 bytes (7.03 GiB), below the 21 GiB gate.
All explicit criteria passed: `memory_safe`, `finite_loss`,
`adapter_updated`, `base_model_unchanged`, and `deterministic_reset`.

TTT freezes every base parameter, reinitializes adapters and AdamW per task,
uses fixed CPU/CUDA seeds, disables cache during fitting, restores model state,
and clears task-local optimizer/gradient state.  In the clean validation every
task again reported an updated adapter and unchanged base fingerprint.

## Development evidence

The earlier five-task development baseline had Any-of-K 3/5 and Top-1 1/5 for
likelihood, verifier, and fusion.  Its two candidate misses were
`67a423a3` and `d4c90558`; applying the fixed TTT path to those two did not
convert either (0/2 Any-of-K).  The separate two-task TTT smoke did convert
one candidate (`d406998b`), establishing that the end-to-end TTT candidate
path can work without a target.

The generic verifier caused zero rescues and zero harms in those development
artifacts.  It is therefore retained as an auditable train-only comparator,
not claimed as a ranking improvement.

## One clean untouched validation

Both kernels used two mapped L4 workers (`worker0 -> GPU0`, `worker1 -> GPU1`),
the same fixed native prompt, eight reversible augmentations, and no Internet.
The model occupied 6,930.5 MiB per worker.  The source was the private,
solution-free Kaggle dataset version 8.

| Condition | Wall runtime | Unique candidates | Top-1 likelihood | Top-1 verifier | Top-1 fusion | Any-of-K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 949.18 s | 23 | 0/5 | 0/5 | 0/5 | 1/5 |
| TTT, all five tasks target-blind | 1,153.81 s | 21 | 1/5 | 1/5 | 1/5 | 2/5 |
| Baseline + TTT union | — | 34 | 1/5 | 1/5 | 1/5 | 2/5 |

Baseline candidate coverage was only `a64e4611`.  TTT retained that hit and
added `88a10436`; it did not produce a candidate for `25ff71a9`, `d5d6de2d`,
or `e41c6fd3`.  This is a generation-recall improvement of one task (1/5 to
2/5), not an oracle solver score.  The top candidate on `88a10436` was exact,
but the existing candidate on `a64e4611` was not selected.  Thus the clean
validation also confirms the ranking bottleneck remains.

The union normalizes likelihood within each source/task, exactly deduplicates
predictions, and is then reranked by the same train-only verifier.  Its status
was frozen before exact scoring.  It did not improve over the TTT pool.

## Artifact chain

- Baseline candidates: `native_ttt_verifier_untouched_v2_baseline_v1_output/UNTOUCHED_V2_BASELINE_CANDIDATES_FROZEN.json`
- Baseline train-only rerank: `ARC2_NATIVE_TTT_VERIFIER_UNTOUCHED_V2_BASELINE_RERANKED_FROZEN.json`
- Baseline score: `ARC2_NATIVE_TTT_VERIFIER_UNTOUCHED_V2_BASELINE_SCORED.json`
- TTT candidates: `native_ttt_verifier_untouched_v2_ttt_v1_output/UNTOUCHED_V2_TTT_CANDIDATES_FROZEN.json`
- TTT train-only rerank: `ARC2_NATIVE_TTT_VERIFIER_UNTOUCHED_V2_TTT_RERANKED_FROZEN.json`
- TTT score: `ARC2_NATIVE_TTT_VERIFIER_UNTOUCHED_V2_TTT_SCORED.json`
- Combined candidates/rerank/score: `ARC2_NATIVE_TTT_VERIFIER_UNTOUCHED_V2_COMBINED_*.json`

## Outcome

The primary remaining blocker is low native candidate coverage and, separately,
selection among valid candidates.  TTT is now memory-safe and reproducible and
showed one clean target-blind candidate conversion, but it did not establish a
reliable broad recall gain.  The current generic verifier did not rescue a
ranking failure.  No task-specific rules, test-target access during inference,
or frozen-cohort tuning were used.
