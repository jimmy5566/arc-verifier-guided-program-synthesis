# Frozen60 augmentation ablation: 8 vs 16 vs 32

**Status:** frozen on 2026-09-19.  This note records a target-blind
augmentation-count ablation before any subsequent Kaggle submission work.

## Scope and protocol

- Cohort: the immutable `untouched60` training cohort, 60 tasks, task-ID hash
  `af9a47ea80c5fad7bdc9b8cc547d73f034923788df573e479dbaab5cb75014c8`.
- Model, tokenizer, prompt serialization, seeds, greedy decoding, candidate
  budget per augmentation, B-SUPPORT selector, and four-worker L4 topology
  were held constant.  Only the deterministic nested augmentation prefix was
  varied: 4, 8, or 16.  The historical 32-augmentation pool was reused rather
  than regenerated.
- Every candidate pool and B-SUPPORT selection artifact was validated and
  frozen before `arc-agi_training_solutions.json` was opened for scoring.
- The recovered Kaggle run initially failed only in the post-freeze reporting
  path.  Its frozen 4/8/16 candidate and selection artifacts were downloaded
  intact and scored locally without CUDA, model loading, reranking, or new
  candidate generation.

## Frozen results

| Augmentations | Any-of-K | Top-1 | Top-2 | Runtime | Relative to 32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 21/60 | 19/60 | 19/60 | 1,597.772 s | 0.167x |
| 16 | 28/60 | 17/60 | 21/60 | 3,162.455 s | 0.330x |
| 32 (historical reuse) | 30/60 | 16/60 | 21/60 | 9,577.911 s | 1.000x |

`32` supplies two additional diagnostic Any-of-K hits versus `16`, but no
Top-2 gain and about 3.0x the wall time.  It is therefore **deprecated for
production use**.  Its historical artifacts remain immutable and must not be
deleted.

`16` preserves the best observed Top-2 score (`21/60`) at roughly one third
of the historical 32-augmentation runtime.  It is the current
accuracy/runtime production candidate.  `8` loses two Top-2 task hits versus
`16`, while being about 2.0x faster than `16` and about 6.0x faster than `32`;
it is the preferred configuration for the next minimum-cost diagnostic
submission investigating the unexplained public-LB 0.00 result.

The final competition configuration remains an explicit **8 vs 16** decision;
this note does not authorize a production change or a Kaggle submission.

## Provenance and preserved artifacts

All paths below are local, preserved artifacts.  SHA256 values refer to the
exact recovered files used for scoring.

| Condition | Candidate artifact | SHA256 | B-SUPPORT selection artifact | SHA256 |
| ---: | --- | --- | --- | --- |
| 4 | `artifacts/frozen60_aug_ablation_v2_failed_output/artifacts/frozen60_aug_ablation/aug4/A_candidates_frozen.json` | `3f2799c5f9452cd75894b86f53380f051ba72a913664f0494f49d766f4ddef6b` | `.../aug4/B_support_selection_frozen.json` | `05d7e3608bb445956baf7554e37f5a8271e1280b628c43542529c207b29f701f` |
| 8 | `artifacts/frozen60_aug_ablation_v2_failed_output/artifacts/frozen60_aug_ablation/aug8/A_candidates_frozen.json` | `21fdd67a6c3adc34572dbabf77c0f536d4f6421296c27c32e20ac2be33a7b8a2` | `.../aug8/B_support_selection_frozen.json` | `b29fb0f611e09d1da71a10e771f76b346ff6a85d40b1821a5d0d63ed7dda08c5` |
| 16 | `artifacts/frozen60_aug_ablation_v2_failed_output/artifacts/frozen60_aug_ablation/aug16/A_candidates_frozen.json` | `4544e38888626b043eb8640b861613f6afae4358b1c16543f9dbb3c42e9b49ab` | `.../aug16/B_support_selection_frozen.json` | `df34597103ab3af1c6811b4801fb2580a07794f9d4698b50d8e7818fc4284b37` |
| 32 historical | `artifacts/untouched60_v33_output_final/artifacts/untouched60/A_candidates_frozen.json` | `72b3e856cdc1259fd524d314a9109a1eecbdbbe1301d6230a500c97cd5ae5f15` | `artifacts/frozen60_support_ablation/B_SUPPORT_predictions_frozen.json` | `ede89abe402062091dad9fecfce8842c8f6032915ac338e5aee87fd283a52816` |

- Frozen cohort manifest:
  `artifacts/untouched60_manifest.json`, SHA256
  `4e361abf8b99db06b506b903bea217a2431d137d3361a90c58379e3ee090cd98`.
- Historical 32 runtime source:
  `artifacts/untouched60_v33_output_final/artifacts/untouched60/B_selection_frozen.json`,
  SHA256 `643d989d72e837588ca34f865120f7e2887c9895e27d474c94a45fcb63b6b298`.
- Recovered final report:
  `artifacts/frozen60_aug_ablation_v2_failed_output/artifacts/frozen60_aug_ablation/AUGMENTATION_ABLATION_REPORT.json`,
  SHA256 `a77a3a0690e3c97275929dc64d1ed074d3cef56e9d3b42066459da902fe6c015`.

## Reproducibility boundary

`scripts/run_frozen60_augmentation_ablation.py` and
`scripts/score_frozen60_augmentation_ablation.py` are experiment/reporting
tools.  They do not modify the production inference implementation.  The
reporting script now explicitly resolves the local `src/` package, preserves
a single historical candidate as a duplicated second attempt for evaluation,
and writes a deterministic union schema for paired CSV rows.  These changes
fix only post-freeze artifact scoring/reporting.

No GPU inference, Kaggle submission, production inference change, candidate
generation, reranking, or target-informed method change occurred while
creating this freeze note.
