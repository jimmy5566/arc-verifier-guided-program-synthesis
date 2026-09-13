# ARC2 native likelihood ranker v2 fast

## Scope

This experiment changes only candidate ranking.  Candidate generation remains
the existing native, deterministic, eight-augmentation runner; no TTT was
enabled.  Each existing candidate was teacher-forced under four fixed,
reversible geometry views (identity, 90°, 180°, and 270° rotations).  The
candidate grid is transformed into the corresponding view before scoring.

Methods are original likelihood, multi-view mean, median, trimmed mean, and a
precommitted calibrated likelihood.  Calibration is a softmax over each view's
leave-one-train-pair-out likelihood.  It never sees a test output.  Mean,
median, and trimmed mean are reported as robustness controls; the calibrated
method was fixed before the untouched run.

## Smoke: frozen development candidates

The smoke reused the immutable two-task candidate pool (`d406998b`,
`d4c90558`) and generated no grids.  Multi-view reranking took 24.01 seconds
on one L4.  It changed `d406998b`'s original top candidate from index 15 to
index 4 (zero-based) for every V2 aggregation.  After the rerank artifact was
frozen, exact scoring showed:

| Method | Top-1 | Rank@3 | Rank@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Original likelihood | 0/2 | 0/2 | 1/2 | 0.10 |
| Calibrated / mean / median / trimmed | 1/2 | 1/2 | 1/2 | 0.50 |

The rescued candidate is `d406998b`; `d4c90558` had no correct candidate.
This smoke selected no weights and did not alter the fixed untouched method.

## One untouched evaluation

The cohort was selected and committed before candidate generation in
`7832102`: `ad173014`, `e50d258f`, `aaef0977`, `6f473927`, `bd14c3bf`,
`9356391f`, `3d6c6e23`, `3cd86f4f`.  It excludes frozen30, prior ranker, and
prior TTT-verifier cohorts.  Cohort hash:
`f260a9096d52f3269d69dc511883f288fb9ab4c7ac0ae5a84501ba6cbe1e0e20`.

Candidate generation was unchanged (2 L4 workers, 8 existing augmentations),
producing 36 unique candidates in 795.99 seconds.  The immutable candidate
artifact was uploaded as a private source and reranked once on one L4.  The
rerank froze after 89.81 seconds; only then were solutions opened locally.

| Method | Top-1 | Rank@3 | Rank@5 | MRR | Candidate existed but not Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original likelihood | 2/8 | 2/8 | 3/8 | 0.28125 | 1 |
| Multi-view mean | 2/8 | 3/8 | 3/8 | 0.29167 | 1 |
| Multi-view median | 2/8 | 3/8 | 3/8 | 0.31250 | 1 |
| Multi-view trimmed mean | 2/8 | 3/8 | 3/8 | 0.31250 | 1 |
| Calibrated likelihood (fixed V2) | 2/8 | 3/8 | 3/8 | 0.29167 | 1 |

Any-of-K is 3/8 for every ranking method: correct candidates exist for
`6f473927`, `aaef0977`, and `bd14c3bf`.  `aaef0977` remains a ranking miss at
rank 3 under calibrated/mean and rank 2 under median/trimmed.  Five tasks have
no correct candidate and are generation failures.

The V2 method rescued zero original Top-1 failures and harmed zero previously
correct Top-1 outputs.  Its extra ranking wall time is 89.81 seconds, about
11.3% of the separate candidate-generation wall time (not a same-GPU
throughput comparison: generation used two L4 workers and reranking one).

## Audit and outcome

Every prediction candidate artifact was frozen before multiview scoring, and
the multiview artifact was frozen before `load_solutions` was imported.  No
task-specific features, test targets, TTT, or post-result reruns were used.

The principal remaining blocker is candidate coverage (5/8 tasks have no
correct candidate).  For the remaining ranking failure, robust aggregation
improves rank but not Top-1.  The one allowed untouched evaluation is complete;
this experiment stops here.

Artifacts:

- `native_multiview_ranker_untouched_candidates_v1_output/UNTOUCHED_V2_CANDIDATES_FROZEN.json`
- `native_multiview_ranker_untouched_rerank_v1_output/UNTOUCHED_V2_MULTIVIEW_RERANKED_FROZEN.json`
- `ARC2_NATIVE_LIKELIHOOD_RANKER_V2_UNTOUCHED_SCORED.json`
