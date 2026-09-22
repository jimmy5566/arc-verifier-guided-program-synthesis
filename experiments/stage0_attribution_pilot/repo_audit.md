# Stage-0 repository audit

## Scope and isolation

This pilot is restricted to `experiments/stage0_attribution_pilot/`.  It uses
cached local artifacts only.  It does not invoke a model, CUDA, a Kaggle CLI or
API, notebook tooling, submission code, candidate generation, or any production
ARC execution path.

## Reproducible frozen sources

1. `artifacts/ARC2_STRATEGY_AWARE_RANKER_FROZEN30_STRATEGY_ONLY_RERANKED_FROZEN.json`
   contains 30 keyed `records`, each with the frozen candidate grids,
   `candidate_scores`, `ranked_candidate_indices`, augmentation metadata, and
   ranking/generation telemetry.  Its documented configuration SHA-256 is
   `c9b07d78f1cc1adaaa7eb1858a2e1301bfc6f9ea7ad42c67415a50abd150cf28`.
2. `artifacts/ARC2_NATIVE_RANKER_FROZEN30_FORENSICS.json` independently records
   `any=21`, `top_at_1=9`, and 30 tasks, and documents that the old artifact was
   inspected before local solutions were imported.
3. `data/raw/arc-agi_training_solutions.json` provides already-local labels used
   only to exact-score the loaded frozen candidate grids.

## Independent reproduction

`src/extract_arc30.py` first loads all frozen candidates/rank order and only
then opens the cached local solution file.  It recomputes Any-of-K from every
candidate and Top-1 from `ranked_candidate_indices[0]`; it does not trust a
summary field.  The reproducible expected result is 21/30 Any-of-K and 9/30
Top-1.  The generated CSV carries the source artifact SHA-256 per row.

## Available and unavailable signal

Available: candidate grids, candidate count, exact post-hoc candidate labels,
the chosen candidate index, likelihood scores in candidate order, score margins
under the frozen rank ordering, augmentation metadata, config SHA, and task IDs.

Unavailable: persisted augmentation support/count provenance for the legacy
artifact (as its own forensic report states), genuine execution/environment
failure labels, interactive-online traces, and calibrated probabilistic
selector-confidence labels.  The cached likelihood scores are ranking scores,
not correctness probabilities.  Therefore score-aware calibration is not used.

No regeneration is required for this Stage-0 dataset.  If either stated frozen
artifact or the local labels is removed, the extractor fails closed and will not
attempt to regenerate candidates.
