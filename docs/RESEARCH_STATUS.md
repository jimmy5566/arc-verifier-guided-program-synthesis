# Research status

## Current position

ARC2 is an independent research project on neuro-symbolic reasoning and reliable test-time inference for ARC-AGI. It is not presented as a completed general ARC solver, a peer-reviewed publication, or a competition result.

The current evidence package is organized in the [research audit](RESEARCH_AUDIT.md), [experiment index](EXPERIMENT_INDEX.md), [paper draft](paper/README.md), and [portfolio materials](portfolio/).

## Completed research assets

- Bounded symbolic libraries, typed capability infrastructure, deterministic execution, and hard train-pair verification.
- Macro DSL program-synthesis studies and retained negative results, including the frozen V2 0/50 symbolic and 0/50 direct pilot.
- Evidence/RuleSpec representation, parameter/binding pipeline, and train-only representability audits.
- Native public-model candidate generation with reversible augmentation, exact deduplication, and provenance-aware selection.
- Frozen and target-blind protocol support, separate post-freeze scoring, ranking forensics, and an untouched60 A/B comparison.
- Checkpoint/resume infrastructure, complete-per-GPU workers, and a dynamic task scheduler with CPU-only historical replay evidence.

## Interpretation boundary

The strongest held-out result is a single target-blind untouched60 A/B study: 21/60 versus 18/60 two-attempt exact solves, with paired exact p=0.375. It is a directional observation with limited power, not a statistically significant improvement claim. Candidate recall, Top-1, and two-attempt success are distinct and are not merged across cohorts.

The project uses external public Qwen/NVARC-derived assets in some studies; it does not claim author training of those checkpoints. Detailed local artifacts may be ignored by Git to avoid redistribution of restricted data or generated predictions.

## Open research questions

- Which target-blind candidate-quality signals can improve selection on a larger preregistered cohort?
- How can RuleSpec represent composition/state updates and repeat semantics without task-specific templates?
- How should inference compute be allocated adaptively while preserving frozen, target-blind evaluation?

## Competition status

| Field | Status |
| --- | --- |
| Public LB | Pending |
| Final rank | Pending |
| Medal | Pending |
