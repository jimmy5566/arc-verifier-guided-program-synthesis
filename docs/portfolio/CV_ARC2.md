# CV material — ARC-AGI Reasoning & Test-Time Inference

## A. One-line project entry

**ARC-AGI Reasoning & Test-Time Inference — Independent Research:** Developed and evaluated a verifier-guided neuro-symbolic candidate-generation and provenance-aware selection system using frozen, target-blind ARC cohorts and checkpointed multi-GPU inference.

## B. Academic CV — three bullets

- Designed a neuro-symbolic ARC research program spanning typed RuleSpec representations, deterministic execution/hard verification, public-model test-time candidate generation, and provenance-aware two-attempt selection.
- Established frozen-cohort, prediction-before-scoring protocols that separate candidate recall, Top-1 ranking, and two-attempt exact success; on one untouched 60-task A/B study, observed 21/60 versus 18/60 two-attempt exact solves (paired p=0.375; directional, not significant).
- Engineered resumable multi-GPU inference with atomic checkpoints and a dynamic scheduler; a CPU-only replay of 60 frozen task runtimes estimated a 26.38% makespan reduction without changing inference semantics.

## C. Detailed research CV — four bullets

- Built evidence extraction, parameter inference, RuleSpec, deterministic executor, and train-pair hard-verifier components to make candidate programs auditable; a train-only construction audit increased exact coverage from 3/30 to 10/30 with zero recorded regressions.
- Diagnosed the distinction between generation and selection: frozen30 forensics found correct candidates for 21/30 tasks but likelihood Top-1 for 9/30 (median correct rank 2), motivating dedicated ranking studies.
- Implemented reversible augmentation, bounded candidate search, exact-output deduplication, origin/provenance accounting, and predeclared distinct-attempt ranking around public pretrained Qwen/NVARC-derived assets (not author-trained checkpoints).
- Preserved negative findings—including 0/50 exact for both V2 Macro DSL pilot routes—and used failure taxonomy, frozen configs, and post-freeze scoring to constrain claims.

## Competition status placeholders

| Field | Status |
| --- | --- |
| Public LB | Pending |
| Final Rank | Pending |
| Medal | Pending |
