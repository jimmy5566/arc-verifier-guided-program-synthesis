# ARC-AGI Reasoning & Test-Time Inference — Independent Research

**Research question.** How can ARC-style reasoning systems generate and select candidate solutions reliably when the correct test output is inaccessible at inference time?

**Method.** I built and studied a neuro-symbolic research stack that separates candidate generation from deterministic execution, hard train-pair verification, candidate deduplication/provenance, and fixed target-blind selection. The work progressed from bounded symbolic program libraries through Macro DSL failure analysis to a typed RuleSpec architecture, then to specialized public-model candidate search, reversible augmentations, and two-attempt portfolio selection. Experiments use frozen cohorts and separate prediction freeze from scoring.

**Original contributions.** I designed the experiments and implemented the evidence/RuleSpec representation, parameter inference, deterministic executor and verifier, augmentation pipeline, provenance-aware candidate handling, selection diagnostics, frozen-cohort workflow, failure attribution, atomic resume/checkpointing, and dynamic multi-GPU scheduling. Public Qwen/NVARC-derived model assets were used as external pretrained components; I do not claim to have trained them.

**Most important evidence.** On a 60-task target-blind cohort frozen before inference, provenance-aware method B achieved 21/60 two-attempt exact solves versus 18/60 for baseline A, while both had 30/60 diagnostic candidate recall. The paired exact p-value was 0.375 (five discordant tasks), so this is a directional observation, not a significant improvement claim. On a frozen30 diagnostic, correct candidates existed for 21/30 tasks but likelihood Top-1 was correct for 9/30, revealing a measurable generation–selection gap. A train-only RuleSpec construction audit improved exact coverage from 3/30 to 10/30 without regressions; it does not claim end-to-end test performance.

**What I learned.** Representation capacity, neural recognition, candidate recall, and ranking are separable bottlenecks. More candidates can improve oracle recall without improving the first answer. Verification is valuable for auditability and train consistency, but it is not a test-correctness oracle. Reliable evaluation requires frozen task cohorts, fixed selection rules, and prediction freeze before labels are read.

**Current limitations.** Cohorts are small; the held-out A/B comparison has limited power; results depend on public pretrained ARC-specific weights; and the principal open problem is target-blind candidate selection. The 4×L4 scheduler result is a CPU-only replay of frozen runtimes, not a fresh inference benchmark.

**Future directions.** Preregister a larger selection study; develop train-only candidate-quality features; improve RuleSpec composition and repeat semantics; and study adaptive compute allocation without target leakage.

GitHub: <https://github.com/jimmy5566/arc-verifier-guided-program-synthesis>
