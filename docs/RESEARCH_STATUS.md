# Research status

## Completed

- Deterministic ARC solver infrastructure, typed capability registry, executor, and hard train-pair verifier.
- V1 low-level primitive-program experiment and its executable-path diagnosis.
- V2 Macro DSL, symbolic/direct parameter formulation, compiler, and deterministic verification architecture.
- Qwen3-8B local Transformers backend with offline operation.
- Kaggle four-NVIDIA-L4 task-parallel runner, one worker per GPU, atomic checkpoints, and resume/no-duplicate behavior.
- Fresh-kernel cold-load contention diagnosis and mitigation through sequential artifact warm-up and staggered worker initialization.
- Frozen 50-task symbolic V2 Pilot.
- Frozen 50-task direct-parameter V2 Pilot.
- Final V2 decision: **NO_GO**. Both conditions completed, and both produced zero exact solves.

## Not yet completed

- Detailed V2 failure taxonomy and stage-funnel analysis.
- Controlled ARC-SFT comparison under the same symbolic architecture.
- Verifier-guided iterative repair.
- Held-out final evaluation.
- Full benchmark evaluation.

## Interpretation boundary

The completed infrastructure demonstrates a reproducible experimental path from planner output to deterministic verification. It does not demonstrate an effective ARC solver. Roadmap items are hypotheses for future work, not results.
