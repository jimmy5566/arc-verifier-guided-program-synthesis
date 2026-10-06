# Dots3 -> ARC Specialist v1

Goal: improve the **model prior**, not replace the existing ARC solver.

Pipeline:

1. Dots3 produces novel ARC transformation descriptions + executable generators.
2. Generated code is sandbox-checked and executed over many seeds.
3. Only deterministic, valid, diverse, non-degenerate puzzles survive.
4. Whole concept families are split into train/validation/test before augmentation.
5. `qwen3_4b_grids15_sft139` receives QLoRA continued-SFT.
6. M0 and M1 are compared on the same held-out synthetic episodes using Gold-token likelihood/rank before any TTT/search comparison.
7. Only a passing M1 is allowed to retrain task-specific TTT adapters and enter the existing search pipeline.

The teacher is **offline only**. Dots3 is never required at Kaggle/test time.

## Why this design

The existing search experiments show that cumulative NLL remains the most robust local signal while learned rerankers have not reliably improved Top1. This project therefore asks whether a stronger training curriculum can move the Gold path upward *inside the Qwen distribution itself*.

## Clean-evaluation rule

ARC-AGI-2 public evaluation/Micro12 tasks are forbidden from teacher prompts and student SFT. They may be used later only as a mechanistic diagnostic, not as a clean generalization claim.

See `PREREGISTRATION.json` for frozen gates.
