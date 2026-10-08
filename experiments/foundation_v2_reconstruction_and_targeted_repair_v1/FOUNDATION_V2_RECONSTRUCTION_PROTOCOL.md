# FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1

The original Foundation-V2 adapter is unrecoverable. It remains a frozen historical reference, not a launchable checkpoint. This protocol creates a new adapter named `RECONSTRUCTED_FOUNDATION_V2_V1`; it may only be described as **RECONSTRUCTED_FROM_FROZEN_RECIPE**.

Before any GPU optimizer work, the RunPod-mounted base, tokenizer, Novel V1.1 data, replay shard, frozen schedule, source commit, software environment, and budget ledger must pass the recipe freeze. The training recipe is fixed: the verified BF16 Qwen3 base; frozen base weights; rank-64, alpha-32, zero-dropout LoRA on q/k/v/o and gate/up/down projections; PagedAdamW8bit; 5e-5 learning rate; steps 1–3 warmup then constant; context 8704; microbatch one; accumulation four; effective batch four; SDPA; gradient checkpointing; seed 2000031; and the frozen 3:1 Novel:Replay schedule.

Reconstruction has a 7,200-second reservation inside the single 28,800-second actual-new-GPU-training cap. The 2M nominal-token historical recipe targets 2,017,545 actual transformer tokens in 172 steps. Checkpoints are saved near 500k, 1M, 1.5M, and 2M tokens. All input mismatches, budget shortfalls, and post-cap optimizer steps fail closed.

The equivalence gate is preregistered in `RECONSTRUCTION_EQUIVALENCE_GATE.json`. It uses only fixed non-final-selection surfaces: Novel validation, replay retention, generation, protected retention, and a frozen capability-direction subset. It requires strong recovery of Novel performance, strong replay retention, generation improvement, no important protected regression, and broadly compatible capability direction. The exposed Novel holdout remains report-only after gate-based checkpoint selection.

`EQUIVALENCE_CONFIRMED` makes the selected and newly hashed reconstruction eligible for a separate `TARGETED_CAPABILITY_REPAIR_V1R` stage using the remaining budget. `EQUIVALENCE_PARTIAL` and `EQUIVALENCE_FAILED` preserve evidence and require Director escalation. Neither permits targeted repair.
