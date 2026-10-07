# TARGETED_CAPABILITY_REPAIR_V1

Status: **PROTOCOL_FROZEN_PENDING_DIRECTOR**. GPU training has not started.

The protocol uses the frozen Foundation-V2 adapter identity, targets high-demand independent deficits first, uses a 50/30/20 atomic/composition/replay Round-1 mix, and keeps FINAL_AUDIT sealed until final model selection. It preserves BF16 frozen-base LoRA (r=64, alpha=32, dropout=0) and forbids Eval60 Gold, taxonomy changes, QLoRA, and diagnostic-example training.

Ordinary checkpoints use 192 TARGET_DEV and 96 retention episodes; full diagnostic remains final-stage only. A round requires paired target progress and acceptable retention; two valid no-gain rounds trigger curriculum diagnosis.
