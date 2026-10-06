# M1a Runbook — Dots3 ARC Specialist

Branch: `model/dots3-arc-specialist-v1`

## External requirement

Dots3 must be exposed through an OpenAI-compatible endpoint. The official open-weight release can be served by SGLang/vLLM; the public Dots API may also be used when an account/API key is available.

**Do not paste API keys into Git or experiment receipts.**

Set on the execution machine:

```bash
export DOTS_API_KEY='...'
```

Then run:

```bash
python scripts/model_training/run_dots3_arc_specialist_m1a.py \
  --work-dir /root/arc-runtime-3090-v5/active_run/dots3_arc_specialist_m1a \
  --model-path /PATH/TO/qwen3_4b_grids15_sft139 \
  --teacher-base-url https://YOUR_DOTS_OPENAI_COMPATIBLE_ENDPOINT/v1
```

The controller is resumable and will not redo completed stages.

## Frozen stage order

1. Generate Dots3 program drafts from abstract curriculum families.
2. Execute/filter drafts. Require >=256 accepted programs.
3. Split whole families into train/val/test and build ARC 16-token conversations.
4. Evaluate frozen M0 on held-out test episodes.
5. Train one-epoch QLoRA M1a.
6. Evaluate M1a on the exact same episodes.
7. Apply the preregistered scale gate.

M1b / TTT / search are forbidden unless `M1A_SCALE_GATE.json: scale_gate_pass=true`.

## Important interpretation

This test measures whether Dots3-derived training moves the **correct output tokens upward in Qwen's own distribution**. It intentionally precedes TTT and search so that any gain can be attributed to the student model prior.
