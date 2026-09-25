# RTX 3090 Eval60 runners

`scripts/run_eval60_d1_ampere.py` is a research-only runner for the frozen
TTT24/TTT48 fixed-4+4 per-output D1 evidence path. It accepts a full mounted
challenge plus the pre-existing, target-blind Eval60 cohort manifest. It never
accepts a solution file and refuses a challenge whose bound cohort hash differs.

Before either mode, use the persisted bootstrap on the Pod:

```bash
cd /workspace/arc2/bootstrap/3090-ampere-env-v1
ARC2_REQUIRED_GPUS=1 bash scripts/bootstrap_3090_ampere_env.sh
# Or, only on a Pod verified to expose two RTX 3090 GPUs:
ARC2_REQUIRED_GPUS=2 bash scripts/bootstrap_3090_ampere_env.sh
```

Materialize the runtime-only configuration from the pinned D1 release config.
This changes only the verified PTXAS executable and records the Ampere hardware
identity; it rejects source configurations that do not satisfy the frozen D1
release contract.

```bash
PY=/root/arc-runtime-3090/env/3090-ampere-env-v1/bin/python
$PY scripts/materialize_d1_ampere_runtime_config.py \
  --frozen-release-config /path/to/frozen_d1_release_config.json \
  --output /root/arc-runtime-3090/config/d1_ampere_runtime.json
```

## One RTX 3090: serialized TTT and generation concurrency two

`single-gen2` starts two independent full-model processes on physical GPU 0.
For each pair of tasks it serializes TTT and likelihood scoring, then sends
generation to both processes concurrently. A CUDA OOM in any concurrent phase
is deferred and retried after the second model has stopped; non-OOM failures
remain hard failures. The runner is intentionally not a shared-base or
multi-adapter implementation.

```bash
$PY scripts/run_eval60_d1_ampere.py \
  --mode single-gen2 \
  --challenge /path/to/arc-agi_evaluation_challenges.json \
  --eval60-manifest /workspace/arc2/benchmarks/eval60/eval60_manifest.json \
  --release-config /root/arc-runtime-3090/config/d1_ampere_runtime.json \
  --model-path /root/arc-runtime-3090/model-stage/qwen3_4b_grids15_sft139 \
  --native-config-dir /root/arc-runtime-3090/arc2/configs/nvarc_native_846d0198 \
  --checkpoint-dir /root/arc-runtime-3090/active_run/eval60-3090-gen2/checkpoints \
  --output /root/arc-runtime-3090/active_run/eval60-3090-gen2/candidates_frozen.json \
  --resume
```

## Two RTX 3090: dynamic complete-task queue

`dual-gpu` starts one persistent model per physical GPU. Each worker executes
both source trajectories for a task and immediately receives the next task when
it finishes. It neither shares model/adapters between devices nor runs two TTT
phases on the same GPU.

```bash
$PY scripts/run_eval60_d1_ampere.py \
  --mode dual-gpu \
  --challenge /path/to/arc-agi_evaluation_challenges.json \
  --eval60-manifest /workspace/arc2/benchmarks/eval60/eval60_manifest.json \
  --release-config /root/arc-runtime-3090/config/d1_ampere_runtime.json \
  --model-path /root/arc-runtime-3090/model-stage/qwen3_4b_grids15_sft139 \
  --native-config-dir /root/arc-runtime-3090/arc2/configs/nvarc_native_846d0198 \
  --checkpoint-dir /root/arc-runtime-3090/active_run/eval60-2x3090/checkpoints \
  --output /root/arc-runtime-3090/active_run/eval60-2x3090/candidates_frozen.json \
  --resume
```

Both commands intentionally omit selector scoring and target solutions. Run the
existing CPU D1 selector only after the candidate/evidence artifact is frozen.
