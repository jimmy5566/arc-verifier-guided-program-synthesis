#!/usr/bin/env bash
# One persistent model process per GPU; two same-adapter V5 cells per forward.
set -Eeuo pipefail
source "${ARC2_RUNTIME_ENV:?set ARC2_RUNTIME_ENV}"
REPO_ROOT=${ARC2_REPO_ROOT:?missing ARC2_REPO_ROOT}
RUN_ROOT=${ARC2_V5_RUN_ROOT:?missing ARC2_V5_RUN_ROOT}
PYTHON=${ARC2_PYTHON:?missing ARC2_PYTHON}
SCRIPT="$REPO_ROOT/scripts/run_eval60_v5_shared_model.py"
ARGS=(--output "$RUN_ROOT" --model-path "$ARC2_MODEL_PATH" --native-config-dir "$ARC2_NATIVE_CONFIG_DIR" --adapter-manifest "$ARC2_ADAPTER_MANIFEST" --lease-seconds 900)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SCRIPT" server "${ARGS[@]}" --gpu-id 0 --worker-id gpu_server_0 & p0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$SCRIPT" server "${ARGS[@]}" --gpu-id 1 --worker-id gpu_server_1 & p1=$!
wait "$p0" "$p1"
"$PYTHON" "$SCRIPT" reports --output "$RUN_ROOT"
