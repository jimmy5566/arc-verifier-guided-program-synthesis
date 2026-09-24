#!/usr/bin/env bash
# Bootstrap/verification entry point for the pinned RTX 5090 research stack.
# It never rebuilds xFormers or downloads models implicitly.  A verified
# Network Volume must contain reusable assets before a fresh Pod is trusted.
set -Eeuo pipefail

PERSISTENT_ROOT="${ARC2_PERSISTENT_ROOT:-/network-volume/arc2}"
RUNTIME_ROOT="${ARC2_RUNTIME_ROOT:-/root/arc-runtime}"
ENV_PATH="${ARC2_5090_ENV_PATH:-${PERSISTENT_ROOT}/environments/5090-blackwell-env-v2}"
MODEL_PATH="${ARC2_MODEL_PATH:-${PERSISTENT_ROOT}/models/qwen3_4b_grids15_sft139}"
WHEEL_DIR="${ARC2_WHEEL_DIR:-${PERSISTENT_ROOT}/wheels}"

if ! findmnt -T "${PERSISTENT_ROOT}" >/dev/null 2>&1; then
  echo "PERSISTENT_STORAGE_UNAVAILABLE: ${PERSISTENT_ROOT} is not a mounted Network Volume" >&2
  exit 2
fi
if [[ ! -w "${PERSISTENT_ROOT}" ]]; then
  echo "PERSISTENT_STORAGE_UNWRITABLE: ${PERSISTENT_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUNTIME_ROOT}"/{hf_cache,torch_cache,triton_cache,uv_cache,temp,active_run}
export HF_HOME="${RUNTIME_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${RUNTIME_ROOT}/torch_cache"
export TRITON_CACHE_DIR="${RUNTIME_ROOT}/triton_cache"
export UV_CACHE_DIR="${RUNTIME_ROOT}/uv_cache"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "${ENV_PATH}/bin/python" ]]; then
  echo "PINNED_ENVIRONMENT_MISSING: ${ENV_PATH}; restore the audited environment from ${PERSISTENT_ROOT} first" >&2
  exit 3
fi
if [[ ! -d "${MODEL_PATH}" || ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "PINNED_MODEL_MISSING: ${MODEL_PATH}" >&2
  exit 4
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "GPU_PREFLIGHT_FAILED: nvidia-smi unavailable" >&2
  exit 5
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${ENV_PATH}/bin/python" - "${MODEL_PATH}" "${WHEEL_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch
import unsloth
import transformers
import peft
import xformers

model_path = Path(sys.argv[1])
wheel_dir = Path(sys.argv[2])
if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
    raise SystemExit(f"GPU_PREFLIGHT_FAILED: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable'}")
if not (model_path / "config.json").is_file():
    raise SystemExit("MODEL_MANIFEST_FAILED")
wheel_files = sorted(str(path.name) for path in wheel_dir.glob("*xformers*")) if wheel_dir.is_dir() else []
payload = {
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "unsloth": unsloth.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "xformers": xformers.__version__,
    "model_config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
    "reusable_xformers_wheels": wheel_files,
}
print(json.dumps(payload, sort_keys=True))
PY

echo "BOOTSTRAP_5090_RESEARCH_ENV_PASS"
