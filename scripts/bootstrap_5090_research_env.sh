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
ENVIRONMENT_MANIFEST="${ARC2_ENVIRONMENT_MANIFEST:-${PERSISTENT_ROOT}/manifests/5090-blackwell-env-v2.environment.json}"
MODEL_MANIFEST="${ARC2_MODEL_MANIFEST:-${PERSISTENT_ROOT}/manifests/qwen3_4b_grids15_sft139.model.json}"

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
if [[ ! -f "${ENVIRONMENT_MANIFEST}" || ! -f "${MODEL_MANIFEST}" ]]; then
  echo "PERSISTED_MANIFEST_MISSING: environment=${ENVIRONMENT_MANIFEST} model=${MODEL_MANIFEST}" >&2
  exit 4
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "GPU_PREFLIGHT_FAILED: nvidia-smi unavailable" >&2
  exit 5
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"${ENV_PATH}/bin/python" - "${MODEL_PATH}" "${WHEEL_DIR}" "${ENVIRONMENT_MANIFEST}" "${MODEL_MANIFEST}" <<'PY'
import hashlib
import importlib.metadata
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
environment_manifest_path = Path(sys.argv[3])
model_manifest_path = Path(sys.argv[4])
environment_manifest = json.loads(environment_manifest_path.read_text(encoding="utf-8"))
model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
    raise SystemExit(f"GPU_PREFLIGHT_FAILED: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable'}")
if not (model_path / "config.json").is_file():
    raise SystemExit("MODEL_MANIFEST_FAILED")
config_sha256 = hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest()
if environment_manifest.get("environment_id") != "5090-blackwell-env-v2":
    raise SystemExit("ENVIRONMENT_MANIFEST_ID_MISMATCH")
expected = environment_manifest.get("versions", {})
actual = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "unsloth": unsloth.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "xformers": xformers.__version__,
}
if any(expected.get(name) != value for name, value in actual.items()):
    raise SystemExit(f"ENVIRONMENT_VERSION_MISMATCH: expected={expected} actual={actual}")
if model_manifest.get("model_config_sha256") != config_sha256:
    raise SystemExit("MODEL_CONFIG_HASH_MISMATCH")
distribution = importlib.metadata.distribution("xformers")
record = distribution.read_text("RECORD") or ""
record_sha256 = hashlib.sha256(record.encode("utf-8")).hexdigest()
if environment_manifest.get("xformers_distribution_record_sha256") != record_sha256:
    raise SystemExit("XFORMERS_DISTRIBUTION_IDENTITY_MISMATCH")
wheel_files = sorted(str(path.name) for path in wheel_dir.glob("*xformers*")) if wheel_dir.is_dir() else []
payload = {
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "unsloth": unsloth.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "xformers": xformers.__version__,
    "model_config_sha256": config_sha256,
    "xformers_distribution_record_sha256": record_sha256,
    "attention_backend_expected": environment_manifest.get("attention_backend"),
    "attention_backend_runtime_attestation": "REQUIRED_FROM_REAL_MODEL_RUN",
    "reusable_xformers_wheels": wheel_files,
}
print(json.dumps(payload, sort_keys=True))
PY

echo "BOOTSTRAP_5090_RESEARCH_ENV_PASS"
