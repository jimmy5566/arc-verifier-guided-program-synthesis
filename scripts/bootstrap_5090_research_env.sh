#!/usr/bin/env bash
# Restore and verify the portable Blackwell research environment. This is
# infrastructure only: it stages frozen assets and runs a BF16 xFormers smoke.
set -Eeuo pipefail

GLOBAL_MOUNT="${ARC2_GLOBAL_MOUNT:-/workspace-global}"
PERSISTENT_ROOT="${ARC2_PERSISTENT_ROOT:-${GLOBAL_MOUNT}/arc2}"
RUNTIME_ROOT="${ARC2_RUNTIME_ROOT:-/root/arc-runtime}"
EXPECTED_GPU_COUNT="${ARC2_EXPECTED_GPU_COUNT:-2}"
MODEL_BUNDLE="${PERSISTENT_ROOT}/models/qwen3_4b_grids15_sft139"
ENV_BUNDLE="${PERSISTENT_ROOT}/env/5090-blackwell-env-v2"
ENV_MANIFEST="${PERSISTENT_ROOT}/env/environment_manifest.json"
MODEL_MANIFEST="${MODEL_BUNDLE}/model_manifest.json"
XFORMERS_IDENTITY="${PERSISTENT_ROOT}/wheels/xformers_source_identity.json"
LOCAL_ENV="${RUNTIME_ROOT}/env/5090-blackwell-env-v2"
LOCAL_MODEL="${RUNTIME_ROOT}/model-stage/qwen3_4b_grids15_sft139"
REPO_DIR="${ARC2_REPO_DIR:-${RUNTIME_ROOT}/arc2}"
REPO_URL="${ARC2_REPO_URL:-}"

if ! findmnt -T "${GLOBAL_MOUNT}" >/dev/null 2>&1; then
  echo "GLOBAL_VOLUME_UNAVAILABLE: ${GLOBAL_MOUNT} is not a mounted volume" >&2
  exit 2
fi
if [[ ! -d "${PERSISTENT_ROOT}" || ! -r "${PERSISTENT_ROOT}" ]]; then
  echo "PORTABLE_ASSET_ROOT_MISSING: ${PERSISTENT_ROOT}" >&2
  exit 2
fi
for required in "${ENV_BUNDLE}" "${MODEL_BUNDLE}" "${ENV_MANIFEST}" "${MODEL_MANIFEST}" "${XFORMERS_IDENTITY}"; do
  if [[ ! -e "${required}" ]]; then
    echo "PORTABLE_ASSET_MISSING: ${required}" >&2
    exit 3
  fi
done
if ! command -v rsync >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "BOOTSTRAP_PREREQUISITE_MISSING: rsync and nvidia-smi are required" >&2
  exit 4
fi

mkdir -p "${RUNTIME_ROOT}"/{hf_cache,torch_cache,triton_cache,uv_cache,model-stage,active_run,tmp,env}
export HF_HOME="${RUNTIME_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${RUNTIME_ROOT}/torch_cache"
export TRITON_CACHE_DIR="${RUNTIME_ROOT}/triton_cache"
export UV_CACHE_DIR="${RUNTIME_ROOT}/uv_cache"
export TOKENIZERS_PARALLELISM=false

stage_directory() {
  local source="$1" destination="$2" staging="${destination}.staging.$$"
  if [[ -e "${destination}" ]]; then
    rsync -a --checksum "${source}/" "${destination}/"
    return
  fi
  if [[ -e "${staging}" ]]; then
    echo "LOCAL_STAGE_ALREADY_EXISTS: ${staging}; inspect or remove this exact stale staging directory manually" >&2
    exit 5
  fi
  rsync -a --checksum "${source}/" "${staging}/"
  mv "${staging}" "${destination}"
}
stage_directory "${ENV_BUNDLE}" "${LOCAL_ENV}"
stage_directory "${MODEL_BUNDLE}" "${LOCAL_MODEL}"

if [[ ! -x "${LOCAL_ENV}/bin/python" || ! -f "${LOCAL_MODEL}/config.json" ]]; then
  echo "LOCAL_STAGE_INVALID" >&2
  exit 5
fi
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  if [[ -z "${REPO_URL}" ]]; then
    echo "ARC2_REPO_URL_REQUIRED: no local checkout exists" >&2
    exit 6
  fi
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
PINNED_COMMIT="$("${LOCAL_ENV}/bin/python" - "${ENV_MANIFEST}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['git_commit'])
PY
)"
git -C "${REPO_DIR}" fetch --quiet origin "${PINNED_COMMIT}"
git -C "${REPO_DIR}" checkout --detach --quiet "${PINNED_COMMIT}"

"${LOCAL_ENV}/bin/python" - "${ENV_MANIFEST}" "${MODEL_MANIFEST}" "${XFORMERS_IDENTITY}" "${LOCAL_MODEL}" "${EXPECTED_GPU_COUNT}" "${PERSISTENT_ROOT}" "${RUNTIME_ROOT}" <<'PY'
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import unsloth
import torch
import transformers
import peft
import xformers
import xformers.ops as xops

env_manifest = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
model_manifest = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
xformers_identity = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
model_path = Path(sys.argv[4])
expected_gpu_count = int(sys.argv[5])
persistent_root = sys.argv[6]
runtime_root = sys.argv[7]

if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpu_count:
    raise SystemExit(f'GPU_INVENTORY_MISMATCH: expected={expected_gpu_count}, actual={torch.cuda.device_count()}')
inventory = []
for index in range(expected_gpu_count):
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    if 'RTX 5090' not in name or capability != (12, 0):
        raise SystemExit(f'BLACKWELL_GPU_MISMATCH: index={index}, name={name}, capability={capability}')
    inventory.append({'physical_gpu_id': index, 'name': name, 'capability': list(capability)})
config_sha256 = hashlib.sha256((model_path / 'config.json').read_bytes()).hexdigest()
if model_manifest.get('model_config_sha256') != config_sha256:
    raise SystemExit('MODEL_CONFIG_HASH_MISMATCH')
actual_versions = {
    'python': '.'.join(map(str, sys.version_info[:3])),
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'transformers': transformers.__version__,
    'unsloth': unsloth.__version__,
    'peft': peft.__version__,
    'xformers': xformers.__version__,
    'torchao': importlib.metadata.version('torchao'),
    'triton': importlib.metadata.version('triton'),
}
if env_manifest.get('versions') != actual_versions:
    raise SystemExit(f'ENVIRONMENT_VERSION_MISMATCH: expected={env_manifest.get("versions")}, actual={actual_versions}')
record = importlib.metadata.distribution('xformers').read_text('RECORD') or ''
record_sha256 = hashlib.sha256(record.encode('utf-8')).hexdigest()
if record_sha256 != env_manifest.get('xformers_distribution_record_sha256'):
    raise SystemExit('XFORMERS_DISTRIBUTION_IDENTITY_MISMATCH')
if record_sha256 != xformers_identity.get('xformers_distribution_record_sha256'):
    raise SystemExit('XFORMERS_SOURCE_IDENTITY_MISMATCH')

smoke = []
for index in range(expected_gpu_count):
    with torch.cuda.device(index):
        q = torch.randn((1, 16, 4, 64), device='cuda', dtype=torch.bfloat16)
        output = xops.memory_efficient_attention(q, q, q)
        torch.cuda.synchronize(index)
        if not bool(torch.isfinite(output).all()):
            raise SystemExit(f'XFORMERS_BF16_SMOKE_NONFINITE: gpu={index}')
        smoke.append({'physical_gpu_id': index, 'shape': list(output.shape), 'dtype': str(output.dtype)})
payload = {
    'event': 'READY_FOR_ARC2_EXPERIMENTS',
    'persistent_root': persistent_root,
    'local_runtime_root': runtime_root,
    'git_commit': env_manifest['git_commit'],
    'model_config_sha256': config_sha256,
    'xformers_distribution_record_sha256': record_sha256,
    'gpu_inventory': inventory,
    'bf16_xformers_attention_smoke': smoke,
    'worker_mapping': [
        {'worker_id': index, 'physical_gpu_id': index, 'cuda_visible_devices': str(index), 'execution': 'independent_arc_task_worker'}
        for index in range(expected_gpu_count)
    ],
    'scientific_code_changed': False,
}
print(json.dumps(payload, sort_keys=True))
PY

echo "READY_FOR_ARC2_EXPERIMENTS"
