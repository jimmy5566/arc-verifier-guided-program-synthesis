#!/usr/bin/env bash
# Restore the frozen 4090 Ada runtime to local NVMe.  This is infrastructure
# only: it does not launch an ARC task or access a target/solution file.
set -Eeuo pipefail

PERSISTENT_ROOT="${ARC2_PERSISTENT_ROOT:-/workspace/arc2}"
RUNTIME_ROOT="${ARC2_RUNTIME_ROOT:-/root/arc-runtime}"
REPO_URL="${ARC2_REPO_URL:-https://github.com/jimmy5566/arc-verifier-guided-program-synthesis.git}"
SOURCE_COMMIT="${ARC2_SOURCE_COMMIT:-9602b851dea2b4b4ae69b25585a9e591af1de523}"
REQUESTED_GPUS="${ARC2_REQUIRED_GPUS:-auto}"
ENVIRONMENT_ID="${ARC2_ENVIRONMENT_ID:-4090-ada-env-v1}"
ENV_ARCHIVE_LAYOUT="${ARC2_ENV_ARCHIVE_LAYOUT:-4090-ada-env-v1}"
EXPECTED_GPU_NAME="${ARC2_EXPECTED_GPU_NAME:-RTX 4090}"
EXPECTED_GPU_CAPABILITY="${ARC2_EXPECTED_GPU_CAPABILITY:-8,9}"

ENV_ARCHIVE="${PERSISTENT_ROOT}/env/4090-ada-env-v1.tar.zst"
ENV_SHA="${ENV_ARCHIVE}.sha256"
PYTHON_ARCHIVE="${PERSISTENT_ROOT}/env/uv-python-3.11.13.tar.zst"
PYTHON_SHA="${PYTHON_ARCHIVE}.sha256"
MODEL_ARCHIVE="${PERSISTENT_ROOT}/models/qwen3_4b_grids15_sft139-transformers-bfloat16-v1.tar.gz"
MODEL_SHA="${MODEL_ARCHIVE}.sha256"
MODEL_IMPORTANT_SHA="${PERSISTENT_ROOT}/models/qwen3_4b_grids15_sft139-transformers-bfloat16-v1.important_sha256.txt"
LOCAL_ENV="${RUNTIME_ROOT}/env/${ENVIRONMENT_ID}"
LOCAL_MODEL="${RUNTIME_ROOT}/model-stage/qwen3_4b_grids15_sft139-transformers-bfloat16-v1"
REPO_DIR="${ARC2_REPO_DIR:-${RUNTIME_ROOT}/arc2}"
UV_PYTHON_ROOT="/root/.local/share/uv/python"
UV_PYTHON_DIR="${UV_PYTHON_ROOT}/cpython-3.11.13-linux-x86_64-gnu"

fail() {
  echo "BOOTSTRAP_FAIL: $*" >&2
  exit 1
}

verify_archive() {
  local archive="$1" checksum="$2"
  [[ -f "${archive}" && -s "${archive}" ]] || fail "missing archive ${archive}"
  [[ -f "${checksum}" ]] || fail "missing checksum ${checksum}"
  local expected actual
  expected="$(awk 'NR==1 {print $1}' "${checksum}")"
  actual="$(sha256sum "${archive}" | awk '{print $1}')"
  [[ -n "${expected}" && "${actual}" == "${expected}" ]] || fail "hash mismatch ${archive}"
}

stage_tar_zst() {
  local archive="$1" extracted_name="$2" parent="$3" destination="$4"
  [[ -e "${destination}" ]] && return 0
  local staging="${parent}/.${extracted_name}.staging.$$"
  [[ ! -e "${staging}" ]] || fail "stale staging exists: ${staging}"
  mkdir -p "${staging}"
  tar --zstd -C "${staging}" -xf "${archive}"
  [[ -e "${staging}/${extracted_name}" ]] || fail "archive layout mismatch: ${archive}"
  mv "${staging}/${extracted_name}" "${destination}"
  rmdir "${staging}"
}

stage_tar_gz() {
  local archive="$1" destination="$2"
  [[ -e "${destination}" ]] && return 0
  local staging="${destination}.staging.$$"
  [[ ! -e "${staging}" ]] || fail "stale staging exists: ${staging}"
  mkdir -p "${staging}"
  tar -C "${staging}" -xzf "${archive}"
  [[ -f "${staging}/config.json" ]] || fail "model archive layout mismatch"
  mv "${staging}" "${destination}"
}

for item in "${ENV_ARCHIVE}" "${ENV_SHA}" "${PYTHON_ARCHIVE}" "${PYTHON_SHA}" "${MODEL_ARCHIVE}" "${MODEL_SHA}" "${MODEL_IMPORTANT_SHA}"; do
  [[ -e "${item}" ]] || fail "persistent asset missing: ${item}"
done
command -v nvidia-smi >/dev/null || fail "nvidia-smi unavailable"
command -v git >/dev/null || fail "git unavailable"
command -v tar >/dev/null || fail "tar unavailable"
command -v zstd >/dev/null || fail "zstd unavailable"

mkdir -p "${RUNTIME_ROOT}"/{hf_cache,torch_cache,triton_cache,uv_cache,model-stage,active_run,tmp,env}
export HF_HOME="${RUNTIME_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${RUNTIME_ROOT}/torch_cache"
export TRITON_CACHE_DIR="${RUNTIME_ROOT}/triton_cache"
export UV_CACHE_DIR="${RUNTIME_ROOT}/uv_cache"
export TOKENIZERS_PARALLELISM=false

verify_archive "${ENV_ARCHIVE}" "${ENV_SHA}"
verify_archive "${PYTHON_ARCHIVE}" "${PYTHON_SHA}"
verify_archive "${MODEL_ARCHIVE}" "${MODEL_SHA}"
mkdir -p "${UV_PYTHON_ROOT}"
stage_tar_zst "${PYTHON_ARCHIVE}" "cpython-3.11.13-linux-x86_64-gnu" "${UV_PYTHON_ROOT}" "${UV_PYTHON_DIR}"
stage_tar_zst "${ENV_ARCHIVE}" "${ENV_ARCHIVE_LAYOUT}" "${RUNTIME_ROOT}/env" "${LOCAL_ENV}"
stage_tar_gz "${MODEL_ARCHIVE}" "${LOCAL_MODEL}"
[[ -x "${LOCAL_ENV}/bin/python" ]] || fail "restored Python is unavailable"
[[ -f "${LOCAL_MODEL}/config.json" ]] || fail "restored model is unavailable"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" config core.filemode false
git -C "${REPO_DIR}" fetch --quiet origin "${SOURCE_COMMIT}"
git -C "${REPO_DIR}" checkout --detach --quiet "${SOURCE_COMMIT}"
[[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" == "${SOURCE_COMMIT}" ]] || fail "source commit mismatch"
[[ -z "$(git -C "${REPO_DIR}" status --porcelain)" ]] || fail "source checkout is dirty"

export ARC2_REQUESTED_GPUS="${REQUESTED_GPUS}"
"${LOCAL_ENV}/bin/python" - "${MODEL_IMPORTANT_SHA}" "${LOCAL_MODEL}" "${REQUESTED_GPUS}" "${PERSISTENT_ROOT}" "${RUNTIME_ROOT}" "${SOURCE_COMMIT}" "${ENVIRONMENT_ID}" "${EXPECTED_GPU_NAME}" "${EXPECTED_GPU_CAPABILITY}" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import torch
import xformers
import xformers.ops as xops
from xformers.ops.fmha.common import Inputs
from xformers.ops.fmha.dispatch import _dispatch_fw

important_hashes = Path(sys.argv[1])
model_path = Path(sys.argv[2])
requested = sys.argv[3]
persistent_root = sys.argv[4]
runtime_root = sys.argv[5]
commit = sys.argv[6]
environment_id = sys.argv[7]
expected_name = sys.argv[8]
expected_capability = tuple(int(part) for part in sys.argv[9].split(","))

if requested == "auto":
    selected_count = 2 if torch.cuda.device_count() >= 2 else 1
else:
    selected_count = int(requested)
if selected_count not in {1, 2}:
    raise SystemExit(f"GPU_REQUEST_INVALID: {selected_count}; use 1, 2, or auto")
if not torch.cuda.is_available() or torch.cuda.device_count() < selected_count:
    raise SystemExit(f"GPU_INVENTORY_MISMATCH: requested={selected_count}, actual={torch.cuda.device_count()}")

for line in important_hashes.read_text(encoding="utf-8").splitlines():
    expected, relative = line.split(maxsplit=1)
    actual_path = model_path / Path(relative.strip()).name
    if not actual_path.is_file():
        raise SystemExit(f"MODEL_FILE_MISSING: {actual_path.name}")
    actual = hashlib.sha256(actual_path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"MODEL_FILE_HASH_MISMATCH: {actual_path.name}")

inventory = []
for index in range(selected_count):
    name = torch.cuda.get_device_name(index)
    capability = tuple(torch.cuda.get_device_capability(index))
    if expected_name not in name or capability != expected_capability:
        raise SystemExit(
            f"GPU_CONTRACT_MISMATCH: expected_name={expected_name}, "
            f"expected_capability={expected_capability}, index={index}, "
            f"name={name}, capability={capability}"
        )
    inventory.append({"worker_id": index, "physical_gpu_id": index, "name": name, "capability": list(capability)})
if not torch.cuda.is_bf16_supported():
    raise SystemExit("BF16_UNAVAILABLE")

import unsloth
import transformers
import peft
versions = {
    "python": ".".join(map(str, sys.version_info[:3])),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "triton": importlib.metadata.version("triton"),
    "unsloth": unsloth.__version__,
    "unsloth_zoo": importlib.metadata.version("unsloth-zoo"),
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "torchao": importlib.metadata.version("torchao"),
    "xformers": xformers.__version__,
}
expected_versions = {
    "python": "3.11.13",
    "torch": "2.8.0+cu128",
    "cuda": "12.8",
    "triton": "3.4.0",
    "unsloth": "2025.9.7",
    "unsloth_zoo": "2025.9.9",
    "transformers": "4.55.4",
    "peft": "0.17.1",
    "torchao": "0.14.1",
    "xformers": "0.0.32.post2",
}
if versions != expected_versions:
    raise SystemExit(f"ENVIRONMENT_VERSION_MISMATCH: expected={expected_versions}, actual={versions}")

smoke = []
for index in range(selected_count):
    with torch.cuda.device(index):
        torch.cuda.reset_peak_memory_stats(index)
        q = torch.randn((1, 16, 4, 64), device=f"cuda:{index}", dtype=torch.bfloat16, requires_grad=True)
        selected_op = _dispatch_fw(Inputs(query=q, key=q, value=q), needs_gradient=True)
        backend = getattr(selected_op, "NAME", repr(selected_op))
        if not backend.startswith("fa"):
            raise SystemExit(f"FASTPATH_UNAVAILABLE: gpu={index}, backend={backend}")
        output = xops.memory_efficient_attention(q, q, q)
        output.float().square().mean().backward()
        torch.cuda.synchronize(index)
        if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(q.grad).all()):
            raise SystemExit(f"XFORMERS_BF16_SMOKE_NONFINITE: gpu={index}")
        smoke.append({
            "physical_gpu_id": index,
            "attention_backend": backend,
            "dtype": str(output.dtype),
            "peak_allocated_mb": int(torch.cuda.max_memory_allocated(index) / (1024 * 1024)),
            "peak_reserved_mb": int(torch.cuda.max_memory_reserved(index) / (1024 * 1024)),
        })

payload = {
    "event": "READY_FOR_ARC2_EXPERIMENTS",
    "environment_id": environment_id,
    "persistent_root": persistent_root,
    "local_runtime_root": runtime_root,
    "source_commit": commit,
    "gpu_inventory": inventory,
    "bf16_xformers_attention_smoke": smoke,
    "worker_semantics": "one independent ARC task worker per selected GPU; no tensor/model parallelism",
    "scientific_code_changed": False,
    "gpu_experiment_started": False,
    "versions": versions,
}
Path(runtime_root, "READY_FOR_ARC2_EXPERIMENTS.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
