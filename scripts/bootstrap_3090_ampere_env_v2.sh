#!/usr/bin/env bash
# Rebuild ARC2's Ampere candidate runtime from immutable Global assets.
# The venv is always local; Global Storage holds only wheel/model source assets
# and small reproducibility manifests.
set -Eeuo pipefail

GLOBAL_ROOT=${ARC2_GLOBAL_ROOT:-/workspace/arc2}
RUNTIME_ROOT=${ARC2_RUNTIME_ROOT:-/root/arc-runtime-3090-v2}
REPO_URL=${ARC2_REPO_URL:-https://github.com/jimmy5566/arc-verifier-guided-program-synthesis.git}
SOURCE_REF=${ARC2_SOURCE_REF:?Set ARC2_SOURCE_REF to an exact Git SHA}
REQUIRED_GPUS=${ARC2_REQUIRED_GPUS:-1}
ENV_ID=3090-ampere-env-v2
PYTEST_VERSION=8.3.5
WHEEL_TAR=$GLOBAL_ROOT/wheelhouse/wheelhouse-py311-cu128-v1.tar
WHEEL_REQ=$GLOBAL_ROOT/wheelhouse/requirements-py311-cu128-v1.txt
XFORMERS_WHEEL=$GLOBAL_ROOT/wheels/xformers-0.0.33+5d4b92a.d20260925-cp39-abi3-linux_x86_64.whl
MODEL_SOURCE=$GLOBAL_ROOT/models/qwen3_4b_grids15_sft139
VENV=$RUNTIME_ROOT/env/$ENV_ID
MODEL=$RUNTIME_ROOT/model-stage/qwen3_4b_grids15_sft139
REPO=$RUNTIME_ROOT/arc2

fail() { echo "AMPERE_V2_BOOTSTRAP_FAIL=$*" >&2; exit 2; }
need_file() { [[ -f "$1" && -s "$1" ]] || fail "missing file $1"; }
need_dir() { [[ -d "$1" ]] || fail "missing directory $1"; }
verify_sha_sidecar() {
  local path=$1 expected actual
  need_file "$path"; need_file "$path.sha256"
  expected=$(awk 'NR==1 {print $1}' "$path.sha256")
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ -n "$expected" && "$expected" == "$actual" ]] || fail "sha256 mismatch $path"
}

[[ "$REQUIRED_GPUS" == 1 || "$REQUIRED_GPUS" == 2 ]] || fail 'ARC2_REQUIRED_GPUS must be 1 or 2'
command -v python3.11 >/dev/null || fail python311_missing
command -v nvidia-smi >/dev/null || fail nvidia_smi_missing
need_file "$WHEEL_REQ"; need_file "$XFORMERS_WHEEL"; need_dir "$MODEL_SOURCE"; need_file "$MODEL_SOURCE/model_manifest.json"
verify_sha_sidecar "$WHEEL_TAR"

inventory=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)
[[ $(printf '%s\n' "$inventory" | wc -l) -ge $REQUIRED_GPUS ]] || fail "insufficient GPUs: $inventory"
printf '%s\n' "$inventory" | head -n "$REQUIRED_GPUS" | grep -Eq '^NVIDIA GeForce RTX 3090, 8\.6$' || fail "expected RTX3090/sm86, got: $inventory"

mkdir -p "$RUNTIME_ROOT"/{env,model-stage,hf_cache,torch_cache,triton_cache,uv_cache,pip_cache,active_run,tmp}
export HF_HOME=$RUNTIME_ROOT/hf_cache HUGGINGFACE_HUB_CACHE=$RUNTIME_ROOT/hf_cache/hub
export TORCH_HOME=$RUNTIME_ROOT/torch_cache TRITON_CACHE_DIR=$RUNTIME_ROOT/triton_cache UV_CACHE_DIR=$RUNTIME_ROOT/uv_cache
export PIP_CACHE_DIR=$RUNTIME_ROOT/pip_cache
export TOKENIZERS_PARALLELISM=false

if [[ ! -d "$VENV" ]]; then
  wheels=$RUNTIME_ROOT/wheelhouse-py311-cu128-v1
  [[ ! -e "$wheels" ]] || fail "stale wheel extraction $wheels"
  mkdir "$wheels"; tar -C "$wheels" -xf "$WHEEL_TAR"
  python3.11 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --no-index --find-links "$wheels" -r "$WHEEL_REQ"
  "$VENV/bin/python" -m pip install --no-index --force-reinstall --no-deps "$XFORMERS_WHEEL"
fi
if [[ ! -d "$MODEL" ]]; then
  stage=$MODEL.staging.$$; mkdir -p "$stage"
  tar -C "$MODEL_SOURCE" -cf - . | tar -C "$stage" -xf -
  mv "$stage" "$MODEL"
fi
[[ -x "$VENV/bin/python" ]] || fail local_venv_missing

# Gate-0 and runtime-control validation are part of the initialized ARC2
# environment. pytest is pinned because it is not included in the immutable
# CUDA wheelhouse above; this install is deliberately outside the scientific
# model/runtime dependency set.
"$VENV/bin/python" -m pip install --disable-pip-version-check --no-input "pytest==$PYTEST_VERSION"
"$VENV/bin/python" - "$PYTEST_VERSION" <<'PY'
import importlib.metadata, sys
expected = sys.argv[1]
actual = importlib.metadata.version("pytest")
if actual != expected:
    raise SystemExit(f"PYTEST_VERSION_MISMATCH={actual}!={expected}")
print(f"PYTEST_READY={actual}")
PY

"$VENV/bin/python" - "$MODEL/model_manifest.json" "$MODEL" <<'PY'
import hashlib, json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
for name, expected in sorted(m['file_sha256'].items()):
    p = Path(sys.argv[2]) / name
    if not p.is_file(): raise SystemExit('MODEL_FILE_MISSING=' + name)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    if h.hexdigest() != expected: raise SystemExit('MODEL_HASH_MISMATCH=' + name)
PY

if [[ ! -d "$REPO/.git" ]]; then
  if [[ -n ${GH_TOKEN:-} ]]; then
    askpass=$RUNTIME_ROOT/.git_askpass; umask 077
    printf '#!/bin/sh\nprintf "%%s\\n" "${GH_TOKEN}"\n' > "$askpass"; chmod 700 "$askpass"
    GIT_ASKPASS=$askpass GIT_TERMINAL_PROMPT=0 git clone "$REPO_URL" "$REPO"; rm -f "$askpass"
  else
    git clone "$REPO_URL" "$REPO"
  fi
fi
git -C "$REPO" config core.filemode false
git -C "$REPO" fetch --quiet origin "$SOURCE_REF"
git -C "$REPO" checkout --detach --quiet "$SOURCE_REF"
[[ $(git -C "$REPO" rev-parse HEAD) == "$SOURCE_REF" ]] || fail source_ref_mismatch

"$VENV/bin/python" - "$REQUIRED_GPUS" "$RUNTIME_ROOT" "$SOURCE_REF" "$XFORMERS_WHEEL" <<'PY'
import hashlib, importlib.metadata, json, sys
from pathlib import Path
import torch, xformers.ops as xops
from xformers.ops.fmha.common import Inputs
from xformers.ops.fmha.dispatch import _dispatch_fw
n, root, ref, xwheel = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
if not torch.cuda.is_available() or torch.cuda.device_count() < n: raise SystemExit('CUDA_INVENTORY_MISMATCH')
if not torch.cuda.is_bf16_supported(): raise SystemExit('BF16_UNAVAILABLE')
gpus=[]
for i in range(n):
    with torch.cuda.device(i):
        if tuple(torch.cuda.get_device_capability(i)) != (8,6): raise SystemExit('SM86_REQUIRED')
        q=torch.randn((1,16,4,64), device=f'cuda:{i}', dtype=torch.bfloat16, requires_grad=True)
        op=_dispatch_fw(Inputs(query=q,key=q,value=q),needs_gradient=True); backend=getattr(op,'NAME',repr(op))
        if not backend.startswith('fa'): raise SystemExit('FASTPATH_UNAVAILABLE='+backend)
        out=xops.memory_efficient_attention(q,q,q); out.float().square().mean().backward(); torch.cuda.synchronize(i)
        if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(q.grad).all()): raise SystemExit('NONFINITE_FASTPATH_SMOKE')
        gpus.append({'index':i,'name':torch.cuda.get_device_name(i),'capability':[8,6],'attention_backend':backend})
payload={'event':'READY_FOR_ARC2_AMPERE_V2','source_commit':ref,'gpu_inventory':gpus,'wheel_sha256':hashlib.sha256(xwheel.read_bytes()).hexdigest(),'versions':{k:importlib.metadata.version(k) for k in ('torch','transformers','unsloth','unsloth-zoo','peft','torchao','xformers')}}
(root/'READY_FOR_ARC2_AMPERE_V2.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,sort_keys=True))
PY
