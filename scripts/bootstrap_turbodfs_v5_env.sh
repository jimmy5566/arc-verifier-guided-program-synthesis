#!/usr/bin/env bash
# Rebuild a Pod-local V5 runtime from Git plus Global immutable assets.
# This intentionally never copies a venv, cache, Hugging Face hub tree or
# search-output run directory into Global Storage.
set -Eeuo pipefail

GLOBAL_ROOT=${ARC2_GLOBAL_ROOT:-/workspace/arc2}
RUNTIME_ROOT=${ARC2_RUNTIME_ROOT:-/root/arc-runtime-turbodfs-v5}
REPO_URL=${ARC2_REPO_URL:-https://github.com/jimmy5566/arc-verifier-guided-program-synthesis.git}
SOURCE_REF=${ARC2_SOURCE_REF:?set ARC2_SOURCE_REF to the exact Git commit}
REQUIRED_GPUS=${ARC2_REQUIRED_GPUS:-1}
VENV="$RUNTIME_ROOT/env/turbodfs-v5"
REPO="$RUNTIME_ROOT/repo"
MODEL_SOURCE="$GLOBAL_ROOT/models/qwen3_4b_grids15_sft139"
MODEL="$RUNTIME_ROOT/model-stage/qwen3_4b_grids15_sft139"
WHEEL_TAR="$GLOBAL_ROOT/wheelhouse/wheelhouse-py311-cu128-v1.tar"
WHEEL_REQ="$GLOBAL_ROOT/wheelhouse/requirements-py311-cu128-v1.txt"
XFORMERS_WHEEL="$GLOBAL_ROOT/wheels/xformers-0.0.33+5d4b92a.d20260925-cp39-abi3-linux_x86_64.whl"
XFORMERS_SHA=3f4609e50df81543e545f1e925a1bdce61dca9fe419aeb22b95d81d3dc1c123c
ADAPTER_MANIFEST="$GLOBAL_ROOT/turbodfs_v5/eval60_adapter_manifest.csv"
REFERENCE_BUNDLE="$GLOBAL_ROOT/assets/reference_bundle.tar.zst"

fail() { echo "TURBODFS_V5_BOOTSTRAP_FAIL=$*" >&2; exit 2; }
need_file() { [[ -s "$1" ]] || fail "missing_file:$1"; }
need_dir() { [[ -d "$1" ]] || fail "missing_dir:$1"; }
[[ "$REQUIRED_GPUS" == 1 || "$REQUIRED_GPUS" == 2 ]] || fail invalid_gpu_count
command -v python3.11 >/dev/null || fail python311_missing
need_file "$WHEEL_TAR"; need_file "$WHEEL_TAR.sha256"; need_file "$WHEEL_REQ"; need_file "$XFORMERS_WHEEL"; need_dir "$MODEL_SOURCE"; need_file "$MODEL_SOURCE/model_manifest.json"; need_file "$ADAPTER_MANIFEST"; need_file "$REFERENCE_BUNDLE"
[[ $(sha256sum "$WHEEL_TAR" | awk '{print $1}') == $(awk 'NR==1 {print $1}' "$WHEEL_TAR.sha256") ]] || fail wheelhouse_hash_mismatch
[[ $(sha256sum "$XFORMERS_WHEEL" | awk '{print $1}') == "$XFORMERS_SHA" ]] || fail xformers_hash_mismatch
inventory=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)
[[ $(printf '%s\n' "$inventory" | wc -l) -ge $REQUIRED_GPUS ]] || fail insufficient_gpu_inventory
printf '%s\n' "$inventory" | head -n "$REQUIRED_GPUS" | grep -Eq '^NVIDIA GeForce RTX 3090, 8\.6$' || fail sm86_rtx3090_required

mkdir -p "$RUNTIME_ROOT"/{env,model-stage,hf_cache,torch_cache,triton_cache,uv_cache,pip_cache,tmp}
export HF_HOME="$RUNTIME_ROOT/hf_cache" HUGGINGFACE_HUB_CACHE="$RUNTIME_ROOT/hf_cache/hub" TORCH_HOME="$RUNTIME_ROOT/torch_cache" TRITON_CACHE_DIR="$RUNTIME_ROOT/triton_cache" UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache" PIP_CACHE_DIR="$RUNTIME_ROOT/pip_cache" TOKENIZERS_PARALLELISM=false
if [[ ! -x "$VENV/bin/python" ]]; then
  wheels="$RUNTIME_ROOT/wheelhouse-py311-cu128-v1"; [[ ! -e "$wheels" ]] || fail stale_wheel_extraction
  mkdir "$wheels"; tar -C "$wheels" -xf "$WHEEL_TAR"; python3.11 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --no-index --find-links "$wheels" -r "$WHEEL_REQ"
  "$VENV/bin/python" -m pip install --no-index --force-reinstall --no-deps "$XFORMERS_WHEEL"
fi
if [[ ! -d "$MODEL" ]]; then stage="$MODEL.staging.$$"; mkdir -p "$stage"; tar -C "$MODEL_SOURCE" -cf - . | tar -C "$stage" -xf -; mv "$stage" "$MODEL"; fi
if [[ ! -d "$REPO/.git" ]]; then git clone "$REPO_URL" "$REPO"; fi
git -C "$REPO" fetch --quiet origin "$SOURCE_REF"; git -C "$REPO" checkout --detach --quiet "$SOURCE_REF"; [[ $(git -C "$REPO" rev-parse HEAD) == "$SOURCE_REF" ]] || fail source_commit_mismatch
NATIVE_CONFIG="$REPO/reference_assets/nvarc_native_846d0198"
if [[ ! -d "$NATIVE_CONFIG" ]]; then
  mkdir -p "$REPO/reference_assets"
  tar --zstd -C "$REPO/reference_assets" -xf "$REFERENCE_BUNDLE"
fi
need_dir "$NATIVE_CONFIG"
PTXAS=${TRITON_PTXAS_PATH:-$(command -v ptxas || true)}
[[ -x "$PTXAS" ]] || fail ptxas_missing
export TRITON_PTXAS_PATH="$PTXAS" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$REPO/src:$REPO"

"$VENV/bin/python" - "$MODEL" "$ADAPTER_MANIFEST" "$REQUIRED_GPUS" "$REPO" "$NATIVE_CONFIG" <<'PY'
import csv, hashlib, importlib.metadata, json, os, sys
from pathlib import Path
model, adapters, n, repo, native = (Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
manifest=json.loads((model/'model_manifest.json').read_text())
for name, expected in manifest['file_sha256'].items():
    p=model/name
    if not p.is_file(): raise SystemExit('MODEL_FILE_MISSING='+name)
    h=hashlib.sha256();
    with p.open('rb') as f:
      for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    if h.hexdigest()!=expected: raise SystemExit('MODEL_HASH_MISMATCH='+name)
rows=list(csv.DictReader(adapters.open()))
if len(rows)!=180: raise SystemExit('ADAPTER_MANIFEST_COUNT_MISMATCH')
for row in rows:
    p=Path(row['global_path'])
    if not p.is_file(): raise SystemExit('ADAPTER_MISSING='+str(p))
import torch, xformers.ops as xops
if not torch.cuda.is_available() or torch.cuda.device_count()<n or not torch.cuda.is_bf16_supported(): raise SystemExit('CUDA_BF16_PREFLIGHT_FAIL')
for i in range(n):
  with torch.cuda.device(i):
    if tuple(torch.cuda.get_device_capability(i))!=(8,6): raise SystemExit('SM86_REQUIRED')
    q=torch.randn((1,16,4,64),device=f'cuda:{i}',dtype=torch.bfloat16,requires_grad=True); out=xops.memory_efficient_attention(q,q,q); out.float().mean().backward(); torch.cuda.synchronize(i)
versions={k:importlib.metadata.version(k) for k in ('torch','transformers','peft','unsloth','unsloth-zoo','torchao','triton','safetensors','xformers')}
# This checks the actual offline loading contract, not merely file presence.
from unsloth import FastLanguageModel
from inference.nvarc_native import checkpoint_native_tokenizer
from scripts import run_eval60_adaptive_inference_joint_v2 as common
config=json.loads((repo/'configs'/'turbodfs_opt_v5_frontier_floor.json').read_text())
checkpoint_tokenizer_model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(model), full_finetuning=False, load_in_4bit=False,
    local_files_only=True, use_gradient_checkpointing=False,
    max_seq_length=8192,
)
tokenizer, _ = checkpoint_native_tokenizer(model, native)
if len(checkpoint_tokenizer) != 16 or len(tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != tokenizer.get_vocab():
    raise SystemExit('NATIVE_TOKENIZER_CONTRACT_FAIL')
loaded = FastLanguageModel.get_peft_model(
    checkpoint_tokenizer_model, r=256,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','embed_tokens','lm_head'],
    lora_alpha=32, lora_dropout=0.0, bias='none', use_gradient_checkpointing=False,
    random_state=42, use_rslora=True, loftq_config=None,
)
for depth in (12,24,48):
    row=next(row for row in rows if int(row['depth'])==depth)
    path=Path(row['global_path'])
    from safetensors import safe_open
    with safe_open(str(path), framework='pt', device='cpu') as handle: tensor_count=len(handle.keys())
    common.load_adapter(model=loaded, metadata={'checkpoint_path':str(path),'checkpoint_sha256':row['sha256'],'tensor_count':tensor_count})
decoder_sha=hashlib.sha256((repo/'configs'/'turbodfs_opt_v5_frontier_floor.json').read_bytes()).hexdigest()
print(json.dumps({'READY_FOR_TURBODFS_V5':True,'repo':str(repo),'model':str(model),'native_config':str(native),'adapters':len(rows),'decoder_config_sha256':decoder_sha,'versions':versions},sort_keys=True))
PY
echo "READY_FOR_TURBODFS_V5 repo=$REPO model=$MODEL venv=$VENV adapters=$ADAPTER_MANIFEST"
