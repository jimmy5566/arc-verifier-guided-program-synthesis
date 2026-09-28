#!/usr/bin/env bash
# Pod-local TurboDFS V5 runtime bootstrap. Source is always this exact Git checkout.
set -Eeuo pipefail

BOOTSTRAP_SCHEMA_VERSION=3
GLOBAL_ROOT=${ARC2_GLOBAL_ROOT:-/workspace/arc2}
RUNTIME_ROOT=${ARC2_RUNTIME_ROOT:-/root/arc-runtime-turbodfs-v5}
SOURCE_REF=${ARC2_SOURCE_REF:?set ARC2_SOURCE_REF to the exact Git commit}
REQUIRED_GPUS=${ARC2_REQUIRED_GPUS:-1}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)
VENV="$RUNTIME_ROOT/env/turbodfs-v5"
WHEEL_DIR="$RUNTIME_ROOT/wheelhouse-py311-cu128-v1"
MODEL_SOURCE="$GLOBAL_ROOT/models/qwen3_4b_grids15_sft139"
MODEL_STAGE="$RUNTIME_ROOT/model-stage/qwen3_4b_grids15_sft139"
REFERENCE_STAGE="$RUNTIME_ROOT/reference-assets"
WHEEL_TAR="$GLOBAL_ROOT/wheelhouse/wheelhouse-py311-cu128-v1.tar"
WHEEL_TAR_SHA="$GLOBAL_ROOT/wheelhouse/wheelhouse-py311-cu128-v1.tar.sha256"
WHEEL_REQ="$GLOBAL_ROOT/wheelhouse/requirements-py311-cu128-v1.txt"
XFORMERS_WHEEL="$GLOBAL_ROOT/wheels/xformers-0.0.33+5d4b92a.d20260925-cp39-abi3-linux_x86_64.whl"
XFORMERS_SHA=3f4609e50df81543e545f1e925a1bdce61dca9fe419aeb22b95d81d3dc1c123c
ADAPTER_MANIFEST="$GLOBAL_ROOT/turbodfs_v5/eval60_adapter_manifest.csv"
GLOBAL_MANIFEST="$GLOBAL_ROOT/turbodfs_v5/GLOBAL_ASSET_MANIFEST.json"
REFERENCE_BUNDLE="$GLOBAL_ROOT/assets/reference_bundle.tar.zst"
FINAL_CONFIG="$GLOBAL_ROOT/active_runs/eval60_turbodfs_v5/FINAL_TURBODFS_CONFIG.json"
REFERENCE_CONFIG_SOURCE="$GLOBAL_ROOT/active_runs/eval60_authoritative_greedy_v1/reference_ttt_config_cuda128_runtime.json"
ARTIFACT_DIR=${ARC2_BOOTSTRAP_ARTIFACT_DIR:-"$RUNTIME_ROOT/bootstrap_artifacts"}
GIT_CHECKOUT_SECONDS=${ARC2_GIT_CHECKOUT_SECONDS:-0}

fail() { echo "TURBODFS_V5_BOOTSTRAP_FAIL=$*" >&2; exit 2; }
need_file() { [[ -s "$1" ]] || fail "missing_file:$1"; }
need_dir() { [[ -d "$1" ]] || fail "missing_dir:$1"; }
now_ns() { date +%s%N; }
seconds() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.6f", (b-a)/1000000000 }'; }

[[ "$REQUIRED_GPUS" == 1 || "$REQUIRED_GPUS" == 2 ]] || fail invalid_gpu_count
git_started=$(now_ns)
[[ $(git -C "$REPO" rev-parse HEAD) == "$SOURCE_REF" ]] || fail source_commit_mismatch
[[ -z $(git -C "$REPO" status --porcelain) ]] || fail source_checkout_dirty
git_done=$(now_ns)
command -v python3.11 >/dev/null || fail python311_missing
need_dir "$GLOBAL_ROOT"; need_dir "$MODEL_SOURCE"; need_file "$MODEL_SOURCE/model_manifest.json"
need_file "$WHEEL_TAR"; need_file "$WHEEL_TAR_SHA"; need_file "$WHEEL_REQ"; need_file "$XFORMERS_WHEEL"
need_file "$ADAPTER_MANIFEST"; need_file "$GLOBAL_MANIFEST"; need_file "$REFERENCE_BUNDLE"; need_file "$FINAL_CONFIG"; need_file "$REFERENCE_CONFIG_SOURCE"
inventory=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)
[[ $(printf '%s
' "$inventory" | wc -l) -ge "$REQUIRED_GPUS" ]] || fail insufficient_gpu_inventory
printf '%s
' "$inventory" | head -n "$REQUIRED_GPUS" | grep -Eq '^NVIDIA GeForce RTX 3090, 8\.6$' || fail sm86_rtx3090_required
PTXAS=${TRITON_PTXAS_PATH:-$(command -v ptxas || true)}
[[ -x "$PTXAS" ]] || fail ptxas_missing
"$PTXAS" --version >/dev/null
[[ $(sha256sum "$WHEEL_TAR" | awk '{print $1}') == $(awk 'NR==1 {print $1}' "$WHEEL_TAR_SHA") ]] || fail wheelhouse_hash_mismatch
[[ $(sha256sum "$XFORMERS_WHEEL" | awk '{print $1}') == "$XFORMERS_SHA" ]] || fail xformers_hash_mismatch

mkdir -p "$RUNTIME_ROOT"/{env,model-stage,hf_cache,torch_cache,triton_cache,uv_cache,pip_cache,tmp} "$ARTIFACT_DIR"
export HF_HOME="$RUNTIME_ROOT/hf_cache" HUGGINGFACE_HUB_CACHE="$RUNTIME_ROOT/hf_cache/hub" TORCH_HOME="$RUNTIME_ROOT/torch_cache" TRITON_CACHE_DIR="$RUNTIME_ROOT/triton_cache" UV_CACHE_DIR="$RUNTIME_ROOT/uv_cache" PIP_CACHE_DIR="$RUNTIME_ROOT/pip_cache" TOKENIZERS_PARALLELISM=false

started=$(now_ns)
lock_sha=$(sha256sum "$WHEEL_REQ" | awk '{print $1}')
torch_requirement=$(awk -F'==' '$1=="torch" {print $2}' "$WHEEL_REQ")
python_mm=$(python3.11 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
cuda_contract=$(nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader | head -1)
fingerprint="$VENV/ARC2_ENV_FINGERPRINT.json"
expected=$(python3.11 - "$lock_sha" "$torch_requirement" "$python_mm" "$cuda_contract" "$BOOTSTRAP_SCHEMA_VERSION" <<'PY'
import json, sys
print(json.dumps({"environment_lock_sha256":sys.argv[1],"torch_requirement":sys.argv[2],"python_major_minor":sys.argv[3],"cuda_contract":sys.argv[4],"bootstrap_schema_version":int(sys.argv[5])}, sort_keys=True))
PY
)
venv_started=$(now_ns); venv_rebuilt=false; dependency_install_seconds=0
if [[ -x "$VENV/bin/python" && -f "$fingerprint" && $(cat "$fingerprint") == "$expected" ]]; then
  "$VENV/bin/python" - <<'PY'
import torch, transformers, peft, unsloth, xformers
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert tuple(torch.cuda.get_device_capability(0)) == (8, 6)
print("VENV_FAST_VALIDATION=PASS")
PY
else
  [[ ! -e "$VENV" ]] || rm -rf -- "$VENV"
  [[ ! -e "$WHEEL_DIR" ]] || rm -rf -- "$WHEEL_DIR"
  mkdir -p "$WHEEL_DIR"; tar -C "$WHEEL_DIR" -xf "$WHEEL_TAR"; python3.11 -m venv "$VENV"
  install_started=$(now_ns)
  "$VENV/bin/python" -m pip install --no-index --find-links "$WHEEL_DIR" -r "$WHEEL_REQ"
  "$VENV/bin/python" -m pip install --no-index --force-reinstall --no-deps "$XFORMERS_WHEEL"
  install_done=$(now_ns); dependency_install_seconds=$(seconds "$install_started" "$install_done")
  printf '%s' "$expected" > "$fingerprint"; venv_rebuilt=true
fi
venv_done=$(now_ns)

asset_started=$(now_ns)
"$VENV/bin/python" - "$GLOBAL_MANIFEST" "$ADAPTER_MANIFEST" "$MODEL_SOURCE" "$FINAL_CONFIG" "$PTXAS" <<'PY'
import csv, json, sys
from pathlib import Path
from safetensors import safe_open
import torch, xformers.ops as xops
manifest, adapters, model, final, ptxas = map(Path, sys.argv[1:])
assets = {str(a.get("absolute_path")): a for a in json.loads(manifest.read_text()).get("assets", [])}
rows = list(csv.DictReader(adapters.open()))
if len(rows) != 180 or not (model/"model_manifest.json").is_file() or not final.is_file() or not ptxas.is_file(): raise SystemExit("GLOBAL_CONTRACT_INVALID")
model_manifest = json.loads((model / "model_manifest.json").read_text())
for name in model_manifest.get("file_sha256", {}):
    path = model / name
    if not path.is_file() or str(path) not in assets:
        raise SystemExit(f"GLOBAL_MODEL_INVENTORY_MISMATCH:{name}")
for row in rows:
    p=Path(row["global_path"]); asset=assets.get(str(p))
    if not p.is_file() or p.stat().st_size != int(row["size"]) or asset is None or asset.get("sha256") != row["sha256"]: raise SystemExit("GLOBAL_ADAPTER_ATTESTATION_MISMATCH")
with safe_open(rows[0]["global_path"], framework="pt", device="cpu") as h:
    if not h.keys(): raise SystemExit("ADAPTER_HEADER_EMPTY")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or tuple(torch.cuda.get_device_capability(0)) != (8, 6): raise SystemExit("CUDA_BF16_PREFLIGHT_FAIL")
q=torch.randn((1,16,4,64),device="cuda:0",dtype=torch.bfloat16,requires_grad=True)
xops.memory_efficient_attention(q,q,q).float().mean().backward(); torch.cuda.synchronize()
print("GLOBAL_ASSET_FAST_VALIDATION=PASS")
PY
asset_done=$(now_ns)

if [[ ! -d "$MODEL_STAGE" ]]; then
  stage="$MODEL_STAGE.staging.$$"; mkdir -p "$stage"
  tar -C "$MODEL_SOURCE" -cf - . | tar -C "$stage" -xf -
  mv "$stage" "$MODEL_STAGE"
fi
[[ -f "$MODEL_STAGE/model_manifest.json" ]] || fail local_model_stage_missing_manifest
NATIVE_CONFIG="$REFERENCE_STAGE/nvarc_native_846d0198"
if [[ ! -d "$NATIVE_CONFIG" ]]; then mkdir -p "$REFERENCE_STAGE"; tar --zstd -C "$REFERENCE_STAGE" -xf "$REFERENCE_BUNDLE"; fi
need_dir "$NATIVE_CONFIG"

# PTXAS is an execution-environment path.  Preserve every scientific field in
# the frozen reference config while materializing only this path into the
# Pod-local runtime copy.
REFERENCE_CONFIG_RUNTIME="$RUNTIME_ROOT/reference_ttt_config_runtime.json"
"$VENV/bin/python" - "$REFERENCE_CONFIG_SOURCE" "$REFERENCE_CONFIG_RUNTIME" "$PTXAS" <<'PY'
import json, sys
from pathlib import Path
source, target, ptxas = map(Path, sys.argv[1:])
payload = json.loads(source.read_text(encoding="utf-8"))
payload["ptxas_path"] = str(ptxas)
temporary = target.with_suffix(target.suffix + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
PY

runtime_env="$RUNTIME_ROOT/runtime_paths.env"
cat > "$runtime_env" <<EOF
ARC2_GLOBAL_ROOT=$GLOBAL_ROOT
ARC2_REPO_ROOT=$REPO
ARC2_MODEL_SOURCE=$MODEL_SOURCE
ARC2_MODEL_PATH=$MODEL_STAGE
ARC2_ADAPTER_ROOT=$GLOBAL_ROOT/adapters/eval60_authoritative_greedy_v1
ARC2_ADAPTER_MANIFEST=$ADAPTER_MANIFEST
ARC2_GLOBAL_ASSET_MANIFEST=$GLOBAL_MANIFEST
ARC2_FINAL_TURBODFS_CONFIG=$FINAL_CONFIG
ARC2_REFERENCE_CONFIG=$REFERENCE_CONFIG_RUNTIME
ARC2_NATIVE_CONFIG_DIR=$NATIVE_CONFIG
TRITON_PTXAS_PATH=$PTXAS
PYTHONPATH=$REPO:$REPO/src
EOF
export TRITON_PTXAS_PATH="$PTXAS" PYTHONPATH="$REPO:$REPO/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
done_ns=$(now_ns)

"$VENV/bin/python" - "$ARTIFACT_DIR" "$REPO" "$VENV" "$runtime_env" "$fingerprint" "$MODEL_SOURCE" "$ADAPTER_MANIFEST" "$FINAL_CONFIG" "$REFERENCE_CONFIG_RUNTIME" "$PTXAS" "$started" "$git_started" "$git_done" "$venv_started" "$venv_done" "$asset_started" "$asset_done" "$done_ns" "$venv_rebuilt" "$dependency_install_seconds" "$GIT_CHECKOUT_SECONDS" <<'PY'
import json, sys
from pathlib import Path
out, repo, venv, paths, fp, model, adapters, final, reference_config, ptxas = map(Path, sys.argv[1:11])
started, gs, ge, vs, ve, ast, ae, done = map(int, sys.argv[11:19]); rebuilt=sys.argv[19]=="true"; install=float(sys.argv[20]); checkout=float(sys.argv[21])
sec=lambda a,b:(b-a)/1e9
out.mkdir(parents=True,exist_ok=True)
(out/"runtime_paths.json").write_text(json.dumps({"source_code_origin":"GIT_EXACT_COMMIT","arc2_repo_root":str(repo),"venv_path":str(venv),"venv_location":"POD_LOCAL","runtime_env":str(paths),"model_source":str(model),"adapter_manifest":str(adapters),"final_config":str(final),"reference_config_runtime":str(reference_config),"ptxas_path":str(ptxas),"python_executable":sys.executable,"pythonpath":str(repo)+":"+str(repo/"src"),"global_source_snapshot_used":False,"global_source_snapshot_deprecated":True},indent=2,sort_keys=True)+"\n")
(out/"environment_fingerprint.json").write_text(Path(fp).read_text()+"\n")
(out/"bootstrap_timing.json").write_text(json.dumps({"git_checkout_seconds":checkout,"git_identity_validation_seconds":sec(gs,ge),"venv_validation_seconds":sec(vs,ve),"venv_rebuild_seconds":sec(vs,ve) if rebuilt else 0.0,"dependency_install_seconds":install,"asset_validation_seconds":sec(ast,ae),"total_bootstrap_seconds":sec(started,done),"venv_rebuilt_this_run":rebuilt},indent=2,sort_keys=True)+"\n")
(out/"bootstrap_validation.json").write_text(json.dumps({"status":"BOOTSTRAP_COMPLETE","global_source_snapshot_used":False,"pythonpath_valid":True,"model_path_valid":True,"adapter_manifest_valid":True,"ptxas_valid":True,"model_ready_smoke":"PENDING_FIRST_WORKER","workers_share_controller_venv":"PENDING_FIRST_WORKER","worker_bootstrap_calls":"PENDING_FIRST_WORKER","authoritative_assets_modified":False,"global_small_file_explosion":False},indent=2,sort_keys=True)+"\n")
(out/"BOOTSTRAP_RUNTIME_CONTRACT.md").write_text("# Bootstrap runtime contract\n\n- Source: exact Git checkout only.\n- Global: immutable large assets only; source snapshot is deprecated.\n- Runtime/venv/caches/model stage: Pod-local and rebuildable.\n- Workers inherit the controller venv and never bootstrap.\n")
PY
echo "BOOTSTRAP_COMPLETE repo=$REPO venv=$VENV model=$MODEL_STAGE runtime_env=$runtime_env"
