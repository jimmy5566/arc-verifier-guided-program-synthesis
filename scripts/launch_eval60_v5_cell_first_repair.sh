#!/usr/bin/env bash
# Execution-only recovery controller: Fast 2x2, then Heavy 2x1, then isolation.
set -Eeuo pipefail

RUNTIME_ENV=${ARC2_RUNTIME_ENV:?set ARC2_RUNTIME_ENV}
# shellcheck disable=SC1090
source "$RUNTIME_ENV"
REPO_ROOT=${ARC2_REPO_ROOT:?missing ARC2_REPO_ROOT}
RUN_ROOT=${ARC2_V5_RUN_ROOT:?missing ARC2_V5_RUN_ROOT}
MODEL_PATH=${ARC2_MODEL_PATH:?missing ARC2_MODEL_PATH}
NATIVE_CONFIG_DIR=${ARC2_NATIVE_CONFIG_DIR:?missing ARC2_NATIVE_CONFIG_DIR}
ADAPTER_MANIFEST=${ARC2_ADAPTER_MANIFEST:?missing ARC2_ADAPTER_MANIFEST}
PYTHON=${ARC2_PYTHON:?missing ARC2_PYTHON}
SCRIPT="$REPO_ROOT/scripts/run_eval60_v5_cell_first_repair.py"
LOG_DIR="$RUN_ROOT/logs/cell_first"
mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1

run_worker() {
  local gpu=$1 worker=$2 pass=$3
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT" worker --output "$RUN_ROOT" --gpu-id "$gpu" --worker-id "$worker" --pass-mode "$pass" --lease-seconds 900 --model-path "$MODEL_PATH" --native-config-dir "$NATIVE_CONFIG_DIR" --adapter-manifest "$ADAPTER_MANIFEST"
}

pending() {
  local pass=$1 statuses
  case "$pass" in
    fast) statuses="'PENDING'";;
    heavy) statuses="'RETRY_HEAVY_OOM','RETRY_TRANSIENT'";;
    isolated) statuses="'HEAVY_OOM_SINGLE_WORKER'";;
  esac
  "$PYTHON" - "$RUN_ROOT/run_state.sqlite" "$statuses" <<'PY'
import sqlite3,sys
c=sqlite3.connect(sys.argv[1]); print(c.execute(f"select count(*) from cells where status in ({sys.argv[2]})").fetchone()[0])
PY
}

supervise() {
  local gpu=$1 worker=$2 pass=$3
  while [[ $(pending "$pass") -gt 0 ]]; do
    set +e
    run_worker "$gpu" "$worker" "$pass" >>"$LOG_DIR/${worker}_${pass}.log" 2>&1
    rc=$?
    set -e
    [[ $rc -eq 0 || $rc -eq 75 ]] || { echo "worker $worker pass $pass failed rc=$rc" >&2; exit "$rc"; }
  done
}

if [[ "${ARC2_SKIP_RECOVERY:-0}" != "1" ]]; then
  "$PYTHON" "$SCRIPT" recover --output "$RUN_ROOT" --adapter-manifest "$ADAPTER_MANIFEST" | tee "$LOG_DIR/recover.json"
else
  # Recovery has already been reconciled and hashed before this controller
  # starts.  Re-running it is unnecessary and can block on Global Volume
  # SQLite close semantics.
  [[ -f "$RUN_ROOT/pre_patch_recovery_summary.json" ]] || { echo "missing recovered-state summary" >&2; exit 2; }
fi
supervise 0 fast0 fast & p0=$!
supervise 0 fast1 fast & p1=$!
supervise 1 fast2 fast & p2=$!
supervise 1 fast3 fast & p3=$!
wait "$p0" "$p1" "$p2" "$p3"
"$PYTHON" "$SCRIPT" reports --output "$RUN_ROOT" | tee "$LOG_DIR/fast_summary.json"

supervise 0 heavy0 heavy & p0=$!
supervise 1 heavy1 heavy & p1=$!
wait "$p0" "$p1"
"$PYTHON" "$SCRIPT" reports --output "$RUN_ROOT" | tee "$LOG_DIR/heavy_summary.json"

# Strictly one active GPU worker for the final OOM-only isolation pass.
supervise 0 isolated0 isolated
"$PYTHON" "$SCRIPT" reports --output "$RUN_ROOT" | tee "$LOG_DIR/final_summary.json"
"$PYTHON" "$SCRIPT" freeze --output "$RUN_ROOT" | tee "$LOG_DIR/freeze.log"
