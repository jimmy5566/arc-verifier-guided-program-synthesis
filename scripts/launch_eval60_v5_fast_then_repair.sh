#!/usr/bin/env bash
# Four independent fast workers, then two-worker and isolated V5 repair.
set -Eeuo pipefail
source "${ARC2_RUNTIME_ENV:?set ARC2_RUNTIME_ENV}"
REPO_ROOT=${ARC2_REPO_ROOT:?missing ARC2_REPO_ROOT}
RUN_ROOT=${ARC2_V5_RUN_ROOT:?missing ARC2_V5_RUN_ROOT}
STATE_DB=${ARC2_V5_STATE_DB:?missing ARC2_V5_STATE_DB}
PYTHON=${ARC2_PYTHON:-/root/arc-runtime-turbodfs-v5-benchmark/env/turbodfs-v5/bin/python}
SCRIPT="$REPO_ROOT/scripts/run_eval60_v5_fast_then_repair.py"
LOG_DIR="$RUN_ROOT/logs/fast_then_repair"; mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1
ARGS=(--output "$RUN_ROOT" --model-path "$ARC2_MODEL_PATH" --native-config-dir "$ARC2_NATIVE_CONFIG_DIR" --adapter-manifest "$ARC2_ADAPTER_MANIFEST" --lease-seconds 900)

count_phase() { "$PYTHON" - "$STATE_DB" "$1" <<'PY'
import sqlite3,sys
phase={'fast':('PENDING','RETRY_TRANSIENT'),'heavy':('RETRY_HEAVY_OOM','RETRY_TRANSIENT'),'isolated':('RETRY_ISOLATED_OOM')}[sys.argv[2]]
c=sqlite3.connect(sys.argv[1]); print(c.execute('select count(*) from cells where status in (%s)' % ','.join('?'*len(phase)),phase).fetchone()[0])
PY
}
# `worker` is foreground, so its Python PID has already exited before this is
# called.  Do not wait for *other* independent workers on the same GPU: that
# accidentally serialises the fast phase after every cell-local OOM.
wait_released() { sleep 1; }
worker() { local gpu=$1 id=$2 phase=$3; CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT" worker "${ARGS[@]}" --phase "$phase" --gpu-id "$gpu" --worker-id "$id"; }
supervise() { local gpu=$1 id=$2 phase=$3; while [[ $(count_phase "$phase") -gt 0 ]]; do set +e; worker "$gpu" "$id" "$phase" >>"$LOG_DIR/${id}_${phase}.log" 2>&1; rc=$?; set -e; [[ $rc -eq 0 || $rc -eq 75 || $rc -eq 76 ]] || exit "$rc"; [[ $rc -eq 0 ]] && break; wait_released "$gpu"; done; }

"$PYTHON" "$SCRIPT" recover --output "$RUN_ROOT" | tee "$LOG_DIR/recover.json"
supervise 0 fast0 fast & a=$!; supervise 0 fast1 fast & b=$!; supervise 1 fast2 fast & c=$!; supervise 1 fast3 fast & d=$!; wait "$a" "$b" "$c" "$d"
touch "$RUN_ROOT/FAST_PASS_COMPLETE.flag"; "$PYTHON" "$SCRIPT" summary --output "$RUN_ROOT" --phase fast | tee "$RUN_ROOT/artifacts/eval60_v5_fast_then_repair_v1/fast_pass_summary.json"
supervise 0 heavy0 heavy & a=$!; supervise 1 heavy1 heavy & b=$!; wait "$a" "$b"; "$PYTHON" "$SCRIPT" summary --output "$RUN_ROOT" --phase heavy | tee "$RUN_ROOT/artifacts/eval60_v5_fast_then_repair_v1/heavy_repair_summary.json"
supervise 0 isolated0 isolated; "$PYTHON" "$SCRIPT" summary --output "$RUN_ROOT" --phase isolated | tee "$RUN_ROOT/artifacts/eval60_v5_fast_then_repair_v1/isolated_repair_summary.json"
"$PYTHON" "$SCRIPT" freeze --output "$RUN_ROOT" | tee "$LOG_DIR/freeze.json"
