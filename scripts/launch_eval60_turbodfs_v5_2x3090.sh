#!/usr/bin/env bash
# Autonomous two-GPU V5 controller.  It never opens Gold until the two worker
# collectors have exited cleanly and the target-blind generation freeze exists.
set -Eeuo pipefail

RUN_ROOT=${ARC2_TURBODFS_RUN_ROOT:?set ARC2_TURBODFS_RUN_ROOT}
AUTHORITATIVE_ROOT=${ARC2_AUTHORITATIVE_ROOT:?set ARC2_AUTHORITATIVE_ROOT}
CHALLENGE=${ARC2_CHALLENGE:?set ARC2_CHALLENGE}
REFERENCE_CONFIG=${ARC2_REFERENCE_CONFIG:?set ARC2_REFERENCE_CONFIG}
MODEL_PATH=${ARC2_MODEL_PATH:?set ARC2_MODEL_PATH}
NATIVE_CONFIG_DIR=${ARC2_NATIVE_CONFIG_DIR:?set ARC2_NATIVE_CONFIG_DIR}
ADAPTER_MANIFEST=${ARC2_ADAPTER_MANIFEST:?set ARC2_ADAPTER_MANIFEST}
GLOBAL_ASSET_MANIFEST=${ARC2_GLOBAL_ASSET_MANIFEST:?set ARC2_GLOBAL_ASSET_MANIFEST}
SOLUTIONS=${ARC2_SOLUTIONS:?set ARC2_SOLUTIONS; this path is only consumed after freeze}
PYTHON=${ARC2_PYTHON:-python}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR="$RUN_ROOT/logs"; STATUS="$RUN_ROOT/controller_status.json"
mkdir -p "$LOG_DIR" "$RUN_ROOT/worker_status"

fail() { printf 'TURBODFS_V5_CONTROLLER_FAIL=%s\n' "$*" >&2; exit 2; }
[[ -f "$RUN_ROOT/FINAL_TURBODFS_CONFIG.json" ]] || fail final_decoder_not_frozen
[[ -f "$AUTHORITATIVE_ROOT/GREEDY_GENERATION_FROZEN.flag" ]] || fail authoritative_greedy_not_frozen
[[ -f "$ADAPTER_MANIFEST" ]] || fail global_adapter_manifest_missing
[[ -f "$GLOBAL_ASSET_MANIFEST" ]] || fail global_asset_manifest_missing
[[ -f "$CHALLENGE" && -f "$REFERENCE_CONFIG" ]] || fail runtime_input_missing
[[ -f "$SOLUTIONS" ]] || fail explicit_gold_file_missing
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -ge 2 ]] || fail two_gpu_required

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python - "$STATUS" <<'PY'
import json, sys, time
from pathlib import Path
p=Path(sys.argv[1]); p.write_text(json.dumps({'status':'STARTING','solutions_accessed':False,'started_unix':time.time()},sort_keys=True)+'\n')
PY

worker() {
  local physical_gpu=$1 worker_index=$2
  CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYTHON" "$REPO_ROOT/scripts/run_eval60_turbodfs_v5_blocks.py" \
    --output "$RUN_ROOT" --authoritative-root "$AUTHORITATIVE_ROOT" \
    --challenge "$CHALLENGE" --reference-config "$REFERENCE_CONFIG" \
    --model-path "$MODEL_PATH" --native-config-dir "$NATIVE_CONFIG_DIR" \
    --adapter-manifest "$ADAPTER_MANIFEST" --global-asset-manifest "$GLOBAL_ASSET_MANIFEST" --gpu-id "$physical_gpu" \
    --worker-index "$worker_index" --workers 2 --resume
}

worker 0 0 >"$LOG_DIR/worker_0.log" 2>&1 & pid0=$!
worker 1 1 >"$LOG_DIR/worker_1.log" 2>&1 & pid1=$!
set +e
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
set -e
if [[ $rc0 -ne 0 || $rc1 -ne 0 ]]; then
  python - "$STATUS" "$rc0" "$rc1" <<'PY'
import json, sys, time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'status':'WORKER_FAILURE','worker_0_rc':int(sys.argv[2]),'worker_1_rc':int(sys.argv[3]),'solutions_accessed':False,'updated_unix':time.time()},sort_keys=True)+'\n')
PY
  exit 3
fi

"$PYTHON" "$REPO_ROOT/scripts/finalize_eval60_turbodfs_v5.py" freeze --output "$RUN_ROOT" --authoritative-root "$AUTHORITATIVE_ROOT" >"$LOG_DIR/freeze.log" 2>&1
[[ -f "$RUN_ROOT/TURBODFS_GENERATION_FROZEN.flag" ]] || fail generation_freeze_missing
"$PYTHON" "$REPO_ROOT/scripts/finalize_eval60_turbodfs_v5.py" gold --output "$RUN_ROOT" --authoritative-root "$AUTHORITATIVE_ROOT" --solutions "$SOLUTIONS" >"$LOG_DIR/gold.log" 2>&1
python - "$STATUS" <<'PY'
import json, sys, time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'status':'COMPLETE','solutions_accessed_after_verified_freeze':True,'completed_unix':time.time()},sort_keys=True)+'\n')
PY
echo "TURBODFS_V5_CONTROLLER_COMPLETE=$RUN_ROOT"
