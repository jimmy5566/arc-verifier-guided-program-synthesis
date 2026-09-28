#!/usr/bin/env bash
# One-shot autonomous V5 gate controller: calibration -> global assets -> clean
# bootstrap cell -> detached two-GPU Eval60 surface.  No Gold is touched here.
set -Eeuo pipefail

RUN_ROOT=${ARC2_TURBODFS_RUN_ROOT:?set ARC2_TURBODFS_RUN_ROOT}
AUTHORITATIVE_ROOT=${ARC2_AUTHORITATIVE_ROOT:?set ARC2_AUTHORITATIVE_ROOT}
CHALLENGE=${ARC2_CHALLENGE:?set ARC2_CHALLENGE}
REFERENCE_CONFIG=${ARC2_REFERENCE_CONFIG:?set ARC2_REFERENCE_CONFIG}
MODEL_SOURCE=${ARC2_MODEL_SOURCE:?set ARC2_MODEL_SOURCE}
NATIVE_CONFIG_SOURCE=${ARC2_NATIVE_CONFIG_SOURCE:?set ARC2_NATIVE_CONFIG_SOURCE}
SOLUTIONS=${ARC2_SOLUTIONS:?set ARC2_SOLUTIONS; only used by the later verified-freeze controller}
SOURCE_COMMIT=${ARC2_SOURCE_REF:?set ARC2_SOURCE_REF to the exact Git commit}
PYTHON=${ARC2_PYTHON:-python}
GLOBAL_ROOT=${ARC2_GLOBAL_ROOT:-/workspace/arc2}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PID_FILE="$RUN_ROOT/v5_full_calibration_cuda128.pid"
STATUS="$RUN_ROOT/post_calibration_controller.json"
CLEAN_ROOT=${ARC2_CLEAN_RUNTIME_ROOT:-/root/arc-runtime-turbodfs-v5-clean}

fail() { printf 'TURBODFS_V5_POST_CALIBRATION_FAIL=%s\n' "$*" >&2; exit 2; }
record() { "$PYTHON" - "$STATUS" "$1" <<'PY'
import json, sys, time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'status':sys.argv[2], 'solutions_accessed':False, 'updated_unix':time.time()},sort_keys=True)+'\n')
PY
}
record WAITING_FOR_CALIBRATION
if [[ -s "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
fi
"$PYTHON" - "$RUN_ROOT/turbodfs_v5_full_calibration.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
if x.get('status')!='PASS' or x.get('solutions_accessed') is not False:
    raise SystemExit('CALIBRATION_GATE_FAIL')
PY

record FREEZING_DECODER
"$PYTHON" "$REPO_ROOT/scripts/freeze_turbodfs_v5_config.py" --run-root "$RUN_ROOT" \
  --config "$RUN_ROOT/turbodfs_opt_v5_config.json" --source-commit "$SOURCE_COMMIT"
record PROMOTING_GLOBAL_ASSETS
"$PYTHON" "$REPO_ROOT/scripts/freeze_turbodfs_v5_global_assets.py" --global-root "$GLOBAL_ROOT" \
  --model-source "$MODEL_SOURCE" --authoritative-root "$AUTHORITATIVE_ROOT" --repo-root "$REPO_ROOT" \
  --source-commit "$SOURCE_COMMIT" --native-config-dir "$NATIVE_CONFIG_SOURCE" \
  --final-config "$RUN_ROOT/FINAL_TURBODFS_CONFIG.json"

[[ ! -e "$CLEAN_ROOT" ]] || fail clean_runtime_root_already_exists
record CLEAN_BOOTSTRAP
ARC2_RUNTIME_ROOT="$CLEAN_ROOT" ARC2_SOURCE_REF="$SOURCE_COMMIT" ARC2_REQUIRED_GPUS=1 \
  ARC2_GLOBAL_ROOT="$GLOBAL_ROOT" "$REPO_ROOT/scripts/bootstrap_turbodfs_v5_env.sh"
read -r task_id output_index depth view < <("$PYTHON" - "$RUN_ROOT/turbodfs_v5_micro_cohort.csv" <<'PY'
import csv,sys
with open(sys.argv[1], newline='') as f:
    row=next(csv.DictReader(f))
print(row['task_id'], row['output_index'], row['depth'], row['view'])
PY
)
ARC2_TURBODFS_RUN_ROOT="$RUN_ROOT" ARC2_AUTHORITATIVE_ROOT="$AUTHORITATIVE_ROOT" \
  ARC2_CHALLENGE="$CHALLENGE" ARC2_REFERENCE_CONFIG="$REFERENCE_CONFIG" \
  ARC2_MODEL_PATH="$CLEAN_ROOT/model-stage/qwen3_4b_grids15_sft139" \
  ARC2_NATIVE_CONFIG_DIR="$CLEAN_ROOT/repo/reference_assets/nvarc_native_846d0198" \
  ARC2_ADAPTER_MANIFEST="$GLOBAL_ROOT/turbodfs_v5/eval60_adapter_manifest.csv" \
  CUDA_VISIBLE_DEVICES=0 "$CLEAN_ROOT/env/turbodfs-v5/bin/python" "$CLEAN_ROOT/repo/scripts/clean_bootstrap_turbodfs_v5_cell.py" \
    --output "$RUN_ROOT/clean_bootstrap_turbodfs_cell.json" --authoritative-root "$AUTHORITATIVE_ROOT" \
    --challenge "$CHALLENGE" --reference-config "$REFERENCE_CONFIG" \
    --model-path "$CLEAN_ROOT/model-stage/qwen3_4b_grids15_sft139" \
    --native-config-dir "$CLEAN_ROOT/repo/reference_assets/nvarc_native_846d0198" \
    --adapter-manifest "$GLOBAL_ROOT/turbodfs_v5/eval60_adapter_manifest.csv" \
    --final-config "$RUN_ROOT/FINAL_TURBODFS_CONFIG.json" --task-id "$task_id" \
    --output-index "$output_index" --depth "$depth" --view "$view" --gpu-id 0

record FULL_EVAL60_LAUNCHED
setsid env \
  ARC2_TURBODFS_RUN_ROOT="$RUN_ROOT" ARC2_AUTHORITATIVE_ROOT="$AUTHORITATIVE_ROOT" \
  ARC2_CHALLENGE="$CHALLENGE" ARC2_REFERENCE_CONFIG="$REFERENCE_CONFIG" \
  ARC2_MODEL_PATH="$CLEAN_ROOT/model-stage/qwen3_4b_grids15_sft139" \
  ARC2_NATIVE_CONFIG_DIR="$CLEAN_ROOT/repo/reference_assets/nvarc_native_846d0198" \
  ARC2_ADAPTER_MANIFEST="$GLOBAL_ROOT/turbodfs_v5/eval60_adapter_manifest.csv" \
  ARC2_GLOBAL_ASSET_MANIFEST="$GLOBAL_ROOT/turbodfs_v5/GLOBAL_ASSET_MANIFEST.json" \
  ARC2_SOLUTIONS="$SOLUTIONS" ARC2_PYTHON="$CLEAN_ROOT/env/turbodfs-v5/bin/python" \
  bash "$CLEAN_ROOT/repo/scripts/launch_eval60_turbodfs_v5_2x3090.sh" \
  >"$RUN_ROOT/logs/turbodfs_v5_controller.log" 2>&1 < /dev/null &
echo $! > "$RUN_ROOT/turbodfs_v5_controller.pid"
echo "TURBODFS_V5_POST_CALIBRATION_LAUNCHED=$RUN_ROOT"
