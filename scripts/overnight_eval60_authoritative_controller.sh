#!/usr/bin/env bash
# Target-blind, fail-closed continuation controller for the authoritative Eval60 run.
# It never reads solutions until the Greedy freeze flag has been written and verified.
set -euo pipefail

RUN="${RUN:-/workspace/arc2/active_runs/eval60_authoritative_greedy_v1}"
SRC="${SRC:-/root/arc-runtime-adaptive-2x3090/arc2-turbodfs-v2-target-blind}"
PY="${PY:-/root/arc-runtime-adaptive-2x3090/env/3090-ampere-env-v2/bin/python}"
CHALLENGE="${CHALLENGE:-/workspace/arc2/benchmarks/arc-agi_evaluation_challenges.json}"
NATIVE_CONFIG="${NATIVE_CONFIG:-/workspace/arc2/arc2/source/configs/nvarc_native_846d0198}"
MODEL="${MODEL:-/root/arc-runtime-adaptive-2x3090/model-stage/qwen3_4b_grids15_sft139}"
CONFIG="$RUN/reference_ttt_config_cuda128_runtime.json"
T0_FILE="$RUN/authoritative_t0_utc.txt"
LOG="$RUN/logs/overnight_controller.log"

mkdir -p "$RUN/logs"
exec >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] controller started"

status_counts() {
  "$PY" - "$RUN" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]) / "task_status"
counts = {"GREEDY_COMPLETE": 0, "FAILED": 0, "RUNNING": 0, "other": 0}
for path in root.glob("*.json"):
    try: status = json.loads(path.read_text())["status"]
    except Exception: status = "other"
    counts[status if status in counts else "other"] += 1
print(counts["GREEDY_COMPLETE"], counts["FAILED"], counts["RUNNING"], counts["other"])
PY
}

while true; do
  read -r complete failed running other < <(status_counts)
  cells=$(find "$RUN/raw/greedy_cells" -type f -name '*.json' 2>/dev/null | wc -l)
  adapters=$(find "$RUN/checkpoints" -type f -name adapter_model.safetensors 2>/dev/null | wc -l)
  echo "[$(date -u +%FT%TZ)] greedy complete=$complete failed=$failed running=$running other=$other cells=$cells adapters=$adapters"
  if [[ "$failed" != 0 ]]; then
    echo "FATAL: worker reported failed task; refusing freeze"
    exit 20
  fi
  if [[ "$complete" == 60 && "$cells" == 1068 && "$adapters" == 180 ]]; then
    break
  fi
  sleep 120
done

echo "[$(date -u +%FT%TZ)] all Greedy evidence present; freezing target-blind evidence"
"$PY" "$SRC/scripts/run_eval60_authoritative_greedy_v1.py" freeze --output "$RUN"
test -f "$RUN/GREEDY_GENERATION_FROZEN.flag"

# This is deliberately the first point in the controller at which solutions are located/read.
SOLUTIONS=$(find /workspace/arc2/benchmarks -maxdepth 2 -type f -iname '*evaluation*solutions*.json' -print -quit)
if [[ -z "$SOLUTIONS" ]]; then
  echo "FATAL: frozen Greedy exists but evaluation solutions file was not found"
  exit 21
fi
echo "[$(date -u +%FT%TZ)] attaching retrospective Gold after verified freeze: $SOLUTIONS"
"$PY" "$SRC/scripts/finalize_eval60_authoritative_greedy_v1.py" gold --output "$RUN" --solutions "$SOLUTIONS"

deadline_epoch=$(( $(date -u -d "$(cat "$T0_FILE")" +%s) + 13*3600 + 15*60 ))
now_epoch=$(date -u +%s)
if (( now_epoch >= deadline_epoch )); then
  echo "[$(date -u +%FT%TZ)] TURBODFS V3 NOT STARTED: 13h15 deadline reached"
  "$PY" "$SRC/scripts/make_eval60_authoritative_handoff.py" --output "$RUN"
  exit 0
fi

V3="$RUN/turbodfs_opt_v3_config.json"
MICRO="$RUN/turbodfs_v3_micro_cohort.csv"
FULL="$RUN/turbodfs_v3_full_cohort.csv"
echo "[$(date -u +%FT%TZ)] starting target-blind V3 micro calibration"
if ! "$PY" "$SRC/scripts/calibrate_turbodfs_opt_v3.py" \
  --output "$RUN" --challenge "$CHALLENGE" --reference-config "$CONFIG" \
  --model-path "$MODEL" --native-config-dir "$NATIVE_CONFIG" --gpu-id 0 \
  --v3-config "$V3" --cohort "$MICRO" --mode micro; then
  echo "[$(date -u +%FT%TZ)] V3 micro failed; preserving authoritative Greedy package and stopping V3 only"
  "$PY" "$SRC/scripts/make_eval60_authoritative_handoff.py" --output "$RUN"
  exit 0
fi

if [[ $("$PY" -c "import json; print(json.load(open('$RUN/turbodfs_v3_micro_calibration.json'))['status'])") != PASS ]]; then
  echo "FATAL: V3 micro did not pass"
  exit 22
fi

now_epoch=$(date -u +%s)
micro_p90=$("$PY" -c "import json; print(json.load(open('$RUN/turbodfs_v3_micro_calibration.json'))['p90_seconds_per_cell'])")
enough=$(( deadline_epoch - now_epoch ))
needed=$("$PY" -c "print(int(float('$micro_p90')*24*1.15))")
if (( enough < needed )); then
  echo "[$(date -u +%FT%TZ)] V3 full NOT STARTED: remaining=${enough}s estimated=${needed}s"
  "$PY" "$SRC/scripts/make_eval60_authoritative_handoff.py" --output "$RUN"
  exit 0
fi
echo "[$(date -u +%FT%TZ)] starting target-blind V3 full calibration"
if ! "$PY" "$SRC/scripts/calibrate_turbodfs_opt_v3.py" \
  --output "$RUN" --challenge "$CHALLENGE" --reference-config "$CONFIG" \
  --model-path "$MODEL" --native-config-dir "$NATIVE_CONFIG" --gpu-id 0 \
  --v3-config "$V3" --cohort "$FULL" --mode full; then
  echo "[$(date -u +%FT%TZ)] V3 full failed; preserving authoritative Greedy package and stopping V3 only"
  "$PY" "$SRC/scripts/make_eval60_authoritative_handoff.py" --output "$RUN"
  exit 0
fi
if [[ $("$PY" -c "import json; print(json.load(open('$RUN/turbodfs_v3_full_calibration.json'))['status'])") != PASS ]]; then
  echo "FATAL: V3 full did not pass"
  exit 23
fi

# V3 workers enforce the frozen SHA256 block order, a complete 12-cell block
# boundary, and the same T0+13:15 deadline independently before each block.
echo "[$(date -u +%FT%TZ)] V3 full passed; starting bounded two-GPU block expansion"
for gpu in 0 1; do
  nohup "$PY" "$SRC/scripts/run_eval60_turbodfs_opt_v3_blocks.py" \
    --output "$RUN" --challenge "$CHALLENGE" --reference-config "$CONFIG" \
    --model-path "$MODEL" --native-config-dir "$NATIVE_CONFIG" --gpu-id "$gpu" \
    --worker-index "$gpu" --workers 2 --v3-config "$V3" \
    >"$RUN/logs/turbodfs_v3_worker_${gpu}.log" 2>&1 &
  echo $! > "$RUN/turbodfs_v3_worker_${gpu}.pid"
done
wait "$(cat "$RUN/turbodfs_v3_worker_0.pid")"
wait "$(cat "$RUN/turbodfs_v3_worker_1.pid")"
echo "[$(date -u +%FT%TZ)] V3 bounded expansion workers exited; freezing retained completed/partial blocks."
"$PY" "$SRC/scripts/finalize_eval60_turbodfs_opt_v3.py" freeze --output "$RUN"
"$PY" "$SRC/scripts/finalize_eval60_turbodfs_opt_v3.py" gold --output "$RUN" --solutions "$SOLUTIONS"
"$PY" "$SRC/scripts/make_eval60_authoritative_handoff.py" --output "$RUN"
echo "[$(date -u +%FT%TZ)] V3 expansion frozen and labelled post-freeze."
