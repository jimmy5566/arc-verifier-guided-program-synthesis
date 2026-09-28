#!/usr/bin/env bash
# Controller for the authoritative 2-GPU x 2-worker V5 paired surface.
set -Eeuo pipefail

RUN_ROOT=${ARC2_V5_PAIRED_RUN_ROOT:?set ARC2_V5_PAIRED_RUN_ROOT}
AUTHORITATIVE_ROOT=${ARC2_AUTHORITATIVE_ROOT:?set ARC2_AUTHORITATIVE_ROOT}
CHALLENGE=${ARC2_CHALLENGE:?set ARC2_CHALLENGE}
REFERENCE_CONFIG=${ARC2_REFERENCE_CONFIG:?set ARC2_REFERENCE_CONFIG}
MODEL_PATH=${ARC2_MODEL_PATH:?set ARC2_MODEL_PATH}
NATIVE_CONFIG_DIR=${ARC2_NATIVE_CONFIG_DIR:?set ARC2_NATIVE_CONFIG_DIR}
ADAPTER_MANIFEST=${ARC2_ADAPTER_MANIFEST:?set ARC2_ADAPTER_MANIFEST}
GLOBAL_ASSET_MANIFEST=${ARC2_GLOBAL_ASSET_MANIFEST:?set ARC2_GLOBAL_ASSET_MANIFEST}
FINAL_CONFIG=${ARC2_FINAL_TURBODFS_CONFIG:?set ARC2_FINAL_TURBODFS_CONFIG}
FINAL_CONFIG_SHA=${ARC2_FINAL_TURBODFS_CONFIG_SHA:?set ARC2_FINAL_TURBODFS_CONFIG_SHA}
BENCHMARK_ROOT=${ARC2_CONCURRENCY_BENCHMARK:?set ARC2_CONCURRENCY_BENCHMARK}
PYTHON=${ARC2_PYTHON:?set ARC2_PYTHON}
REPO_ROOT=${ARC2_REPO_ROOT:?set ARC2_REPO_ROOT}
LOG_DIR="$RUN_ROOT/logs"

fail() { echo "EVAL60_V5_PAIRED_FULL_SWEEP_FAIL=$*" >&2; exit 2; }
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 2 ]] || fail exactly_two_rtx3090_required
[[ -f "$BENCHMARK_ROOT/benchmark_result.json" ]] || fail frozen_two_worker_per_gpu_benchmark_missing
[[ -f "$FINAL_CONFIG" && -f "$FINAL_CONFIG_SHA" && -f "$ADAPTER_MANIFEST" && -f "$GLOBAL_ASSET_MANIFEST" ]] || fail global_input_missing
[[ -d "$MODEL_PATH" && -f "$CHALLENGE" && -f "$REFERENCE_CONFIG" ]] || fail runtime_input_missing
[[ -x "$PYTHON" ]] || fail controller_python_missing
[[ -z $(git -C "$REPO_ROOT" status --porcelain) ]] || fail dirty_git_checkout

# The benchmark is an infrastructure gate.  It must show a clean two-copy pass;
# no A3 result is accepted or required.
"$PYTHON" - "$BENCHMARK_ROOT/benchmark_result.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
if p.get('status')!='FROZEN_TARGET_BLIND': raise SystemExit('benchmark is not frozen target-blind evidence')
rows={x.get('configuration_id'):x for x in p.get('summaries',[])}
two=rows.get('A2_workers2')
if not two or two.get('status')!='PASS' or int(two.get('oom_count',-1))!=0 or int(two.get('error_count',-1))!=0:
 raise SystemExit('two-workers-per-GPU safety gate FAIL')
if float(two.get('peak_reserved_gib',999.0)) >= 23.0:
 raise SystemExit('two-workers-per-GPU reserved memory is not below 23 GiB')
print('TWO_WORKERS_PER_GPU_SAFETY_GATE=PASS')
PY

SOURCE_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
"$PYTHON" "$REPO_ROOT/scripts/prepare_eval60_v5_paired_full_sweep.py" \
  --output "$RUN_ROOT" --challenge "$CHALLENGE" --reference-config "$REFERENCE_CONFIG" \
  --authoritative-root "$AUTHORITATIVE_ROOT" --adapter-manifest "$ADAPTER_MANIFEST" \
  --global-asset-manifest "$GLOBAL_ASSET_MANIFEST" --final-config "$FINAL_CONFIG" \
  --final-config-sha "$FINAL_CONFIG_SHA" --source-commit "$SOURCE_COMMIT"
mkdir -p "$LOG_DIR"
"$PYTHON" "$REPO_ROOT/scripts/run_eval60_v5_paired_full_sweep.py" init --output "$RUN_ROOT" >"$LOG_DIR/init.log" 2>&1

worker() {
  local gpu=$1 worker=$2
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$REPO_ROOT/scripts/run_eval60_v5_paired_full_sweep.py" worker \
    --output "$RUN_ROOT" --gpu-id "$gpu" --worker-id "$worker" --lease-seconds 7200 \
    --model-path "$MODEL_PATH" --native-config-dir "$NATIVE_CONFIG_DIR" --adapter-manifest "$ADAPTER_MANIFEST" \
    --global-asset-manifest "$GLOBAL_ASSET_MANIFEST"
}
worker 0 worker_0 >"$LOG_DIR/worker_0.log" 2>&1 & p0=$!
worker 0 worker_1 >"$LOG_DIR/worker_1.log" 2>&1 & p1=$!
worker 1 worker_2 >"$LOG_DIR/worker_2.log" 2>&1 & p2=$!
worker 1 worker_3 >"$LOG_DIR/worker_3.log" 2>&1 & p3=$!
pids=($p0 $p1 $p2 $p3)

# All four independent models must prove their model+adapter startup before the
# target-blind work gate opens.  No solution content is opened here.
for _ in $(seq 1 720); do
  ready=$("$PYTHON" - "$RUN_ROOT/run_state.sqlite" <<'PY'
import sqlite3,sys
c=sqlite3.connect(sys.argv[1]); print(c.execute("select count(*) from workers where status='MODEL_READY'").fetchone()[0])
PY
)
  [[ "$ready" -eq 4 ]] && break
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || fail worker_died_before_model_ready; done
  sleep 1
done
[[ "$ready" -eq 4 ]] || fail four_worker_model_ready_timeout
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >"$LOG_DIR/safety_gate_nvidia_smi.csv"
awk -F, '$2+0 >= 23552 {exit 1}' "$LOG_DIR/safety_gate_nvidia_smi.csv" || fail safety_gate_reserved_or_used_memory_too_high
printf 'generation opened after four MODEL_READY confirmations\n' >"$RUN_ROOT/START_GENERATION.flag"

# After the first twelve cells, compare target-blind search work against the
# frozen V5 serial calibration.  A >20%% median work reduction stops new leases.
checked=0
while true; do
  live=$("$PYTHON" - "$RUN_ROOT/run_state.sqlite" "$AUTHORITATIVE_ROOT/../eval60_turbodfs_v5/turbodfs_v5_full_calibration.json" "$RUN_ROOT" <<'PY'
import json,sqlite3,statistics,sys
db=sqlite3.connect(sys.argv[1]); rows=db.execute("select runtime_seconds,nodes_expanded,model_forwards,tokens_advanced,candidate_count from cells where status in ('TEMP_COMPLETE','COMPLETE') and runtime_seconds is not null order by updated_unix").fetchall(); db.close()
if len(rows)<12: print(0); raise SystemExit
c=json.load(open(sys.argv[2])); baseline_rows=[json.load(open(item['path'])) for item in c['cell_hashes']]
def median(rows, idx): return statistics.median(float(r[idx]) for r in rows)
def source_median(name): return statistics.median(float(r[name]) for r in baseline_rows)
first=rows[:12]
payload={'observed_cells':len(rows),'first12':{'runtime_seconds_median':median(first,0),'nodes_expanded_median':median(first,1),'model_forwards_median':median(first,2),'tokens_advanced_median':median(first,3),'candidate_count_median':median(first,4)},'serial_calibration':{'runtime_seconds_median':source_median('runtime_seconds'),'nodes_expanded_median':source_median('nodes_expanded'),'model_forwards_median':source_median('model_forwards'),'tokens_advanced_median':source_median('tokens_advanced'),'candidate_count_median':source_median('candidate_count')},'status':'PASS'}
for label,idx,field in [('nodes',1,'nodes_expanded'),('forwards',2,'model_forwards'),('tokens',3,'tokens_advanced'),('candidates',4,'candidate_count')]:
 ratio=median(first,idx)/source_median(field); payload[label+'_work_ratio']=ratio
 if ratio < .80: payload['status']='CONCURRENCY_DEGRADES_SEARCH_BUDGET'
json.dump(payload,open(sys.argv[3]+'/concurrency_search_work_gate.json','w'),sort_keys=True)
print(1)
PY
)
  if [[ "$live" -eq 1 ]]; then
    checked=1
    status=$("$PYTHON" -c "import json; print(json.load(open('$RUN_ROOT/concurrency_search_work_gate.json'))['status'])")
    if [[ "$status" != PASS ]]; then touch "$RUN_ROOT/STOP_AFTER_CURRENT_TASK.flag"; fi
    break
  fi
  alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
  [[ "$alive" -gt 0 ]] || break
  sleep 20
done
set +e
wait "$p0"; r0=$?; wait "$p1"; r1=$?; wait "$p2"; r2=$?; wait "$p3"; r3=$?
set -e
[[ $r0 -eq 0 && $r1 -eq 0 && $r2 -eq 0 && $r3 -eq 0 ]] || fail worker_failure_rcs_${r0}_${r1}_${r2}_${r3}
[[ -f "$RUN_ROOT/STOP_AFTER_CURRENT_TASK.flag" ]] && fail CONCURRENCY_DEGRADES_SEARCH_BUDGET
"$PYTHON" "$REPO_ROOT/scripts/finalize_eval60_v5_paired_full_sweep.py" freeze --output "$RUN_ROOT" >"$LOG_DIR/freeze.log" 2>&1
[[ -f "$RUN_ROOT/V5_GENERATION_FROZEN.flag" ]] || fail V5_generation_freeze_missing
SOLUTIONS=${ARC2_SOLUTIONS:?set ARC2_SOLUTIONS only after verified V5 generation freeze}
[[ -f "$SOLUTIONS" ]] || fail explicit_gold_solutions_file_missing_after_freeze
"$PYTHON" "$REPO_ROOT/scripts/finalize_eval60_v5_paired_full_sweep.py" gold --output "$RUN_ROOT" --authoritative-root "$AUTHORITATIVE_ROOT" --solutions "$SOLUTIONS" >"$LOG_DIR/gold.log" 2>&1
echo "EVAL60_V5_PAIRED_FULL_SWEEP_COMPLETE=$RUN_ROOT"
