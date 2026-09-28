#!/usr/bin/env bash
# Gate and then migrate the current V5 run to exactly one model server per GPU.
set -Eeuo pipefail
source "${ARC2_RUNTIME_ENV:?set ARC2_RUNTIME_ENV}"
REPO_ROOT=${ARC2_REPO_ROOT:?missing ARC2_REPO_ROOT}
RUN_ROOT=${ARC2_V5_RUN_ROOT:?missing ARC2_V5_RUN_ROOT}
PYTHON=${ARC2_PYTHON:-/root/arc-runtime-turbodfs-v5-benchmark/env/turbodfs-v5/bin/python}
SCRIPT="$REPO_ROOT/scripts/run_eval60_v5_shared_model.py"
ARTIFACT_DIR="$RUN_ROOT/artifacts/turbodfs_shared_model_dual_cell_v1"
VALIDATION="$ARTIFACT_DIR/shared_validation.json"

[[ -f "$VALIDATION" ]] || { echo "missing shared validation: $VALIDATION" >&2; exit 2; }
"$PYTHON" - "$VALIDATION" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1]))
if not payload.get("deploy_pass"):
    raise SystemExit("shared-mode deployment gate failed: " + json.dumps(payload, sort_keys=True))
PY

# The caller must have stopped the legacy model workers.  Refuse to overlap a
# shared server with a second full model process on either 24 GiB GPU.
if pgrep -af 'run_eval60_v5_cell_first_repair.py worker' >/dev/null; then
  echo "legacy V5 model worker still live; refusing shared migration" >&2
  exit 3
fi

"$PYTHON" "$SCRIPT" migrate --output "$RUN_ROOT" | tee "$ARTIFACT_DIR/migration_apply.json"
bash "$REPO_ROOT/scripts/launch_eval60_v5_shared_model.sh" | tee "$ARTIFACT_DIR/shared_server_summary.json"
"$PYTHON" "$SCRIPT" reports --output "$RUN_ROOT" | tee "$ARTIFACT_DIR/final_shared_report.json"

# Freeze only a complete durable cell surface.  The existing freeze function
# rejects retryable states and is deliberately not bypassed here.
"$PYTHON" - "$RUN_ROOT/run_state.sqlite" <<'PY'
import sqlite3,sys
c=sqlite3.connect(sys.argv[1])
done=c.execute("select count(*) from cells where status='DONE'").fetchone()[0]
raise SystemExit(0 if done == 1068 else 4)
PY
"$PYTHON" "$REPO_ROOT/scripts/run_eval60_v5_cell_first_repair.py" freeze --output "$RUN_ROOT"
