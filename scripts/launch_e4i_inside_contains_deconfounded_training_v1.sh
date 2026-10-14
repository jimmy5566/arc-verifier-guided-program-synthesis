#!/usr/bin/env bash
set -euo pipefail
ROOT="${ARC2_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
[[ $# -ge 2 && "$1" == "--binding" && -f "$2" ]] || {
  echo 'usage: --binding FILE [--cpu-self-test]' >&2
  exit 64
}
MODE="--execute"
if [[ $# -eq 3 ]]; then
  [[ "$3" == "--cpu-self-test" ]] || exit 64
  MODE="--cpu-self-test"
elif [[ $# -ne 2 ]]; then
  exit 64
fi
PYTHON_BIN="$(python3 - "$2" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['runtime_python'])
PY
)"
exec "$PYTHON_BIN" scripts/run_e4i_inside_contains_deconfounded_training_v1.py --binding "$2" "$MODE"
