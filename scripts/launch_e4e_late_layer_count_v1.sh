#!/usr/bin/env bash
set -euo pipefail
ROOT="${ARC2_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
if [[ $# -ne 2 || "$1" != "--binding" || -z "$2" ]]; then
  echo "usage: $0 --binding <immutable-binding.json>" >&2; exit 64
fi
BINDING="$2"
[[ -f "$BINDING" ]] || { echo "missing binding: $BINDING" >&2; exit 66; }
RUNTIME_PYTHON="$(python3 - "$BINDING" <<'PY'
import json, sys
v=json.load(open(sys.argv[1], encoding="utf-8")).get("runtime_python")
if not isinstance(v,str) or not v: raise SystemExit("binding runtime_python missing")
print(v)
PY
)"
exec "$RUNTIME_PYTHON" scripts/run_e4e_late_layer_count_production_v1.py --binding "$BINDING" --execute
