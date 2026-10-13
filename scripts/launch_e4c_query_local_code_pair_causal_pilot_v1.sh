#!/usr/bin/env bash
# Exact production path: preclaim contains no model/CUDA initialization; execute enters the one-shot boundary.
set -euo pipefail
if [ "$#" -ne 2 ] || [ "$1" != "--binding" ]; then echo "usage: $0 --binding PATH" >&2; exit 2; fi
binding="$2"
runtime="${ARC2_RUNTIME_PYTHON:-/root/code_pair_py311_cu128_v2/bin/python}"
[ -x "$runtime" ] || { echo "E4C_RUNTIME_PYTHON_MISSING" >&2; exit 3; }
"$runtime" scripts/run_e4c_query_local_code_pair_causal_pilot_v1.py --binding "$binding" --preclaim-only
exec "$runtime" scripts/run_e4c_query_local_code_pair_causal_pilot_v1.py --binding "$binding" --execute
