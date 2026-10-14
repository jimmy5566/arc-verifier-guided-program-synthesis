#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "--binding" || -z "${2:-}" ]]; then echo "usage: $0 --binding RELATIVE_BINDING" >&2; exit 2; fi
ROOT="${ARC2_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
exec /root/code_pair_py311_cu128_v2/bin/python scripts/run_e4d_representation_binding_diagnostic_v1.py --binding "$2" --execute
