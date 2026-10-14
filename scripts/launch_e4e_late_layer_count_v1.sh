#!/usr/bin/env bash
set -euo pipefail
ROOT="${ARC2_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
exec "${PYTHON:-python3}" scripts/run_e4e_late_layer_count_v1.py --binding "$2" --cpu-production-path
