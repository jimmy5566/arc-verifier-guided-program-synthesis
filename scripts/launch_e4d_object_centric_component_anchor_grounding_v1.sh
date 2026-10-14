#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == "--binding" && -n "${2:-}" ]] || { echo 'usage: bash $0 --binding FILE' >&2; exit 2; }
exec "${PYTHON:-python3}" scripts/run_e4d_object_centric_component_anchor_grounding_v1.py --binding "$2" --execute