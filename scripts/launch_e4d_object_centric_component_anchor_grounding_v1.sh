#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "--binding" || -z "${2:-}" ]]; then
  echo 'usage: bash scripts/launch_e4d_object_centric_component_anchor_grounding_v1.sh --binding <path>' >&2; exit 2
fi
# The future authorized execution entrypoint starts with this identical preclaim.
exec "${PYTHON:-python3}" scripts/run_e4d_object_centric_component_anchor_grounding_v1.py --binding "$2" --preclaim-only