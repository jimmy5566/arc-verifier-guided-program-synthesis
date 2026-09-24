#!/usr/bin/env bash
# Publish an already-frozen run atomically. Per-task checkpoints remain local
# until the full run is immutable; partial staging never becomes a run.
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 LOCAL_RUN_DIR [PERSISTENT_ROOT=/workspace-global/arc2]" >&2
  exit 64
fi

source_run="$1"
persistent_root="${2:-${ARC2_PERSISTENT_ROOT:-/workspace-global/arc2}}"
global_mount="${ARC2_GLOBAL_MOUNT:-/workspace-global}"
runs_root="${persistent_root}/frozen-runs"
run_name="$(basename "${source_run}")"
destination="${runs_root}/${run_name}"
staging="${runs_root}/.${run_name}.staging.$$"

if [[ ! -d "${source_run}" ]]; then
  echo "LOCAL_RUN_MISSING: ${source_run}" >&2
  exit 2
fi
if ! findmnt -T "${global_mount}" >/dev/null 2>&1; then
  echo "GLOBAL_VOLUME_UNAVAILABLE: ${global_mount}" >&2
  exit 2
fi
if [[ ! -f "${source_run}/candidates_frozen.json" && ! -f "${source_run}/SERIAL_AA_STORAGE_GOVERNANCE_REPORT.json" ]]; then
  echo "RUN_NOT_FROZEN: expected immutable candidates or completed report" >&2
  exit 3
fi
if [[ -e "${destination}" ]]; then
  echo "DESTINATION_EXISTS: ${destination}; immutable runs are never overwritten" >&2
  exit 4
fi

mkdir -p "${runs_root}"
trap 'rm -rf "${staging}"' EXIT
rsync -a --checksum --delete \
  --exclude='*solution*' --exclude='*token*' --exclude='*.key' --exclude='credentials*' \
  "${source_run}/" "${staging}/"
find "${staging}" -type f -print0 | sort -z | xargs -0 sha256sum > "${staging}/SHA256SUMS"
if [[ -e "${destination}" ]]; then
  echo "DESTINATION_RACE: ${destination}" >&2
  exit 4
fi
mv "${staging}" "${destination}"
trap - EXIT
echo "PERSISTENT_RUN_SYNC_PASS: ${destination}"
