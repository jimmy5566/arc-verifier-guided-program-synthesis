#!/usr/bin/env bash
# Atomically publish one already-frozen ARC2 run to an attached Network Volume.
# The source must contain only completed task checkpoints and immutable run artifacts.
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 LOCAL_RUN_DIR PERSISTENT_RUNS_DIR" >&2
  exit 64
fi

source_run="$1"
persistent_runs="$2"
run_name="$(basename "${source_run}")"
destination="${persistent_runs}/${run_name}"
staging="${persistent_runs}/.${run_name}.staging.$$"

if [[ ! -d "${source_run}" ]]; then
  echo "LOCAL_RUN_MISSING: ${source_run}" >&2
  exit 2
fi
if ! findmnt -T "${persistent_runs}" >/dev/null 2>&1; then
  echo "PERSISTENT_STORAGE_UNAVAILABLE: ${persistent_runs} is not on a mounted volume" >&2
  exit 2
fi
if [[ -e "${destination}" ]]; then
  echo "DESTINATION_EXISTS: ${destination}; preserve the immutable run instead of overwriting it" >&2
  exit 3
fi

mkdir -p "${persistent_runs}"
trap 'rm -rf "${staging}"' EXIT
rsync -a --checksum --delete "${source_run}/" "${staging}/"
find "${staging}" -type f -print0 | sort -z | xargs -0 sha256sum > "${staging}/SHA256SUMS"
mv "${staging}" "${destination}"
trap - EXIT
echo "PERSISTENT_RUN_SYNC_PASS: ${destination}"
