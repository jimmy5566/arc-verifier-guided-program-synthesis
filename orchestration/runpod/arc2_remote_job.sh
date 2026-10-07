#!/usr/bin/env bash
# Generic infrastructure wrapper. It never declares a scientific result.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 REMOTE_ROOT ROUND_ID COMMAND..." >&2
  exit 64
fi

remote_root=$1
round_id=$2
shift 2
round_dir="$remote_root/rounds/ROUND_${round_id}"
receipt="$round_dir/ROUND_${round_id}_TERMINAL_RECEIPT.json"
events="$remote_root/events.jsonl"
mkdir -p "$round_dir" "$(dirname "$events")"

if [[ -e "$receipt" ]]; then
  echo "existing terminal receipt: $receipt" >&2
  exit 65
fi

quoted_command=$(printf '%q ' "$@")
nohup setsid bash -c '
  set +e
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  eval "$1" >"$2/stdout.log" 2>"$2/stderr.log"
  code=$?
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [ "$code" -eq 0 ]; then state=SUCCESS; else state=TRAIN_FAILED; fi
  temp="$3.tmp.$$"
  printf "{\\\"round_id\\\":\\\"%s\\\",\\\"status\\\":\\\"%s\\\",\\\"started_at\\\":\\\"%s\\\",\\\"finished_at\\\":\\\"%s\\\",\\\"exit_code\\\":%s,\\\"scientific_acceptance\\\":\\\"NOT_EVALUATED\\\"}\\n" "$4" "$state" "$started" "$finished" "$code" >"$temp"
  mv "$temp" "$5"
  printf "{\\\"event\\\":\\\"ROUND_DONE\\\",\\\"round_id\\\":\\\"%s\\\",\\\"status\\\":\\\"%s\\\",\\\"receipt\\\":\\\"%s\\\"}\\n" "$4" "$state" "$5" >>"$6"
' _ "$quoted_command" "$round_dir" "$receipt" "$round_id" "$receipt" "$events" </dev/null >/dev/null 2>&1 &
pid=$!
printf '{"event":"ROUND_STARTED","round_id":"%s","pid":%s}\n' "$round_id" "$pid" >>"$events"
printf '%s\n' "$pid"
