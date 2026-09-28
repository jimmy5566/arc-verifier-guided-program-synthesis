#!/usr/bin/env bash
# Build one immutable Eval60 archive group as 1.5 GiB tar.zst shards.
# Large bytes remain outside normal Git. This script reads sources only.
set -euo pipefail

GROUP=""
STAGE="/workspace/arc2/release-staging"
MANIFEST=""
MODEL_STATUS="UNCLEAR"

usage() {
  echo "usage: $0 --group {TTT_ADAPTERS|GREEDY_RAW|V5_RAW|BASE_MODEL} --manifest FILE [--stage DIR] [--model-status ALLOWED]" >&2
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP=$2; shift 2 ;;
    --stage) STAGE=$2; shift 2 ;;
    --manifest) MANIFEST=$2; shift 2 ;;
    --model-status) MODEL_STATUS=$2; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "$GROUP" && -n "$MANIFEST" ]] || { usage; exit 2; }
command -v tar >/dev/null
command -v zstd >/dev/null
command -v split >/dev/null

case "$GROUP" in
  TTT_ADAPTERS)
    BASE=/workspace/arc2
    ROOT=active_runs/eval60_authoritative_greedy_v1/checkpoints
    TAG=data-eval60-adapters-v1
    NAME=eval60_adapters_v1
    ;;
  GREEDY_RAW)
    BASE=/workspace/arc2
    ROOT=active_runs/eval60_authoritative_greedy_v1
    TAG=data-eval60-greedy-raw-v1
    NAME=eval60_greedy_raw_v1
    ;;
  V5_RAW)
    BASE=/workspace/arc2
    ROOT=active_runs/eval60_v5_vs_greedy_4worker_v1
    TAG=data-eval60-v5-snapshot-1068-v1
    NAME=eval60_v5_snapshot_1068_v1
    ;;
  BASE_MODEL)
    [[ "$MODEL_STATUS" == "ALLOWED" ]] || { echo "BASE_MODEL blocked: MODEL_REDISTRIBUTION_STATUS=$MODEL_STATUS" >&2; exit 3; }
    BASE=/root/arc-runtime-turbodfs-v5-benchmark/model-stage
    ROOT=qwen3_4b_grids15_sft139
    TAG=data-arc-base-model-v1
    NAME=arc_base_model_v1
    ;;
  *) usage; exit 2 ;;
esac

SOURCE="$BASE/$ROOT"
[[ -d "$SOURCE" ]] || { echo "missing source: $SOURCE" >&2; exit 1; }
mkdir -p "$STAGE/$GROUP"
LIST="$STAGE/$GROUP/$NAME.files0"
PART_PREFIX="$STAGE/$GROUP/$NAME.tar.zst.part"

# Only file paths expressly inside the selected source are eligible. Exclude
# known credentials, caches, and the nonauthoritative FUSE state database.
python3 - "$BASE" "$ROOT" "$GROUP" "$LIST" "$MANIFEST" <<'PY'
import csv, hashlib, re, sys
from pathlib import Path
base = Path(sys.argv[1])
root = Path(sys.argv[2])
group = sys.argv[3]
list_path = Path(sys.argv[4])
manifest = Path(sys.argv[5])
root_path = base / root
deny_name = re.compile(r'(^\.env($|\.)|(^|/)(credentials?|secrets?)(/|$)|\.(pem|key)$|(^|/)(id_rsa|id_ed25519)(\.pub)?$|(^|/)(cookies?|auth)(/|$))', re.I)
content_marker = re.compile(rb'(github_pat_|ghp_[A-Za-z0-9]|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
rows, included, excluded = [], [], []
for path in sorted(root_path.rglob('*')):
    if not path.is_file():
        continue
    rel = path.relative_to(base).as_posix()
    reason = ''
    if deny_name.search(rel):
        reason = 'SECURITY_NAME_DENYLIST'
    if group == 'GREEDY_RAW' and '/checkpoints/' in f'/{rel}':
        reason = 'ARCHIVED_SEPARATELY_AS_TTT_ADAPTERS'
    if group == 'V5_RAW' and path.name == 'run_state.sqlite':
        reason = 'CORRUPTED_NONAUTHORITATIVE_SQLITE'
    size = path.stat().st_size
    if not reason and size <= 10 * 1024 * 1024:
        try:
            with path.open('rb') as h:
                if content_marker.search(h.read()):
                    reason = 'SECURITY_CONTENT_DENYLIST'
        except OSError:
            reason = 'UNREADABLE'
    digest = hashlib.sha256()
    try:
        with path.open('rb') as h:
            for chunk in iter(lambda: h.read(1024 * 1024), b''):
                digest.update(chunk)
        sha = digest.hexdigest()
    except OSError:
        sha, reason = '', reason or 'UNREADABLE'
    row = [group, rel, size, sha, str(path), 'INCLUDED' if not reason else 'EXCLUDED', reason]
    rows.append(row)
    (included if not reason else excluded).append(rel)
security_reasons = {'SECURITY_NAME_DENYLIST', 'SECURITY_CONTENT_DENYLIST', 'UNREADABLE'}
if any(r[-1] in security_reasons for r in rows if r[-2] == 'EXCLUDED'):
    reasons = sorted({r[-1] for r in rows if r[-1] in security_reasons})
    raise SystemExit('Security exclusions found: ' + ','.join(reasons) + '. Review manifest; do not archive automatically.')
Path(list_path).parent.mkdir(parents=True, exist_ok=True)
with open(list_path, 'wb') as h:
    for rel in included:
        h.write(rel.encode() + b'\0')
write_header = not Path(manifest).exists()
Path(manifest).parent.mkdir(parents=True, exist_ok=True)
with open(manifest, 'a', encoding='utf-8', newline='') as h:
    w = csv.writer(h)
    if write_header: w.writerow(['archive_group','relative_path','size_bytes','sha256','source_path','include_status','exclusion_reason'])
    w.writerows(rows)
print(f'INCLUDED_FILES={len(included)}')
PY

# Streaming keeps a second monolithic 178 GiB archive off the Pod overlay.
rm -f "${PART_PREFIX}"*
tar -C "$BASE" --null --verbatim-files-from --files-from="$LIST" -cf - \
  | nice -n 19 ionice -c3 zstd -3 -T1 -c \
  | split -b 1500M -d -a 3 - "${PART_PREFIX}"

(cd "$STAGE/$GROUP" && sha256sum "${NAME}".tar.zst.part* > "${NAME}.parts.sha256")
printf '%s\n' "$TAG" > "$STAGE/$GROUP/release_tag.txt"
