#!/usr/bin/env bash
# Restore one or more Eval60 release-archive groups into a chosen destination.
set -euo pipefail

REPO="jimmy5566/arc-verifier-guided-program-synthesis"
DEST=/workspace/arc2/restored
MODE=""
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FILE_MANIFEST="$SCRIPT_DIR/artifacts/eval60_full_archive/FULL_ARCHIVE_FILES.csv"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --destination) DEST=$2; shift 2 ;;
    --file-manifest) FILE_MANIFEST=$2; shift 2 ;;
    --model|--adapters|--greedy|--v5) MODE+=" ${1#--}"; shift ;;
    --all) MODE="model adapters greedy v5"; shift ;;
    *) echo "usage: $0 [--destination PATH] --model|--adapters|--greedy|--v5|--all" >&2; exit 2 ;;
  esac
done
[[ -n "$MODE" ]] || { echo "select at least one archive group" >&2; exit 2; }
command -v zstd >/dev/null
mkdir -p "$DEST/.release_downloads"

download_release() {
  local tag=$1 prefix=$2 dir="$DEST/.release_downloads/$tag"
  mkdir -p "$dir"
  if command -v gh >/dev/null 2>&1; then
    gh release download "$tag" --repo "$REPO" --pattern "${prefix}*" --dir "$dir" --clobber
  else
    : "${GH_TOKEN:?GH_TOKEN is required when gh is unavailable}"
    python3 - "$REPO" "$tag" "$prefix" "$dir" <<'PY'
import json, os, sys, urllib.request
repo, tag, prefix, out = sys.argv[1:]
req=urllib.request.Request(f'https://api.github.com/repos/{repo}/releases/tags/{tag}', headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
release=json.load(urllib.request.urlopen(req))
for asset in release['assets']:
    if asset['name'].startswith(prefix):
        r=urllib.request.Request(asset['url'], headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/octet-stream'})
        with urllib.request.urlopen(r) as src, open(os.path.join(out,asset['name']),'wb') as dst:
            while b:=src.read(1024*1024): dst.write(b)
PY
  fi
  (cd "$dir" && sha256sum -c "${prefix}.parts.sha256")
  cat "$dir"/*.tar.zst.part* | zstd -d -c | tar -C "$DEST" -xf -
}
for item in $MODE; do
  case "$item" in
    model) download_release data-arc-base-model-v1 arc_base_model_v1 ;;
    adapters) download_release data-eval60-adapters-v1 eval60_adapters_v1 ;;
    greedy) download_release data-eval60-greedy-raw-v1 eval60_greedy_raw_v1 ;;
    v5) download_release data-eval60-v5-snapshot-1068-v1 eval60_v5_snapshot_1068_v1 ;;
  esac
done

if [[ -f "$FILE_MANIFEST" ]]; then
  python3 - "$FILE_MANIFEST" "$DEST" <<'PY'
import csv, hashlib, sys
manifest, dest = sys.argv[1:]
bad=[]
with open(manifest, newline='', encoding='utf-8') as h:
    for row in csv.DictReader(h):
        if row['include_status'] != 'INCLUDED':
            continue
        p = __import__('pathlib').Path(dest) / row['relative_path']
        if not p.is_file():
            bad.append('missing:'+row['relative_path']); continue
        d=hashlib.sha256(p.read_bytes()).hexdigest()
        if d != row['sha256']: bad.append('hash:'+row['relative_path'])
if bad: raise SystemExit('extracted manifest verification failed: '+','.join(bad[:10]))
print('EXTRACTED_FILE_MANIFEST=PASS')
PY
fi
