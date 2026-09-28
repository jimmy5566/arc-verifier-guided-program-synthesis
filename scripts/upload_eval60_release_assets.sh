#!/usr/bin/env bash
# Resumable GitHub Release asset uploader. Requires GH_TOKEN in the environment.
set -euo pipefail

REPO="jimmy5566/arc-verifier-guided-program-synthesis"
TAG=""
TITLE=""
PART_DIR=""
RESULTS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG=$2; shift 2 ;;
    --title) TITLE=$2; shift 2 ;;
    --part-dir) PART_DIR=$2; shift 2 ;;
    --results) RESULTS=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TAG" && -n "$TITLE" && -n "$PART_DIR" && -n "$RESULTS" ]] || exit 2
: "${GH_TOKEN:?GH_TOKEN must be present in the environment}"

api() { curl --fail-with-body --silent --show-error -H "Authorization: Bearer $GH_TOKEN" -H 'X-GitHub-Api-Version: 2022-11-28' -H 'Accept: application/vnd.github+json' "$@"; }
release_json=$(api "https://api.github.com/repos/$REPO/releases/tags/$TAG" || true)
if [[ -z "$release_json" ]]; then
  body=$(python3 - "$TAG" "$TITLE" <<'PY'
import json,sys
print(json.dumps({'tag_name':sys.argv[1],'name':sys.argv[2],'draft':False,'prerelease':False}))
PY
)
  release_json=$(api -X POST -H 'Content-Type: application/json' -d "$body" "https://api.github.com/repos/$REPO/releases")
fi
release_id=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"$release_json")
# A full adapter archive is deliberately sharded and may exceed GitHub's
# 100-assets-per-page API default.  Fetch every asset page so resume never
# reuploads an existing shard simply because it appears on a later page.
all_assets() {
  local page=1 response count
  local merged='[]'
  while true; do
    response=$(api "https://api.github.com/repos/$REPO/releases/$release_id/assets?per_page=100&page=$page")
    count=$(python3 -c 'import json,sys; print(len(json.load(sys.stdin)))' <<<"$response")
    merged=$(python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1]) + json.load(sys.stdin)))' "$merged" <<<"$response")
    [[ "$count" -lt 100 ]] && break
    page=$((page + 1))
  done
  printf '%s' "$merged"
}
assets_json=$(all_assets)
mkdir -p "$(dirname "$RESULTS")"
[[ -f "$RESULTS" ]] || echo 'release_tag,archive_group,part_index,asset_name,size_bytes,sha256,remote_status,remote_asset_identifier' > "$RESULTS"
for part in "$PART_DIR"/*.tar.zst.part* "$PART_DIR"/*.parts.sha256; do
  [[ -f "$part" ]] || continue
  name=$(basename "$part")
  size=$(stat -c '%s' "$part")
  sha=$(sha256sum "$part" | awk '{print $1}')
  existing=$(python3 -c 'import json,sys
name,size=sys.argv[1],int(sys.argv[2])
for asset in json.load(sys.stdin):
    if asset["name"] == name and asset["size"] == size:
        print(asset["id"]); break' "$name" "$size" <<<"$assets_json")
  if [[ -n "$existing" ]]; then
    echo "$TAG,${TAG#data-},${name##*.part},$name,$size,$sha,SKIPPED_EXISTING,$existing" >> "$RESULTS"
    continue
  fi
  encoded=$(python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$name")
  response=$(curl --fail-with-body --silent --show-error -X POST -H "Authorization: Bearer $GH_TOKEN" -H 'Content-Type: application/octet-stream' --data-binary "@$part" "https://uploads.github.com/repos/$REPO/releases/$release_id/assets?name=$encoded")
  asset_id=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"$response")
  echo "$TAG,${TAG#data-},${name##*.part},$name,$size,$sha,UPLOADED,$asset_id" >> "$RESULTS"
  assets_json=$(all_assets)
done
