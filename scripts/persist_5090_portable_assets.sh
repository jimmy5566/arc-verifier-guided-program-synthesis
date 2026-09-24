#!/usr/bin/env bash
# Seed an immutable Global Volume from the validated Blackwell environment.
# This intentionally refuses to overwrite a previous seed and never copies
# tokens, credentials, private solutions, model caches, or temporary state.
set -Eeuo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 PERSISTENT_ROOT MODEL_DIR ENV_DIR REPO_ROOT [FROZEN_RUN_DIR ...]" >&2
  exit 64
fi

persistent_root="$1"
model_source="$2"
environment_source="$3"
repo_source="$4"
shift 4
global_mount="${ARC2_GLOBAL_MOUNT:-/workspace-global}"
staging="${persistent_root}.staging.$$"

if ! findmnt -T "${global_mount}" >/dev/null 2>&1; then
  echo "GLOBAL_VOLUME_UNAVAILABLE: ${global_mount}" >&2
  exit 2
fi
if [[ -e "${persistent_root}" ]]; then
  echo "PERSISTENT_ROOT_EXISTS: ${persistent_root}; preserve the existing authoritative store" >&2
  exit 3
fi
for required in "${model_source}/config.json" "${environment_source}/bin/python" "${repo_source}/.git"; do
  if [[ ! -e "${required}" ]]; then
    echo "SOURCE_ASSET_MISSING: ${required}" >&2
    exit 4
  fi
done
if ! command -v rsync >/dev/null; then
  echo "RSYNC_REQUIRED" >&2
  exit 5
fi

trap 'rm -rf "${staging}"' EXIT
mkdir -p "${staging}"/{models,wheels,env,benchmarks,bootstrap,frozen-runs}
rsync -a --checksum "${model_source}/" "${staging}/models/qwen3_4b_grids15_sft139/"
rsync -a --checksum "${environment_source}/" "${staging}/env/5090-blackwell-env-v2/"
cp "${repo_source}/governance/benchmarks/legacy-eval3/eval3_manifest.json" "${staging}/benchmarks/eval3_manifest.json"
cp "${repo_source}/governance/registry/cohorts/eval60.json" "${staging}/benchmarks/eval60_manifest.json"
cp "${repo_source}/governance/environments/5090-blackwell-env-v2.json" "${staging}/benchmarks/5090-blackwell-env-v2-governance.json"
cp "${repo_source}/scripts/bootstrap_5090_research_env.sh" "${staging}/bootstrap/"
cp "${repo_source}/scripts/sync_arc2_run_to_persistent.sh" "${staging}/bootstrap/"
cp "${repo_source}/scripts/prepare_2x5090_runtime.py" "${staging}/bootstrap/"

for frozen_run in "$@"; do
  if [[ ! -d "${frozen_run}" ]]; then
    echo "FROZEN_RUN_MISSING: ${frozen_run}" >&2
    exit 6
  fi
  run_name="$(basename "${frozen_run}")"
  rsync -a --checksum --delete \
    --exclude='*solution*' --exclude='*token*' --exclude='*.key' --exclude='credentials*' \
    "${frozen_run}/" "${staging}/frozen-runs/${run_name}/"
done

# Copy existing nonstandard wheels only when the operator explicitly identifies
# a local wheel cache. The validated xFormers distribution remains portable via
# the staged environment even if no original .whl was retained.
if [[ -n "${ARC2_WHEEL_SOURCE_DIR:-}" ]]; then
  if [[ ! -d "${ARC2_WHEEL_SOURCE_DIR}" ]]; then
    echo "WHEEL_SOURCE_MISSING: ${ARC2_WHEEL_SOURCE_DIR}" >&2
    exit 7
  fi
  rsync -a --checksum --include='*.whl' --exclude='*' "${ARC2_WHEEL_SOURCE_DIR}/" "${staging}/wheels/"
fi

"${environment_source}/bin/python" - "${staging}" "${repo_source}" <<'PY'
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import unsloth
import torch
import transformers
import peft
import xformers

root = Path(sys.argv[1])
repo = Path(sys.argv[2])
model = root / 'models' / 'qwen3_4b_grids15_sft139'
def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

versions = {
    'python': '.'.join(map(str, sys.version_info[:3])),
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'transformers': transformers.__version__, 'unsloth': unsloth.__version__,
    'peft': peft.__version__, 'xformers': xformers.__version__,
    'torchao': importlib.metadata.version('torchao'),
    'triton': importlib.metadata.version('triton'),
}
record = importlib.metadata.distribution('xformers').read_text('RECORD') or ''
record_hash = hashlib.sha256(record.encode('utf-8')).hexdigest()
git_commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
(root / 'env' / 'environment_lock.txt').write_text(
    subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8'
)
model_files = [
    {'path': str(path.relative_to(model)), 'bytes': path.stat().st_size, 'sha256': digest(path)}
    for path in sorted(model.rglob('*')) if path.is_file() and path.name != 'model_manifest.json'
]
(model / 'model_manifest.json').write_text(json.dumps({
    'model_identifier': 'sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1',
    'model_config_sha256': digest(model / 'config.json'), 'files': model_files,
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(root / 'env' / 'environment_manifest.json').write_text(json.dumps({
    'environment_id': '5090-blackwell-env-v2', 'git_commit': git_commit,
    'versions': versions, 'attention_backend': 'xformers_verified_in_5090-blackwell-env-v2',
    'xformers_distribution_record_sha256': record_hash,
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
wheels = sorted({'name': path.name, 'sha256': digest(path)} for path in (root / 'wheels').glob('*.whl'))
(root / 'wheels' / 'xformers_source_identity.json').write_text(json.dumps({
    'xformers_distribution_record_sha256': record_hash,
    'xformers_version': xformers.__version__, 'wheels': wheels,
    'identity_source': 'validated installed distribution RECORD',
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
manifest = []
for path in sorted(root.rglob('*')):
    if path.is_file(): manifest.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': digest(path)})
(root / 'portable_storage_manifest.json').write_text(json.dumps({
    'schema_version': 'ARC2_PORTABLE_STORAGE_V1', 'git_commit': git_commit,
    'files': manifest, 'credentials_included': False, 'solutions_included': False,
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(root / 'local_backup_manifest.json').write_text(json.dumps({
    'schema_version': 'ARC2_LOCAL_BACKUP_V1',
    'source_manifest': 'portable_storage_manifest.json',
    'download_root': str(root), 'credentials_included': False,
    'instructions': 'Download or sync only files enumerated by portable_storage_manifest.json; do not copy hf_cache, token files, or private solution files.',
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

if find "${staging}" -type f \( -iname '*.pem' -o -iname '*.key' -o -iname '*credential*' \) -print -quit | grep -q .; then
  echo "CREDENTIAL_EXCLUSION_FAILED" >&2
  exit 8
fi
mv "${staging}" "${persistent_root}"
trap - EXIT
echo "PORTABLE_STORAGE_SEED_PASS: ${persistent_root}"
