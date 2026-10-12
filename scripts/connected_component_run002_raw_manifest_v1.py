#!/usr/bin/env python3
"""Exact-byte manifest and verifier for immutable RUN_002 raw reuse rows.

The manifest deliberately contains no raw generation content.  It binds every
reusable row to the canonical SHA already frozen in the inventory.  A runtime
root is accepted only when all 168 immutable rows validate before any model
load or fresh generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = (ROOT / 'experiments/capability_repair_baseline_v1' /
        'connected_components_object_extraction_capability_pilot_v1' /
        'connected_component_count_fit_vs_generation_diagnostic_v1')
INVENTORY = BASE / 'RUN_002_REUSE_AND_MISSING_ROW_INVENTORY.json'
EXPECTED_INVENTORY_SHA256 = 'b0cc5b45adffcc618fd622786e14dfe3fdbe6aec20222268ba1d694741793d95'


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def build_manifest(inventory: dict[str, Any], inventory_sha256: str) -> dict[str, Any]:
    rows = inventory.get('inventory') if isinstance(inventory, dict) else None
    if inventory_sha256 != EXPECTED_INVENTORY_SHA256:
        raise RuntimeError('RUN_002_INVENTORY_SHA_MISMATCH')
    if not isinstance(rows, list) or len(rows) != 216:
        raise RuntimeError('RUN_002_INVENTORY_LAYOUT_INVALID')
    reused = [row for row in rows if row.get('disposition') == 'REUSE_IMMUTABLE_RUN_002_RAW']
    fresh = [row for row in rows if row.get('disposition') == 'GENERATE_NEW_ONCE_AFTER_FRESH_AUTHORIZATION']
    if len(reused) != 168 or len(fresh) != 48:
        raise RuntimeError('RUN_002_REUSE_PARTITION_INVALID')
    required = ('checkpoint', 'cohort_record_id', 'prompt_task_sha256', 'run_002_raw_file',
                'run_002_record_id', 'run_002_raw_record_sha256', 'run_002_effective_batch_size')
    seen: set[tuple[str, str]] = set()
    expected_rows = []
    for row in reused:
        if any(not isinstance(row.get(key), str) for key in required[:-1]) or row.get('run_002_effective_batch_size') not in {1, 4, 8, 16, 32}:
            raise RuntimeError('RUN_002_REUSED_ROW_SCHEMA_INVALID')
        key = (row['checkpoint'], row['prompt_task_sha256'])
        if key in seen:
            raise RuntimeError('RUN_002_REUSED_ROW_DUPLICATE')
        seen.add(key)
        expected_rows.append({key: row[key] for key in required})
    file_counts = Counter(row['run_002_raw_file'] for row in expected_rows)
    if set(file_counts) != {'V7_RAW.jsonl', 'CONTROL_NARROW_STRUCTURE_RAW.jsonl', 'TREATMENT_STRUCTURE_DIVERSE_RAW.jsonl'}:
        raise RuntimeError('RUN_002_RAW_FILE_SET_INVALID')
    return {
        'schema_version': 1,
        'status': 'RUN_002_RAW_ARTIFACT_MANIFEST_FROZEN_RAW_BYTES_NOT_MATERIALIZED',
        'run_id': 'RUN_002_09ff2d41a20e291a5900c6f73897fed5',
        'inventory_sha256': inventory_sha256,
        'reused_rows': 168,
        'fresh_rows_excluded': 48,
        'raw_file_row_counts': dict(sorted(file_counts.items())),
        'verification': {
            'canonical_record_sha256': 'json.dumps(sort_keys=True,separators=(\',\',\':\'))',
            'verify_before_model_load': True,
            'reject_duplicate_prompt_sha256': True,
            'reject_target_fields': ['target_grid', 'target_token_ids', 'gold', 'dGold'],
            'whole_file_sha256': 'UNAVAILABLE_NOT_SUBSTITUTED_RECORD_LEVEL_IDENTITIES_AUTHORITATIVE',
        },
        'expected_rows': sorted(expected_rows, key=lambda row: (row['checkpoint'], row['prompt_task_sha256'])),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f'RUN_002_RAW_FILE_UNREADABLE:{path.name}') from error
    if not records or any(not isinstance(row, dict) for row in records):
        raise RuntimeError(f'RUN_002_RAW_FILE_LAYOUT_INVALID:{path.name}')
    return records


def verify_run002_raw_root(manifest: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    expected = manifest.get('expected_rows') if isinstance(manifest, dict) else None
    if not isinstance(expected, list) or not expected:
        raise RuntimeError('RUN_002_MANIFEST_LAYOUT_INVALID')
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in expected:
        if not isinstance(row, dict) or not isinstance(row.get('run_002_raw_file'), str):
            raise RuntimeError('RUN_002_MANIFEST_ROW_INVALID')
        grouped.setdefault(row['run_002_raw_file'], []).append(row)
    verified = []
    for name, required in sorted(grouped.items()):
        source = raw_root / name
        if not source.is_file():
            raise RuntimeError(f'RUN_002_RAW_FILE_MISSING:{name}')
        by_prompt: dict[str, dict[str, Any]] = {}
        for record in load_jsonl(source):
            prompt_sha = record.get('prompt_sha256')
            if not isinstance(prompt_sha, str):
                continue
            if prompt_sha in by_prompt:
                raise RuntimeError(f'RUN_002_RAW_PROMPT_DUPLICATE:{name}')
            by_prompt[prompt_sha] = record
        for item in required:
            record = by_prompt.get(item['prompt_task_sha256'])
            if record is None:
                raise RuntimeError(f'RUN_002_RAW_PROMPT_MISSING:{name}')
            if record.get('record_id') != item['run_002_record_id']:
                raise RuntimeError(f'RUN_002_RAW_RECORD_ID_MISMATCH:{name}')
            if record.get('effective_batch_size') != item['run_002_effective_batch_size']:
                raise RuntimeError(f'RUN_002_RAW_BATCH_MISMATCH:{name}')
            if {'target_grid', 'target_token_ids', 'gold', 'dGold'} & set(record):
                raise RuntimeError(f'RUN_002_RAW_TARGET_BOUNDARY_VIOLATION:{name}')
            if canonical_sha256(record) != item['run_002_raw_record_sha256']:
                raise RuntimeError(f'RUN_002_RAW_RECORD_SHA_MISMATCH:{name}')
            verified.append({'checkpoint': item['checkpoint'], 'prompt_task_sha256': item['prompt_task_sha256'],
                             'raw_file': name, 'raw_record_sha256': item['run_002_raw_record_sha256']})
    if len(verified) != len(expected) or len({(row['checkpoint'], row['prompt_task_sha256']) for row in verified}) != len(expected):
        raise RuntimeError('RUN_002_RAW_VERIFICATION_CARDINALITY_INVALID')
    return {'status': 'PASS_RUN_002_EXACT_RAW_RECORD_MANIFEST', 'verified_rows': len(verified),
            'verified_files': sorted(grouped), 'raw_bytes_copied': False, 'model_loaded': False, 'gpu_used': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, default=INVENTORY)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-root', type=Path)
    args = parser.parse_args()
    manifest = build_manifest(json.loads(args.inventory.read_text(encoding='utf-8')), sha_file(args.inventory))
    result = {'manifest_sha256': atomic_json(args.out, manifest), 'status': manifest['status'],
              'reused_rows': manifest['reused_rows'], 'model_loaded': False, 'gpu_used': False}
    if args.verify_root:
        result['verification'] = verify_run002_raw_root(manifest, args.verify_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
