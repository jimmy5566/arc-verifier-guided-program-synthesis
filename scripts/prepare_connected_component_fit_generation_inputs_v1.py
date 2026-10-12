#!/usr/bin/env python3
"""Build target-free real-input material for the component-count successor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = (ROOT / 'experiments/capability_repair_baseline_v1'
        / 'connected_components_object_extraction_capability_pilot_v1')
DIAGNOSTIC = BASE / 'connected_component_count_fit_vs_generation_diagnostic_v1'
INVENTORY = DIAGNOSTIC / 'RUN_002_REUSE_AND_MISSING_ROW_INVENTORY.json'
TRAIN = BASE / 'measurement_contract_v1' / 'TRAIN_FIT_INPUT_PROMPTS.json'
OOD = BASE / 'OOD_INPUT_PROMPTS.json'


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_records(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    return value.get('records', value) if isinstance(value, dict) else value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def source_by_task_hash():
    source = {}
    for split, path in (('TRAIN_FIT', TRAIN), ('OOD', OOD)):
        for record in read_records(path):
            task = record.get('task')
            if not isinstance(task, dict):
                raise RuntimeError(f'{split}_TASK_MISSING')
            digest = canonical_sha256(task)
            if digest in source:
                raise RuntimeError(f'{split}_DUPLICATE_TASK_HASH')
            # Deliberately retain task input/train demonstrations only. This
            # source never reads the sealed correct-token sidecar.
            source[digest] = {'split': split, 'task': task}
    return source


def build():
    inventory = json.loads(INVENTORY.read_text(encoding='utf-8'))
    rows = inventory.get('inventory')
    if not isinstance(rows, list) or len(rows) != 216:
        raise RuntimeError('INVENTORY_ROW_COUNT_INVALID')
    source = source_by_task_hash()
    cohorts = {}
    missing = []
    for row in rows:
        split = row.get('split')
        cohort = row.get('cohort_record_id')
        digest = row.get('prompt_task_sha256')
        item = source.get(digest)
        if (not isinstance(cohort, str) or not isinstance(digest, str)
                or not item or item['split'] != split):
            raise RuntimeError('INVENTORY_PROMPT_SOURCE_MISMATCH')
        previous = cohorts.get(cohort)
        candidate = {'cohort_record_id': cohort, 'split': split,
                     'prompt_task_sha256': digest, 'task': item['task']}
        if previous is not None and previous != candidate:
            raise RuntimeError('COHORT_IDENTITY_COLLISION')
        cohorts[cohort] = candidate
        if row.get('disposition') == 'GENERATE_NEW_ONCE_AFTER_FRESH_AUTHORIZATION':
            missing.append({'checkpoint': row['checkpoint'], **candidate})
    if len(cohorts) != 72 or len(missing) != 48:
        raise RuntimeError('COHORT_OR_MISSING_LAYOUT_INVALID')
    if len({(x['checkpoint'], x['cohort_record_id']) for x in missing}) != 48:
        raise RuntimeError('MISSING_ROW_DUPLICATE')
    return cohorts, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-out', type=Path, required=True)
    parser.add_argument('--missing-out', type=Path, required=True)
    args = parser.parse_args()
    cohorts, missing = build()
    cohort_payload = {'schema_version': 1, 'target_content_present': False,
                      'records': [cohorts[k] for k in sorted(cohorts)]}
    missing_payload = {'schema_version': 1, 'target_content_present': False,
                       'records': sorted(missing, key=lambda x: (x['checkpoint'], x['cohort_record_id']))}
    result = {
        'cohort_records': len(cohort_payload['records']),
        'missing_records': len(missing_payload['records']),
        'cohort_sha256': atomic_json(args.cohort_out, cohort_payload),
        'missing_sha256': atomic_json(args.missing_out, missing_payload),
        'target_content_present': False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
