"""Production-shaped, stop-before-model preclaim for count-token supervision.

The real launch must call ``validate_preclaim`` before it may claim a nonce.
This module intentionally imports no torch, transformers, CUDA, model, or target
files at module load or through the preclaim branch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True, separators=(',', ':'))
        handle.write('\n')
        temporary = Path(handle.name)
    os.replace(temporary, path)


def fail(code: str) -> None:
    raise RuntimeError(f'COUNT_TOKEN_PRECLAIM:{code}')


def serializer_task(item: dict) -> dict:
    task = copy.deepcopy(item['task'])
    task.setdefault('source_id', f"SERIALIZER_VIEW:{item['canonical_base_id']}")
    return task


def validate_binding(binding: dict) -> None:
    required = {'experiment_id', 'scientific_source_commit', 'source_files', 'inputs', 'arms', 'v7', 'runtime'}
    if not required.issubset(binding):
        fail('BINDING_SCHEMA')
    if binding['experiment_id'] != 'CONNECTED_COMPONENT_COUNT_DECISION_TOKEN_SUPERVISION_PILOT_V1':
        fail('EXPERIMENT_ID')
    for relative, expected in binding['source_files'].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            fail(f'SOURCE_IDENTITY:{relative}')
    for name, item in binding['inputs'].items():
        path = ROOT / item['path']
        if not path.is_file() or sha256(path) != item['sha256']:
            fail(f'INPUT_IDENTITY:{name}')
    v7 = binding['v7']
    if not {'checkpoint_manifest_path', 'checkpoint_manifest_sha256'}.issubset(v7):
        fail('V7_BINDING_SCHEMA')
    checkpoint_manifest = ROOT / v7['checkpoint_manifest_path']
    if not checkpoint_manifest.is_file() or sha256(checkpoint_manifest) != v7['checkpoint_manifest_sha256']:
        fail('V7_MANIFEST_IDENTITY')
    if binding['arms']['control']['schedule_sha256'] != binding['arms']['treatment']['schedule_sha256']:
        fail('ARM_SCHEDULE_MISMATCH')
    if binding['arms']['control']['schedule_sha256'] != binding['inputs']['diverse_schedule']['sha256']:
        fail('SCHEDULE_BINDING_MISMATCH')
    treatment = binding['arms']['treatment']['loss_allocation']
    if treatment != {'kind': 'COUNT_TERMINAL_TOKEN_REWEIGHTED', 'decision_token_multiplier': 2.0}:
        fail('TREATMENT_ALLOCATION')
    if binding['arms']['control']['loss_allocation'] != {'kind': 'UNIFORM_SEQUENCE_LOSS'}:
        fail('CONTROL_ALLOCATION')


def validate_runtime_import_closure() -> list[str]:
    # The production worker imports these same CPU-safe modules before model load.
    names = [
        'inference.arc_native_io',
        'inference.nvarc_native',
        'training_data.pipeline',
        'scripts.e04_e_equal_slot_loss',
        'scripts.decision_token_supervision',
        'scripts.connected_components_object_extraction_semantic_oracle',
        'scripts.run_connected_components_object_extraction_pilot_v1',
    ]
    for name in names:
        importlib.import_module(name)
    return names


def validate_schedule(binding: dict) -> dict:
    from scripts.connected_components_object_extraction_semantic_oracle import connected_component_count
    from scripts.decision_token_supervision import count_terminal_label_index, matched_weights
    from training_data.pipeline import task_to_sample

    schedule_path = ROOT / binding['inputs']['diverse_schedule']['path']
    episodes = json.loads(schedule_path.read_text(encoding='utf-8'))['episodes']
    if len(episodes) != 384 or [row.get('slot') for row in episodes] != list(range(384)):
        fail('SCHEDULE_SLOT_CLOSURE')
    expected = {'PRIMARY_CONNECTED_COMPONENT_COUNT': 48, 'PRIMARY_OBJECT_EXTRACTION': 48, 'PROTECTED_REPLAY': 288}
    observed = {role: sum(row.get('role') == role for row in episodes) for role in expected}
    if observed != expected:
        fail(f'SCHEDULE_INVENTORY:{observed}')
    count_checks = []
    supervised = 0
    for row in episodes:
        sample = task_to_sample(serializer_task(row))
        supervised += sample['assistant_token_count']
        if row['role'] != 'PRIMARY_CONNECTED_COMPONENT_COUNT':
            continue
        task = row['task']
        expected_output = connected_component_count(task['test'][-1]['input'])
        if task['test'][-1]['output'] != expected_output:
            fail(f'COUNT_ORACLE:{row["slot"]}')
        index, count = count_terminal_label_index(task)
        expected_token = expected_output[1][count - 1]
        if sample['labels'][index] != expected_token or sample['input_ids'][index] != expected_token:
            fail(f'DECISION_TOKEN_ALIGNMENT:{row["slot"]}')
        control, treatment = matched_weights(sample['labels'], index, multiplier=2.0)
        if set(control) != set(treatment) or abs(sum(control.values()) - sum(treatment.values())) > 1e-12:
            fail(f'LOSS_MASS:{row["slot"]}')
        count_checks.append({'slot': row['slot'], 'token_index': index, 'component_count': count, 'valid_token_mass': sum(control.values())})
    if len(count_checks) != 48:
        fail('COUNT_SLOT_CLOSURE')
    return {'slots': len(episodes), 'inventory': observed, 'supervised_tokens': supervised, 'count_token_checks': len(count_checks), 'count_token_examples': count_checks}


def validate_preclaim(binding_path: Path) -> dict:
    binding = json.loads(binding_path.read_text(encoding='utf-8'))
    validate_binding(binding)
    imports = validate_runtime_import_closure()
    schedule = validate_schedule(binding)
    return {
        'status': 'PASS_PRODUCTION_SHAPED_PRECLAIM_ONLY',
        'experiment_id': binding['experiment_id'],
        'scientific_source_commit': binding['scientific_source_commit'],
        'runtime_imports': imports,
        'schedule': schedule,
        'arms_share_exact_schedule': True,
        'model_loaded': False,
        'gpu_used': False,
        'cuda_initialized': False,
        'sealed_sidecar_opened': False,
        'authorization_claimed': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--preclaim-only', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if not args.preclaim_only:
        raise SystemExit('only --preclaim-only is enabled in this CPU-only stage')
    receipt = validate_preclaim(args.binding)
    if args.receipt:
        atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
