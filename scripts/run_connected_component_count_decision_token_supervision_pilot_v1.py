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
import math
import os
import subprocess
import sys
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
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='\n', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True, separators=(',', ':'))
        handle.write('\n')
        temporary = Path(handle.name)
    os.replace(temporary, path)


def fail(code: str) -> None:
    raise RuntimeError(f'COUNT_CODE_PAIR_PRECLAIM:{code}')


def serializer_task(item: dict) -> dict:
    task = copy.deepcopy(item['task'])
    task.setdefault('source_id', f"SERIALIZER_VIEW:{item['canonical_base_id']}")
    return task


def validate_binding(binding: dict) -> None:
    required = {'experiment_id', 'scientific_source_commit', 'source_files', 'inputs', 'arms', 'v7', 'runtime', 'execution', 'training', 'evaluation', 'dataset_paths'}
    if not required.issubset(binding):
        fail('BINDING_SCHEMA')
    if binding['experiment_id'] != 'CONNECTED_COMPONENT_COUNT_CODE_PAIR_SUPERVISION_PILOT_V1':
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
    if treatment != {'kind': 'COUNT_CODE_PAIR_REWEIGHTED', 'decision_token_multiplier': 2.0}:
        fail('TREATMENT_ALLOCATION')
    if binding['arms']['control']['loss_allocation'] != {'kind': 'UNIFORM_SEQUENCE_LOSS'}:
        fail('CONTROL_ALLOCATION')
    execution = binding['execution']
    if execution.get('launcher_path') != 'scripts/run_connected_component_count_decision_token_supervision_pilot_v1.py':
        fail('EXECUTION_LAUNCHER_PATH')
    if execution.get('worker_path') != 'scripts/run_connected_component_count_code_pair_supervision_worker_v1.py':
        fail('EXECUTION_WORKER_PATH')
    if execution.get('retry') is not False or execution.get('authorization_required') is not True:
        fail('EXECUTION_AUTHORIZATION')


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
        'scripts.score_connected_components_object_extraction_pilot_v1',
    ]
    for name in names:
        importlib.import_module(name)
    return names


def validate_schedule(binding: dict) -> dict:
    from scripts.connected_components_object_extraction_semantic_oracle import connected_component_count
    from scripts.decision_token_supervision import count_code_pair_label_indices, matched_weights
    from training_data.pipeline import task_to_sample

    schedule_path = ROOT / binding['inputs']['diverse_schedule']['path']
    episodes = json.loads(schedule_path.read_text(encoding='utf-8'))['episodes']
    if len(episodes) != 384 or [row.get('slot') for row in episodes] != list(range(384)):
        fail('SCHEDULE_SLOT_CLOSURE')
    expected = {'PRIMARY_CONNECTED_COMPONENT_COUNT': 48, 'PRIMARY_OBJECT_EXTRACTION': 48, 'PROTECTED_REPLAY': 288}
    observed = {role: sum(row.get('role') == role for row in episodes) for role in expected}
    if observed != expected:
        fail(f'SCHEDULE_INVENTORY:{observed}')
    focused_checks = 0
    supervised = 0
    for row in episodes:
        sample = task_to_sample(serializer_task(row))
        supervised += sample['assistant_token_count']
        if row['role'] != 'PRIMARY_CONNECTED_COMPONENT_COUNT':
            continue
        task = row['task']
        for pair in [*task['train'], *task['test']]:
            if pair['output'] != connected_component_count(pair['input']):
                fail(f'COUNT_ORACLE:{row["slot"]}')
        indices, counts = count_code_pair_label_indices(task)
        if len(indices) != 8 or len(counts) != 4 or any(count not in {1, 2, 3} for count in counts):
            fail(f'COUNT_CODE_PAIR_CLOSURE:{row["slot"]}')
        for index in indices:
            if sample['labels'][index] != sample['input_ids'][index]:
                fail(f'COUNT_CODE_PAIR_ALIGNMENT:{row["slot"]}')
        control, treatment = matched_weights(sample['labels'], indices, multiplier=2.0)
        if not (math.isclose(sum(control.values()), 60.0, rel_tol=0.0, abs_tol=1e-9) and math.isclose(sum(treatment.values()), 60.0, rel_tol=0.0, abs_tol=1e-9)):
            fail(f'LOSS_MASS:{row["slot"]}')
        if any(weight != 1.0 for idx, weight in treatment.items() if idx not in indices):
            if any(abs(weight - 11.0 / 13.0) > 1e-12 for idx, weight in treatment.items() if idx not in indices):
                fail(f'COMPENSATION:{row["slot"]}')
        focused_checks += len(indices)
    if focused_checks != 384:
        fail('COUNT_CODE_PAIR_TOTAL')
    return {'slots': len(episodes), 'inventory': observed, 'supervised_tokens': supervised, 'focused_count_code_tokens': focused_checks}


def validate_cpu_weighted_loss_path(binding: dict) -> dict:
    """Exercise the actual weighted equal-slot primitive on a real schedule label.

    This is still stop-before-model-load: CPU logits are synthetic solely to
    traverse the launcher's own label -> weight -> causal loss construction.
    """
    import torch
    from scripts.decision_token_supervision import count_code_pair_label_indices, matched_weights
    from scripts.e04_e_equal_slot_loss import backward_weighted_slot_term
    from training_data.pipeline import task_to_sample
    rows = json.loads((ROOT / binding['inputs']['diverse_schedule']['path']).read_text())['episodes']
    row = next(item for item in rows if item['role'] == 'PRIMARY_CONNECTED_COMPONENT_COUNT')
    sample = task_to_sample(serializer_task(row))
    indices, _ = count_code_pair_label_indices(row['task'])
    _, treatment = matched_weights(sample['labels'], indices, multiplier=2.0)
    weights = [0.0] * len(sample['labels'])
    for index, weight in treatment.items():
        weights[index] = weight
    torch.manual_seed(17)
    logits = torch.randn((1, len(sample['labels']), 16), dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([sample['labels']], dtype=torch.long)
    term, valid_count, mass = backward_weighted_slot_term(logits, labels, weights, expected_slots=4)
    if valid_count != 60 or abs(mass - 60.0) > 1e-6 or logits.grad is None or not bool(torch.isfinite(logits.grad).all().item()):
        fail('CPU_WEIGHTED_LOSS_PATH')
    return {'slot': row['slot'], 'focused_tokens': len(indices), 'valid_tokens': valid_count, 'weight_mass': mass, 'equal_slot_term_finite': bool(torch.isfinite(term).item())}


def validate_preclaim(binding_path: Path) -> dict:
    binding = json.loads(binding_path.read_text(encoding='utf-8'))
    validate_binding(binding)
    imports = validate_runtime_import_closure()
    schedule = validate_schedule(binding)
    weighted_loss = validate_cpu_weighted_loss_path(binding)
    return {
        'status': 'PASS_PRODUCTION_SHAPED_PRECLAIM_ONLY',
        'experiment_id': binding['experiment_id'],
        'scientific_source_commit': binding['scientific_source_commit'],
        'runtime_imports': imports,
        'schedule': schedule,
        'weighted_loss_path': weighted_loss,
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
    parser.add_argument('--cpu-integration', action='store_true')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.preclaim_only == args.cpu_integration:
        raise SystemExit('choose exactly one of --preclaim-only or --cpu-integration')
    if args.output_root is None:
        raise SystemExit('--output-root is required')
    # This is the real launcher-to-worker subprocess edge.  The worker repeats
    # the production preclaim, then stops before a model/CUDA import.
    worker = ROOT / 'scripts' / 'run_connected_component_count_code_pair_supervision_worker_v1.py'
    mode = '--preclaim-only' if args.preclaim_only else '--cpu-integration'
    completed = subprocess.run(
        [sys.executable, str(worker), '--binding', str(args.binding), '--output-root', str(args.output_root), mode],
        cwd=ROOT, env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT), str(SRC), os.environ.get('PYTHONPATH', '')])},
        check=False, text=True, capture_output=True,
    )
    if completed.returncode:
        raise SystemExit(completed.stderr or completed.stdout)
    if args.receipt:
        atomic_json(args.receipt, json.loads(completed.stdout))
    print(completed.stdout.strip())


if __name__ == '__main__':
    main()
