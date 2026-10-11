"""Freeze the CPU-only E04-E V2 equal-slot marker replay pilot.

No torch, transformers, optimizer, model, evaluation target, or GPU imports occur
in this module. It derives all supervised-token and sequence-token accounting from
the repository's frozen native training serializer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(ROOT))
if str(ROOT / 'src') not in __import__('sys').path:
    __import__('sys').path.insert(0, str(ROOT / 'src'))

from training_data.pipeline import IGNORE_INDEX, task_to_sample

E04C = ROOT / 'experiments/capability_repair_baseline_v1/e04_c_matched_fixed_turn_rotation_repair_pilot_v1'
E04D = ROOT / 'experiments/capability_repair_baseline_v1/e04_d_rotation_marker_local_gradient_diagnostic_v1'
OUT = ROOT / 'experiments/capability_repair_baseline_v1/e04_e_equal_slot_weight_marker_replay_preservation_pilot_v2'
PROTOCOL = 'E04_E_EQUAL_SLOT_WEIGHT_MARKER_REPLAY_PRESERVATION_PILOT_V2'
SEED = 'E04_E_EQUAL_SLOT_WEIGHT_MARKER_REPLAY_PRESERVATION_PILOT_V2:selection:20261011'
SLOTS, ROTATION_SLOTS, FIXED_REPLAY_SLOTS, SWAP_SLOTS, STEP_SIZE = 384, 96, 192, 96, 4
DIRECTOR_RESPONSE = ROOT / 'orchestration/director/responses/E04_E_MATCHED_MARKER_REPLAY_PRESERVATION_PILOT_V1_CPU_FEASIBILITY_BRIEF_RESPONSE.json'
E04D_RESULT_RESPONSE = ROOT / 'orchestration/director/responses/E04_D_ROTATION_MARKER_LOCAL_GRADIENT_DIAGNOSTIC_V1_RUN_002_RESULT_BRIEF_RESPONSE.json'

class FreezeFailure(RuntimeError):
    pass

def canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def task_sha(task: dict[str, Any]) -> str:
    return hashlib.sha256(canon(task)).hexdigest()

def rank(namespace: str, value: str) -> str:
    return hashlib.sha256((SEED + ':' + namespace + ':' + value).encode()).hexdigest()

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def sample_counts(task: dict[str, Any], source_id: str) -> dict[str, int]:
    sample = task_to_sample({'source_id': source_id, **task})
    if len(sample['input_ids']) != len(sample['labels']) or sample['attention_mask'] != [1] * len(sample['input_ids']):
        raise FreezeFailure('E04E_SERIALIZER_SHAPE')
    supervised = sum(x != IGNORE_INDEX for x in sample['labels'])
    if supervised != sample['assistant_token_count'] or supervised < 1:
        raise FreezeFailure('E04E_SERIALIZER_LABELS')
    return {'transformer_tokens': len(sample['input_ids']), 'supervised_tokens': supervised}

def _validate_director_response() -> None:
    response = read_json(DIRECTOR_RESPONSE)
    if response.get('decision') != 'CONTINUE_CONTROLLER' or response.get('next_stage') != 'CPU_ONLY_FREEZE_E04_E_V2_EQUAL_SLOT_WEIGHT_MARKER_REPLAY_PRESERVATION_PILOT':
        raise FreezeFailure('E04E_DIRECTOR_RESPONSE')
    if response.get('gpu_authorized_for_next_stage') or response.get('model_loading_authorized_for_next_stage') or response.get('scientific_training_authorized_for_next_stage'):
        raise FreezeFailure('E04E_CPU_ONLY_BOUNDARY')
    if sha_file(E04D_RESULT_RESPONSE) != '702bb188c16e01858db743f65ec4f8d987cd9b7a16914f22e79995981669ca69':
        raise FreezeFailure('E04E_E04D_RESULT_RESPONSE_IDENTITY')

def _base_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    c_manifest = read_json(E04C / 'MANIFEST.json')
    d_manifest = read_json(E04D / 'MANIFEST.json')
    if c_manifest.get('training_split_only') is not True or c_manifest.get('validation_rows_read') != 0:
        raise FreezeFailure('E04E_E04C_TRAIN_ONLY_PROVENANCE')
    if d_manifest.get('validation_rows_read') != 0 or d_manifest.get('protocol_id') != 'E04_D_ROTATION_MARKER_LOCAL_GRADIENT_DIAGNOSTIC_V1':
        raise FreezeFailure('E04E_E04D_TRAIN_ONLY_PROVENANCE')
    c_schedule = read_json(E04C / 'schedule_freeze_v1' / 'CONTROL_SCHEDULE.json')
    t_schedule = read_json(E04C / 'schedule_freeze_v1' / 'TREATMENT_SCHEDULE.json')
    control, treatment = c_schedule.get('episodes'), t_schedule.get('episodes')
    marker = read_jsonl(E04D / 'FROZEN_TRAIN_COHORT.jsonl')
    if not isinstance(control, list) or not isinstance(treatment, list) or len(control) != SLOTS or len(treatment) != SLOTS:
        raise FreezeFailure('E04E_E04C_SCHEDULE_ROWS')
    if len(marker) != 48 or len({row['pair_id'] for row in marker}) != 48:
        raise FreezeFailure('E04E_MARKER_COHORT_ROWS')
    for slot, (c, t) in enumerate(zip(control, treatment, strict=True)):
        if c.get('slot') != slot or t.get('slot') != slot or c.get('pair_id') != t.get('pair_id'):
            raise FreezeFailure('E04E_E04C_ALIGNMENT')
        if slot < ROTATION_SLOTS:
            if t.get('role') != 'E04_C_INTERVENTION' or c.get('control_task') == t.get('treatment_task'):
                raise FreezeFailure('E04E_ROTATION_SOURCE')
        elif c.get('role') != 'PROTECTED_REPLAY' or c.get('control_task') != t.get('treatment_task'):
            raise FreezeFailure('E04E_REPLAY_SOURCE')
    return control, treatment, marker

def _slot_metadata(slot: int, role: str, source: str, canonical_base_id: str, turn: int, train_cell: dict[str, Any]) -> dict[str, Any]:
    return {'slot': slot, 'optimizer_step': slot // STEP_SIZE, 'within_step_role_position': slot % STEP_SIZE,
            'role': role, 'replay_source': source, 'canonical_base_id': canonical_base_id,
            'turn': turn, 'train_cell': train_cell}

def build_schedules() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _validate_director_response()
    original_control, original_treatment, marker_rows = _base_rows()
    fixed_candidates = list(range(ROTATION_SLOTS, SLOTS))
    fixed = set(sorted(fixed_candidates, key=lambda slot: rank('fixed-replay', original_control[slot]['pair_id']))[:FIXED_REPLAY_SLOTS])
    swap_positions = [slot for slot in fixed_candidates if slot not in fixed]
    if len(fixed) != FIXED_REPLAY_SLOTS or len(swap_positions) != SWAP_SLOTS:
        raise FreezeFailure('E04E_REPLAY_ALLOCATION')
    ordered_marker = sorted(marker_rows, key=lambda row: rank('marker', row['pair_id']))
    marker_occurrences = [row for row in ordered_marker for _ in range(2)]
    if len(marker_occurrences) != SWAP_SLOTS:
        raise FreezeFailure('E04E_MARKER_DUPLICATION')
    control_out: list[dict[str, Any]] = []
    treatment_out: list[dict[str, Any]] = []
    marker_by_slot = dict(zip(sorted(swap_positions), marker_occurrences, strict=True))
    for slot in range(SLOTS):
        if slot < ROTATION_SLOTS:
            source = original_treatment[slot]
            meta = _slot_metadata(slot, 'FIXED_TURN_ROTATION', 'E04_C_BYTE_IDENTICAL_ROTATION', source['canonical_base_id'], source['turn'], source['train_cell'])
            task = source['treatment_task']
            row = {**meta, 'source_pair_id': source['pair_id'], 'task_kind': 'FIXED_TURN_ROTATION_CONTROL', 'task': task, 'task_sha256': task_sha(task)}
            control_out.append(row); treatment_out.append(dict(row))
        elif slot in fixed:
            source = original_control[slot]
            meta = _slot_metadata(slot, 'PROTECTED_REPLAY', 'E04_C_BYTE_IDENTICAL_NO_TRANSFORM', source['canonical_base_id'], source['turn'], source['train_cell'])
            task = source['control_task']
            row = {**meta, 'source_pair_id': source['pair_id'], 'task_kind': 'NO_TRANSFORM_RETENTION_CONTROL', 'task': task, 'task_sha256': task_sha(task)}
            control_out.append(row); treatment_out.append(dict(row))
        else:
            marker = marker_by_slot[slot]
            ctask, ttask = marker['tasks']['NO_TRANSFORM_RETENTION_CONTROL'], marker['tasks']['MARKER_BINDING_CONTROL']
            meta = _slot_metadata(slot, 'PROTECTED_REPLAY', 'E04_D_MATCHED_MARKER_REPLAY_SWAP', marker['canonical_base_id'], marker['turn'], marker['train_cell'])
            common = {**meta, 'marker_pair_id': marker['pair_id'], 'matched_on': ['canonical_base_id', 'turn', 'train_cell', 'optimizer_step', 'within_step_role_position']}
            control_out.append({**common, 'task_kind': 'NO_TRANSFORM_RETENTION_CONTROL', 'task': ctask, 'task_sha256': task_sha(ctask)})
            treatment_out.append({**common, 'task_kind': 'MARKER_BINDING_CONTROL', 'task': ttask, 'task_sha256': task_sha(ttask)})
    verify_schedules(control_out, treatment_out)
    return control_out, treatment_out, {'fixed_replay_slots': sorted(fixed), 'swap_slots': sorted(swap_positions)}

def verify_schedules(control: list[dict[str, Any]], treatment: list[dict[str, Any]]) -> None:
    if len(control) != SLOTS or len(treatment) != SLOTS:
        raise FreezeFailure('E04E_SLOT_COUNT')
    marker_counts = Counter()
    for slot, (c, t) in enumerate(zip(control, treatment, strict=True)):
        if c['slot'] != slot or t['slot'] != slot or c['optimizer_step'] != slot // STEP_SIZE or t['optimizer_step'] != slot // STEP_SIZE or c['within_step_role_position'] != slot % STEP_SIZE or t['within_step_role_position'] != slot % STEP_SIZE:
            raise FreezeFailure('E04E_STEP_ROLE_POSITION')
        if any(c[key] != t[key] for key in ('role', 'replay_source', 'canonical_base_id', 'turn', 'train_cell')):
            raise FreezeFailure('E04E_CROSS_ARM_METADATA')
        if slot < ROTATION_SLOTS:
            if c['task'] != t['task'] or c['task_kind'] != 'FIXED_TURN_ROTATION_CONTROL' or t['task_kind'] != 'FIXED_TURN_ROTATION_CONTROL':
                raise FreezeFailure('E04E_ROTATION_NOT_IDENTICAL')
        elif c['replay_source'] == 'E04_C_BYTE_IDENTICAL_NO_TRANSFORM':
            if c['task'] != t['task'] or c['task_kind'] != 'NO_TRANSFORM_RETENTION_CONTROL' or t['task_kind'] != 'NO_TRANSFORM_RETENTION_CONTROL':
                raise FreezeFailure('E04E_FIXED_REPLAY_NOT_IDENTICAL')
        else:
            if c['task'] == t['task'] or c['task_kind'] != 'NO_TRANSFORM_RETENTION_CONTROL' or t['task_kind'] != 'MARKER_BINDING_CONTROL':
                raise FreezeFailure('E04E_MARKER_SWAP_CONTENT')
            if c.get('marker_pair_id') != t.get('marker_pair_id') or c.get('matched_on') != t.get('matched_on'):
                raise FreezeFailure('E04E_MARKER_MATCH_METADATA')
            marker_counts[c['marker_pair_id']] += 1
    if Counter(row['role'] for row in control) != Counter({'FIXED_TURN_ROTATION': 96, 'PROTECTED_REPLAY': 288}):
        raise FreezeFailure('E04E_ROLE_COUNTS')
    if Counter(row['replay_source'] for row in control) != Counter({'E04_C_BYTE_IDENTICAL_ROTATION': 96, 'E04_C_BYTE_IDENTICAL_NO_TRANSFORM': 192, 'E04_D_MATCHED_MARKER_REPLAY_SWAP': 96}):
        raise FreezeFailure('E04E_SOURCE_COUNTS')
    if len(marker_counts) != 48 or set(marker_counts.values()) != {2}:
        raise FreezeFailure('E04E_MARKER_EXACTLY_TWICE')

def accounting(control: list[dict[str, Any]], treatment: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {'serializer': 'training_data.pipeline.task_to_sample', 'model_loaded': False, 'tokenizer_loaded': False, 'transformers_imported': False, 'per_slot': [], 'arms': {}}
    arm_totals = {}
    for arm, rows in (('CONTROL', control), ('TREATMENT', treatment)):
        records = []
        for row in rows:
            counts = sample_counts(row['task'], f'e04e-{arm.lower()}-{row["slot"]}')
            records.append({**counts, 'slot': row['slot'], 'optimizer_step': row['optimizer_step'], 'within_step_role_position': row['within_step_role_position'], 'task_kind': row['task_kind'], 'replay_source': row['replay_source'], 'task_sha256': row['task_sha256']})
        arm_totals[arm] = {'raw_transformer_tokens': sum(x['transformer_tokens'] for x in records), 'raw_supervised_tokens': sum(x['supervised_tokens'] for x in records), 'slot_count': len(records), 'optimizer_steps': SLOTS // STEP_SIZE, 'per_step_slot_coefficients': [0.25, 0.25, 0.25, 0.25]}
        result['arms'][arm] = arm_totals[arm]
        result['per_slot'].append({'arm': arm, 'rows': records})
    # Equal-slot loss rules are structural and independent of raw token totals.
    for arm in ('CONTROL', 'TREATMENT'):
        rows = result['per_slot'][0 if arm == 'CONTROL' else 1]['rows']
        for step in range(SLOTS // STEP_SIZE):
            if len(rows[step * STEP_SIZE:(step + 1) * STEP_SIZE]) != STEP_SIZE:
                raise FreezeFailure('E04E_STEP_SIZE')
    result['raw_token_difference_treatment_minus_control'] = {
        'transformer_tokens': arm_totals['TREATMENT']['raw_transformer_tokens'] - arm_totals['CONTROL']['raw_transformer_tokens'],
        'supervised_tokens': arm_totals['TREATMENT']['raw_supervised_tokens'] - arm_totals['CONTROL']['raw_supervised_tokens'],
    }
    result['interpretation'] = 'Raw totals are reported structural differences only. Both arms use the same per-slot mean supervised-token CE and coefficient 0.25 per slot; raw token totals do not define scalar-loss allocation or update budget.'
    return result

def protocol(control_path: Path, treatment_path: Path, account_path: Path, schedule_meta: dict[str, Any]) -> dict[str, Any]:
    return {
      'schema_version': 1, 'protocol_id': PROTOCOL, 'status': 'CPU_ONLY_FROZEN_PENDING_DIRECTOR_PRELAUNCH_REVIEW',
      'scientific_basis': {'director_response_path': DIRECTOR_RESPONSE.relative_to(ROOT).as_posix(), 'director_response_sha256': sha_file(DIRECTOR_RESPONSE), 'e04d_result_response_path': E04D_RESULT_RESPONSE.relative_to(ROOT).as_posix(), 'e04d_result_response_sha256': sha_file(E04D_RESULT_RESPONSE)},
      'source_provenance': {'e04c_control_schedule_path': (E04C / 'schedule_freeze_v1' / 'CONTROL_SCHEDULE.json').relative_to(ROOT).as_posix(), 'e04c_control_schedule_sha256': sha_file(E04C / 'schedule_freeze_v1' / 'CONTROL_SCHEDULE.json'), 'e04c_treatment_schedule_path': (E04C / 'schedule_freeze_v1' / 'TREATMENT_SCHEDULE.json').relative_to(ROOT).as_posix(), 'e04c_treatment_schedule_sha256': sha_file(E04C / 'schedule_freeze_v1' / 'TREATMENT_SCHEDULE.json'), 'e04c_manifest_sha256': sha_file(E04C / 'MANIFEST.json'), 'e04d_cohort_path': (E04D / 'FROZEN_TRAIN_COHORT.jsonl').relative_to(ROOT).as_posix(), 'e04d_cohort_sha256': sha_file(E04D / 'FROZEN_TRAIN_COHORT.jsonl'), 'e04d_manifest_sha256': sha_file(E04D / 'MANIFEST.json')},
      'checkpoint': {'identity': 'CAPABILITY_REPAIR_BASELINE_V1_V7', 'manifest_path': 'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json', 'manifest_sha256': '1e124cc4f43530bbbc71103703d404b38df83a6998a3e39d7c1da0676c800549', 'independent_fresh_initialization_per_arm': True},
      'arms': {'control': 'identical fixed-turn rotation plus no-transform replay at 192 preserved and 96 matched swap positions', 'treatment': 'identical fixed-turn rotation plus no-transform replay at 192 preserved and marker-binding replay at the same 96 matched swap positions', 'single_intentional_factor': 'equal-weight marker-binding versus no-transform replay at 96 fixed positions'},
      'schedule': {'control_path': control_path.relative_to(ROOT).as_posix(), 'control_sha256': sha_file(control_path), 'treatment_path': treatment_path.relative_to(ROOT).as_posix(), 'treatment_sha256': sha_file(treatment_path), 'accounting_path': account_path.relative_to(ROOT).as_posix(), 'accounting_sha256': sha_file(account_path), 'selection_seed': SEED, 'slots': SLOTS, 'optimizer_steps': SLOTS // STEP_SIZE, 'slots_per_step': STEP_SIZE, 'rotation_slots': ROTATION_SLOTS, 'byte_identical_no_transform_replay_slots': FIXED_REPLAY_SLOTS, 'matched_marker_swap_slots': SWAP_SLOTS, **schedule_meta},
      'objective': {'name': 'COMMON_EQUAL_SLOT_MEAN_SUPERVISED_TOKEN_CE', 'definition': 'At each optimizer step, compute mean supervised-token CE independently for each of four slots; sum 0.25 times each slot mean before backward.', 'common_to_both_arms': True, 'raw_token_totals_not_loss_weights': True, 'forbidden': ['token-weighted accumulation', 'masking or truncating valid target tokens', 'synthetic filler tokens']},
      'recipe': {'optimizer': 'PagedAdamW8bit', 'learning_rate': 0.00005, 'lora_rank': 64, 'lora_alpha': 32, 'lora_dropout': 0.0, 'target_modules': ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], 'precision': 'BF16', 'quantization': 'NONE', 'seed': 2000031, 'optimizer_steps_per_arm': 96, 'arm_order_policy': ['CONTROL_EQUAL_SLOT', 'TREATMENT_EQUAL_SLOT'], 'per_arm_runtime_cap_seconds': 3600, 'joint_runtime_cap_seconds': 7200},
      'evaluation': {'primary_estimand': 'treatment minus control marker-binding native-Greedy exact-grid accuracy, paired by canonical base', 'primary_success': 'at least 8 additional correct marker-binding episodes of 192 and paired bootstrap 95% interval lower bound > 0', 'absolute_marker_protection': 'treatment marker-binding decline vs V7 no more than 8 of 192', 'rotation_efficacy': 'treatment fixed-turn rotation improves at least 8 of 192 vs V7 with CI lower bound > 0 and does not trail matched control by more than 8 of 192', 'no_transform_retention': 'treatment no-transform accuracy decline vs V7 no more than 8 of 192', 'bootstrap': {'seed': 20261011, 'replicates': 10000, 'unit': 'canonical base with all four turns paired'}, 'scope': 'Reused E04 development cohort only; not independent ARC generalization confirmation.'},
      'boundaries': ['TRAIN_ONLY_REPLAY_SELECTION', 'NO_MODEL_OR_TOKENIZER_LOADING_DURING_FREEZE', 'NO_OPTIMIZER_CONSTRUCTION', 'NO_GPU', 'NO_GENERATION', 'NO_EVALUATION_TARGET_ACCESS', 'NO_GOLD', 'NO_DGOLD', 'NO_FINAL_AUDIT'],
      'compute_request': {'requested_after_separate_director_prelaunch_review_only': True, 'two_independent_v7_arms': 2, 'maximum_joint_gpu_seconds': 7200, 'maximum_per_arm_gpu_seconds': 3600, 'no_automatic_launch': True}
    }

def freeze(out: Path) -> dict[str, Any]:
    if out.exists():
        raise FreezeFailure('E04E_REFUSE_OVERWRITE')
    control, treatment, schedule_meta = build_schedules()
    out.mkdir(parents=True)
    cp, tp, ap = out / 'CONTROL_SCHEDULE.json', out / 'TREATMENT_SCHEDULE.json', out / 'TOKEN_ACCOUNTING.json'
    atomic_json(cp, {'protocol_id': PROTOCOL, 'arm': 'CONTROL_EQUAL_SLOT', 'episodes': control})
    atomic_json(tp, {'protocol_id': PROTOCOL, 'arm': 'TREATMENT_EQUAL_SLOT', 'episodes': treatment})
    atomic_json(ap, accounting(control, treatment))
    pp = out / 'E04_E_V2_PRELAUNCH_PROTOCOL.json'
    atomic_json(pp, protocol(cp, tp, ap, schedule_meta))
    manifest = {'protocol_id': PROTOCOL, 'status': 'CPU_ONLY_FROZEN_PENDING_DIRECTOR_PRELAUNCH_REVIEW', 'control_schedule_sha256': sha_file(cp), 'treatment_schedule_sha256': sha_file(tp), 'token_accounting_sha256': sha_file(ap), 'protocol_sha256': sha_file(pp), 'model_loaded': False, 'tokenizer_loaded': False, 'gpu_used': False, 'optimizer_constructed': False, 'training_started': False, 'generation_started': False, 'evaluation_targets_accessed': False, 'final_audit_opened': False}
    atomic_json(out / 'MANIFEST.json', manifest)
    return manifest

def validate_frozen(out: Path) -> dict[str, Any]:
    """Recheck the published V2 CPU artifacts without model-related imports."""
    control_doc, treatment_doc = read_json(out / 'CONTROL_SCHEDULE.json'), read_json(out / 'TREATMENT_SCHEDULE.json')
    control, treatment = control_doc.get('episodes'), treatment_doc.get('episodes')
    if not isinstance(control, list) or not isinstance(treatment, list):
        raise FreezeFailure('E04E_PUBLISHED_SCHEDULE_SCHEMA')
    verify_schedules(control, treatment)
    observed = accounting(control, treatment)
    published = read_json(out / 'TOKEN_ACCOUNTING.json')
    if canon(observed) != canon(published):
        raise FreezeFailure('E04E_PUBLISHED_ACCOUNTING_DRIFT')
    proto = read_json(out / 'E04_E_V2_PRELAUNCH_PROTOCOL.json')
    if proto.get('protocol_id') != PROTOCOL or proto.get('schedule', {}).get('control_sha256') != sha_file(out / 'CONTROL_SCHEDULE.json') or proto.get('schedule', {}).get('treatment_sha256') != sha_file(out / 'TREATMENT_SCHEDULE.json'):
        raise FreezeFailure('E04E_PUBLISHED_PROTOCOL_BINDING')
    return {'status': 'PASS_CPU_ONLY_V2_SCHEDULE_AND_ACCOUNTING_VALIDATION', 'protocol_id': PROTOCOL,
            'control_schedule_sha256': sha_file(out / 'CONTROL_SCHEDULE.json'), 'treatment_schedule_sha256': sha_file(out / 'TREATMENT_SCHEDULE.json'),
            'token_accounting_sha256': sha_file(out / 'TOKEN_ACCOUNTING.json'), 'protocol_sha256': sha_file(out / 'E04_E_V2_PRELAUNCH_PROTOCOL.json'),
            'model_loaded': False, 'tokenizer_loaded': False, 'gpu_used': False, 'optimizer_constructed': False,
            'evaluation_targets_accessed': False, 'final_audit_opened': False}

def self_test() -> None:
    control, treatment, _ = build_schedules(); verify_schedules(control, treatment); acc = accounting(control, treatment)
    if acc['arms']['CONTROL']['slot_count'] != 384 or acc['arms']['TREATMENT']['slot_count'] != 384:
        raise FreezeFailure('E04E_SELF_TEST_ACCOUNTING')
    bad = json.loads(json.dumps(treatment)); bad[100]['role'] = 'WRONG'
    try: verify_schedules(control, bad)
    except FreezeFailure as exc:
        if str(exc) != 'E04E_CROSS_ARM_METADATA': raise
    else: raise FreezeFailure('E04E_SELF_TEST_REJECTS_MUTATION')

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=OUT); p.add_argument('--self-test', action='store_true'); p.add_argument('--validate', action='store_true'); args = p.parse_args()
    if args.self_test:
        self_test(); print('PASS_E04E_V2_CPU_FREEZER')
    elif args.validate:
        print(json.dumps(validate_frozen(args.out), sort_keys=True))
    else:
        print(json.dumps(freeze(args.out), sort_keys=True))
if __name__ == '__main__': main()
