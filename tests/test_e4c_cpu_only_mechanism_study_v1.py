from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_e4c_cpu_only_mechanism_study_v1 import (
    load_raw,
    query_local_code_pair_indices,
    query_local_code_pair_weights,
    run,
)
from scripts.e4b_corrected_code_pair_supervision import count_code_pair_label_indices

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1'
SCHEDULE = BASE / 'TREATMENT_SCHEDULE.json'
E4A = ROOT / 'evidence/e4a_matched_lr_20261013T145829Z_verified'
E4B = ROOT / 'evidence/e4b_20261013T153226Z_e5405455'


def first_count_task():
    rows = json.loads(SCHEDULE.read_text())['episodes']
    return next(row['task'] for row in rows if row['role'] == 'PRIMARY_CONNECTED_COMPONENT_COUNT')


def test_query_local_selector_focuses_only_final_query_code_pair_and_preserves_scalar_mass():
    task = first_count_task()
    full, _ = count_code_pair_label_indices(task)
    query = query_local_code_pair_indices(task)
    assert query == full[-2:]
    assert set(query).isdisjoint(full[:-2])
    labels = [-100] * (max(full) + 2)
    for index in full:
        labels[index] = 7
    control, treatment = query_local_code_pair_weights(labels, task)
    assert abs(sum(control.values()) - sum(treatment.values())) < 1e-12
    assert all(treatment[index] == 2.0 for index in query)
    assert all(treatment[index] < 1.0 for index in full[:-2])


def test_target_sidecars_are_rejected_by_raw_loader():
    with pytest.raises(ValueError, match='TARGET_REUSE_FORBIDDEN'):
        load_raw(Path('SYNTHETIC_TARGET_SIDECAR.json'))


def test_production_shaped_frozen_evidence_audit_has_no_model_or_target_access(tmp_path):
    result = run(E4A, E4B, SCHEDULE, tmp_path)
    assert result['status'] == 'PASS_CPU_ONLY_NO_MODEL_NO_TARGET_REUSE'
    assert result['controlled_intervention_cpu_validation']['count_primary_slots'] == 48
    assert result['controlled_intervention_cpu_validation']['focused_tokens_per_slot'] == 2
    assert result['controlled_intervention_cpu_validation']['model_loaded'] is False
    raw = result['raw_prediction_audit']['code_pair_vs_equal_slot']
    assert raw['TRAIN_FIT_SECONDARY']['parsed_grid_changed'] == 10
    assert raw['OOD_INDEPENDENT']['parsed_grid_changed'] == 15
    assert raw['RETENTION_SECONDARY:MARKER_BINDING']['parsed_grid_changed'] == 2
    assert raw['RETENTION_SECONDARY:FIXED_TURN_ROTATION']['parsed_grid_changed'] == 2
    assert (tmp_path / 'FROZEN_PREDICTION_MECHANISM_AUDIT.json').is_file()
    receipt = json.loads((tmp_path / 'CPU_VALIDATION_RECEIPT.json').read_text())
    assert receipt['model_loaded'] is False
    assert receipt['gpu_used'] is False
    assert receipt['target_sidecar_opened'] is False