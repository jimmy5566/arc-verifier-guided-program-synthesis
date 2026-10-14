import json
from pathlib import Path
from scripts.run_e4d_representation_binding_diagnostic_v1 import final_query_digit_positions, records


def test_real_tokenizer_positions_cover_train_and_frozen_ood_without_targets():
    rows = records()
    assert len(rows) == 144
    assert sum(row['split'] == 'TRAIN_PROBE' for row in rows) == 48
    assert sum(row['split'] == 'OOD' for row in rows) == 96
    for row in rows:
        positions, shape = final_query_digit_positions(row['task'])
        assert len(positions) == shape[0] * shape[1]
        assert positions == sorted(positions)


def test_binding_cpu_integration_three_arm_freeze_guard(tmp_path):
    from scripts.run_e4d_representation_binding_diagnostic_v1 import make_binding, cpu_integration
    binding = tmp_path / 'binding.json'
    make_binding(binding, tmp_path / 'fresh_output', '5fd601e6456dd81657f55422e69ef553', allow_dirty=True)
    result = cpu_integration(binding)
    assert result['status'] == 'PASS_E4D_REPRESENTATION_BINDING_REAL_TOKENIZER_CPU_INTEGRATION'
    assert result['raw_freeze_before_sidecar'] is True
    assert result['score_status'] == 'COMPLETE_AFTER_RAW_FREEZE'
    assert result['sidecar_opened_after_raw_freeze'] is True
    assert result['model_loaded'] is False and result['gpu_used'] is False
