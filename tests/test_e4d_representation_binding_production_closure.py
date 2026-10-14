import json
from pathlib import Path
import pytest
from scripts.run_e4d_representation_binding_diagnostic_v1 import final_query_digit_positions, native_messages_for_task, records


def test_real_tokenizer_positions_cover_train_and_frozen_ood_without_targets():
    rows = records()
    assert len(rows) == 144
    assert sum(row['split'] == 'TRAIN_PROBE' for row in rows) == 48
    assert sum(row['split'] == 'OOD' for row in rows) == 96
    for row in rows:
        positions, shape = final_query_digit_positions(row['task'])
        assert len(positions) == shape[0] * shape[1]
        assert positions == sorted(positions)


def test_native_dict_transport_has_only_observable_train_and_query_input():
    task = records()[0]['task']
    messages = native_messages_for_task(task)
    assert [message['role'] for message in messages] == ['user', 'assistant'] * len(task['train']) + ['user']
    assert messages[-1]['content']


def test_binding_cpu_integration_three_arm_freeze_guard(tmp_path):
    from scripts.run_e4d_representation_binding_diagnostic_v1 import make_binding, cpu_integration
    binding = tmp_path / 'binding.json'
    make_binding(binding, tmp_path / 'fresh_output', '5fd601e6456dd81657f55422e69ef553', allow_dirty=True)
    result = cpu_integration(binding)
    assert result['status'] == 'PASS_E4D_REPRESENTATION_BINDING_REAL_TOKENIZER_CPU_INTEGRATION'
    assert result['raw_freeze_validated_before_sidecar'] is True
    assert result['raw_rehash_status'] == 'PASS'
    assert result['sidecar_opened'] is False and result['targets_opened'] is False
    assert result['model_loaded'] is False and result['gpu_used'] is False


def _raw_rows():
    output = []
    for row in records():
        output.append({
            'record_id': row['record_id'], 'split': row['split'], 'count': row.get('count'),
            'prompt_sha256': row['prompt_sha256'],
            'layer_features': {str(layer): [0.0, 1.0] for layer in (8, 16, 24, 31)},
            'native_count': None, 'generated_token_ids': [], 'parser_valid': False,
        })
    return output


def _write_raws(tmp_path, rows):
    from scripts.run_e4d_representation_binding_diagnostic_v1 import write_jsonl
    paths = {}
    for arm in ('V7', 'EQUAL_SLOT', 'COMPONENT_ANCHOR'):
        paths[arm] = tmp_path / f'{arm}_RAW.jsonl'
        write_jsonl(paths[arm], rows)
    return paths


@pytest.mark.parametrize('mutator', [
    lambda rows: rows[:-1],
    lambda rows: rows + [rows[0]],
    lambda rows: [{**row, 'layer_features': {'31': [0.0, 1.0]}} if index == 0 else row for index, row in enumerate(rows)],
])
def test_raw_freeze_rejects_omission_duplicate_and_schema(mutator, tmp_path):
    from scripts.run_e4d_representation_binding_diagnostic_v1 import raw_freeze
    with pytest.raises(RuntimeError):
        raw_freeze(_write_raws(tmp_path, mutator(_raw_rows())), tmp_path)


def test_sidecar_guard_rehashes_and_rejects_post_freeze_mutation(tmp_path):
    from scripts.run_e4d_representation_binding_diagnostic_v1 import raw_freeze, sidecar_allowed
    from scripts.score_e4d_representation_binding_diagnostic_v1 import rehash_raw_freeze
    paths = _write_raws(tmp_path, _raw_rows())
    raw_freeze(paths, tmp_path)
    with paths['V7'].open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'mutated': True}) + '\n')
    with pytest.raises(RuntimeError, match='RAW_POST_FREEZE_MUTATION'):
        sidecar_allowed(tmp_path)
    with pytest.raises(ValueError, match='RAW_POST_FREEZE_MUTATION'):
        rehash_raw_freeze(paths)
