from types import SimpleNamespace

import pytest

from scripts.run_eval60_ready_restore_v1 import canary_mechanically_valid, capacity, complete, task_groups


def row(task, index, identity='same'):
    return {'task_id': task, 'output_index': index, 'output_id': f'{task}:o{index}',
            'adapter_identity': {'adapter_sha256': identity, 'adapter_config_sha256': 'config'}}


def test_task_grouping_never_mixes_adapters_or_adds_views():
    groups = task_groups([row('b', 0), row('a', 1), row('a', 0)])
    assert [[r['output_id'] for r in g] for g in groups] == [['a:o0', 'a:o1'], ['b:o0']]
    with pytest.raises(RuntimeError, match='different adapter'):
        task_groups([row('a', 0), row('a', 1, 'other')])


@pytest.mark.parametrize('root,lanes,expected', [
    (1000, 16, (16, 16)), (1000, 8, (8, 8)), (2400, 16, (8, 8)),
    (4000, 16, (4, 4)), (8400, 8, (2, 2)),
])
def test_historical_capacity_respects_context_and_available_logical_work(root, lanes, expected):
    assert capacity(root, lanes) == expected


def test_completed_output_cannot_be_reused_with_missing_artifacts(tmp_path):
    import json
    path = tmp_path / 'OUTPUT_RECEIPTS' / 'a_o0.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'status': 'COMPLETE', 'output_id': 'a:o0',
                               'raw_output': 'missing', 'checkpoint_output': 'missing', 'eos_events': 'missing'}))
    with pytest.raises(RuntimeError, match='missing artifacts'):
        complete(tmp_path, row('a', 0))


def test_inherited_wall_censoring_is_reportable_without_extending_scientific_budget():
    cells = [{'nodes_expanded': 500, 'completed_candidates': 2, 'termination_reason': 'wall_time',
              'prefix_reconstruction_failures': [], 'diagnostic_trace': {
                  'prefill': {'cache_type': 'inference.chunked_kv_cache.ChunkedDynamicCache'},
                  'branch_probabilities': [{'full_arc_logprobs': [{'logprob': -0.5}]}],
              }} for _ in range(8)]
    assert canary_mechanically_valid(cells)
    cells[0]['nodes_expanded'] = 1025
    assert not canary_mechanically_valid(cells)
    cells[0]['nodes_expanded'] = 500
    cells[0]['prefix_reconstruction_failures'] = ['bad_ancestry']
    assert not canary_mechanically_valid(cells)
