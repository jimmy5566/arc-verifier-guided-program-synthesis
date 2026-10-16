import gzip
import json
import pytest
from scripts.score_reused_rankings_cpu import b_top2, metrics, terminal_scores
from scripts.score_phase3_eval60_union_v1 import AUG8


def candidate(grid, support, nll, index):
    return {'canonical_grid': [[grid]], 'grid_key': str(grid), 'support_count': support,
            'mean_view_nll': nll, 'stable_representative': {'augmentation_id': AUG8[0],
            'candidate_completion_index': index, 'nodes_expanded_so_far': 10, 'candidate_id': str(index)}}


def test_cached_b_uses_frozen_support_minus_mean_nll():
    payload = {'source_candidates': [candidate(1, 2, 0.1, 1), candidate(2, 3, 0.2, 2), candidate(3, 1, 0.1, 3)]}
    assert b_top2(payload) == [[[2]], [[1]]]


def test_partial_outputs_not_counted_as_full_cohort_or_full_tasks():
    gold = {'t:o0': [[1]], 't:o1': [[2]], 'u:o0': [[3]]}
    result = metrics({'t:o0': [[[1]]], 'u:o0': [[[3]]]}, gold)
    assert result['outputs_evaluated'] == 2 and result['Top2'] == 2
    assert result['fully_observed_tasks_in_panel'] == 1
    assert result['task_all_outputs_Top2'] == 1


def test_empty_frozen_candidate_pool_is_known_failure_not_missing_scoring():
    assert metrics({'t:o0': []}, {'t:o0': [[1]]})['Top2'] == 0


def test_terminal_stream_matches_view_and_id_not_only_id():
    rows = [{'augmentation_id': view, 'node': {'node_id': node, 'cumulative_score': value}}
            for view, node, value in [('a', 1, 2.0), ('b', 1, 3.0), ('a', 2, 4.0)]]
    blob = gzip.compress(b'\n'.join(json.dumps(row).encode() for row in rows))
    assert terminal_scores(blob, {('b', 1), ('a', 2)}) == {('b', 1): 3.0, ('a', 2): 4.0}


def test_terminal_stream_rejects_conflicting_duplicate():
    rows = [{'augmentation_id': 'a', 'node': {'node_id': 3, 'cumulative_score': value}}
            for value in (1.0, 2.0)]
    blob = gzip.compress(b'\n'.join(json.dumps(row).encode() for row in rows))
    with pytest.raises(RuntimeError, match='CONFLICTING_TERMINAL_SCORE'):
        terminal_scores(blob, {('a', 3)})


def test_terminal_stream_retains_missing_score_as_missing():
    assert terminal_scores(gzip.compress(b'{"augmentation_id":"a","node":{"node_id":1}}\n'),
                           {('a', 2)}) == {}


def test_cpu_wrapper_and_scorer_do_not_import_gpu_or_model_packages():
    import ast
    from pathlib import Path
    for name in ('score_reused_rankings_cpu.py', 'run_cpu_selector_completion.py'):
        tree = ast.parse((Path(__file__).parents[1] / 'scripts' / name).read_text())
        modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not any(module and module.split('.')[0] in {'torch', 'transformers', 'peft'} for module in modules)
