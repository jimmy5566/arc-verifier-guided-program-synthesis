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


def test_source_rankings_checkpoint_resume_and_stale_identity(tmp_path, monkeypatch):
    from scripts import score_reused_rankings_cpu as module
    from scripts.score_v7_frozen_oracle_only import write, digest
    root = tmp_path / 'archive'
    for sub in ('manifests', 'tables/node_index', 'tables/candidate_index'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    hashes = {}
    def asset(name, data):
        raw = data if isinstance(data, bytes) else json.dumps(data).encode()
        (root / name).write_bytes(raw)
        hashes[name] = digest(raw)
    asset('CONTRACT.json', {'ttt_depth': 24, 'gold_loaded': False})
    asset('SOURCE_COMPLETENESS_AUDIT.json', {'status': 'PASS'})
    ids = [str(n) + ':o0' for n in range(89)]
    asset('RUN_COHORT.json', {'outputs': [{'output_id': oid} for oid in ids]})
    for oid in ids:
        name = oid.replace(':', '_') + '.jsonl.gz'
        asset('tables/candidate_index/' + name, gzip.compress(json.dumps({
            'checkpoint': 1024, 'canonical_grid': [[3]], 'output_id': oid, 'depth': 24,
            'augmentation_id': AUG8[0], 'terminal_node_id': 3, 'nodes_expanded_so_far': 5}).encode()))
        asset('tables/node_index/' + name, gzip.compress(json.dumps({
            'augmentation_id': AUG8[0], 'node': {'node_id': 3, 'cumulative_score': 0.3}}).encode()))
    write(root / 'manifests/ARCHIVE_HASHES.json', {'status': 'PASS', 'files': hashes})
    write(root / 'manifests/ARCHIVE_HASH_VERIFICATION.json', {'status': 'PASS'})
    checkpoint = tmp_path / 'checkpoints'
    first = module.source_rankings(root, 24, checkpoint)
    assert len(first[1]) == 89 and first[1][ids[0]]['NLL'] == [[[3]]]
    def forbidden(*args):
        raise AssertionError('Checkpoint reuse must not reread node archives')
    monkeypatch.setattr(module, 'terminal_scores', forbidden)
    assert module.source_rankings(root, 24, checkpoint) == first
    hashes['tables/node_index/0_o0.jsonl.gz'] = 'changed'
    write(root / 'manifests/ARCHIVE_HASHES.json', {'status': 'PASS', 'files': hashes})
    with pytest.raises(RuntimeError, match='RANKING_CHECKPOINT_IDENTITY_MISMATCH'):
        module.source_rankings(root, 24, checkpoint)


def test_pairing_uses_sixty_task_clusters_and_full_eighty_nine_outputs():
    from scripts.summarize_cpu_selector_completion import paired
    ids = [str(n) + ':o0' for n in range(60)] + [str(n) + ':o1' for n in range(29)]
    value = paired(['0:o0', '0:o1'], ['1:o0', '1:o1'], ids, draws=100)
    assert value['FIX'] == value['HARM'] == 2 and value['net_outputs'] == 0
    assert value['reference_task_mean_test_exact_Pass2'] == 1/60
    assert value['bootstrap_tasks'] == 60
    with pytest.raises(RuntimeError, match='NOT_EVAL60_89'):
        paired([], [], ids[:-1], draws=100)
