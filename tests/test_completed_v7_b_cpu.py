import json
import pytest
from scripts.score_completed_v7_b_cpu import verify_freeze, decode_attempts, same_top2, enrich_metric
from scripts.score_v7_frozen_oracle_only import write, digest


def fixture(root):
    root.mkdir()
    required = {'index': {str(i): {'identity': {'output_id': str(i//2)+':o0', 'depth': (0,24)[i%2]},
                                 'payload_sha256': str(i)} for i in range(102)}}
    reuse = {'status': 'PASS', 'source_modified': False, 'reused_count': 102,
             'units': [{**row['identity'], 'payload_sha256': row['payload_sha256']}
                       for row in required['index'].values()]}
    write(root/'SHARD_REUSE_RECEIPT.json', reuse)
    write(root/'B_ATTEMPTS_FROZEN.json', {'fixture': True})
    ledger = {name: digest((root/name).read_bytes()) for name in
              ('SHARD_REUSE_RECEIPT.json', 'B_ATTEMPTS_FROZEN.json')}
    write(root/'PRE_GOLD_HASHES.json', ledger)
    freeze = {'status': 'PASS', 'gold_loaded': False, 'B_attempts_frozen': True,
              'output_count': 89, 'task_count': 60, 'source_units': 178,
              'scoring_views_complete': '8/8', 'view_rows': 57808, 'candidate_count': 7226,
              'hash_verification': {'status': 'PASS', 'expected': ledger, 'observed': ledger}}
    write(root/'PRE_GOLD_FREEZE.json', freeze)
    return required


def test_complete_receipt_verifies_required_cache_and_consumed_bytes(tmp_path):
    root = tmp_path/'b'
    required = fixture(root)
    assert verify_freeze(root, required)[0]['source_units'] == 178
    write(root/'B_ATTEMPTS_FROZEN.json', {'tampered': True})
    with pytest.raises(RuntimeError, match='SCORING_INPUT_HASH_MISMATCH'):
        verify_freeze(root, required)


def test_missing_required_reuse_fails_even_when_freeze_passes(tmp_path):
    root = tmp_path/'b'; required = fixture(root)
    required['index']['0']['payload_sha256'] = 'other'
    with pytest.raises(RuntimeError, match='REQUIRED_102_CACHE_REUSE_NOT_ESTABLISHED'):
        verify_freeze(root, required)


def test_incomplete_views_fail_before_scoring(tmp_path):
    root = tmp_path/'b'; required = fixture(root)
    freeze = json.loads((root/'PRE_GOLD_FREEZE.json').read_bytes())
    freeze['view_rows'] -= 1
    write(root/'PRE_GOLD_FREEZE.json', freeze)
    with pytest.raises(RuntimeError, match='FULL_B_FREEZE_CONTRACT_FAILED'):
        verify_freeze(root, required)


def test_full_attempt_identity_and_singleton_cached_parity():
    row = {'canonical_grid': [[1]], 'grid_key': '[[1]]'}
    assert decode_attempts({'x:o0': {'attempt_1': row, 'attempt_2': row}}, ['x:o0']) == {'x:o0': [[[1]], [[1]]]}
    assert same_top2([[[1]]], [[[1]], [[1]]])
    assert not same_top2([[[1]], [[2]]], [[[2]], [[1]]])
    with pytest.raises(RuntimeError, match='B_ATTEMPT_COHORT_MISMATCH'):
        decode_attempts({}, ['x:o0'])


def test_task_weighting_is_not_output_weighting():
    ids = [str(i)+':o0' for i in range(60)] + [str(i)+':o1' for i in range(29)]
    value = enrich_metric({'Top1_hit_output_ids': ['0:o0','0:o1'],
                           'Top2_hit_output_ids': ['0:o0','0:o1']}, ids)
    assert value['output_Top2_rate'] == 2/89
    assert value['task_mean_test_exact_Top2'] == 1/60
