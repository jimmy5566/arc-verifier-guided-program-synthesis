from copy import deepcopy
import pytest
from scripts.recover_frozen_source_path_scores import recover_source_path


def nodes():
    return [{'node_id': 0, 'parent_node_id': None, 'selected_token': None, 'cumulative_score': 0},
            {'node_id': 1, 'parent_node_id': 0, 'selected_token': 2, 'token_logprob': -0.2, 'cumulative_score': 0.2},
            {'node_id': 2, 'parent_node_id': 1, 'selected_token': 15, 'token_logprob': -0.3, 'cumulative_score': 0.5}]


def test_reconstructs_exact_eos_path_without_changing_source():
    source = nodes()
    before = deepcopy(source)
    result = recover_source_path(source, 2, [2, 15])
    assert source == before
    assert result['stored_cumulative_nll'] == 0.5
    assert result['node_ids'] == [0, 1, 2]
    assert result['B_replacement_validated'] is False


@pytest.mark.parametrize('change,reason', [
    (lambda n: n[2].update(parent_node_id=99), 'ANCESTRY'),
    (lambda n: n[2].update(token_logprob=None), 'LOGPROB'),
    (lambda n: n[2].update(cumulative_score=2), 'INCONSISTENT'),
    (lambda n: n[2].update(selected_token=3), 'EOS_MISMATCH'),
])
def test_missing_or_mismatched_evidence_never_becomes_a_score(change, reason):
    source = nodes()
    change(source)
    with pytest.raises(ValueError, match=reason):
        recover_source_path(source, 2, [2, 15])
