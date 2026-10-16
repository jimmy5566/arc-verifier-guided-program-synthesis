import pytest
from scripts.complete_selector_portfolio_cpu import fuse, coverage


def test_rrf_reuses_exact_grids_and_limits_two_attempts():
    assert fuse([[[[1]], [[2]]], [[[2]], [[3]]], [[[2]], [[4]]]]) == [[[2]], [[1]]]


def test_tie_uses_base_rank_not_source_scale():
    assert fuse([[[[7]], [[8]]], [[[2]], [[3]]], []]) == [[[7]], [[2]]]


def test_duplicates_do_not_inflate_evidence():
    with pytest.raises(RuntimeError, match='DUPLICATE_WITHIN_SOURCE'):
        fuse([[[[1]], [[1]]]])


def test_request_gap_and_logical_dedup_are_distinct_from_actual_reuse():
    def row(oid, depth, kind, view):
        return {'output_id': oid, 'depth': str(depth), 'grid_key': '[[1]]', 'request_kind': kind,
                'scoring_view': view, 'prefix_tokens': '10', 'continuation_tokens': '1', 'eos_tokens': '1'}
    rows = [row('a:o0', 24, 'original', ''), row('b:o0', 0, 'original', ''),
            row('b:o0', 0, 'view', 'identity'), row('b:o0', 24, 'view', 'identity')]
    result = coverage(rows, {('a:o0', 24)})
    assert result['cached_requests'] == 1 and result['missing_requests'] == 3
    assert result['pending_output_count'] == 1 and result['pending_source_units'] == 2
    assert result['potential_distinct_pending_request_keys'] == 1
    assert result['dedup_estimate_is_verified_token_equivalence'] is False
