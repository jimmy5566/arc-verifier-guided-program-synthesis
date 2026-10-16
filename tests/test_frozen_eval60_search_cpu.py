import hashlib

import pytest

from scripts.analyze_frozen_eval60_search_cpu import (
    LEGAL, analyze_cell, gold_node_lengths, gold_tokens, prefix_hashes, task_interval, frontier_wait_seconds,
)


def node(i, parent, token, state='pruned', lp=-1.0, **extra):
    return dict(node_id=i, parent_node_id=parent, selected_token=token,
                state=state, token_logprob=lp, **extra)


def test_native_gold_rectangular_aug8():
    expected = {
        'identity': '123\n456', 'flip_ud': '456\n123', 'flip_lr': '321\n654',
        'transpose': '14\n25\n36', 'anti_transpose': '63\n52\n41',
        'rot90': '36\n25\n14', 'rot180': '654\n321', 'rot270': '41\n52\n63',
    }
    for geometry, text in expected.items():
        assert gold_tokens([[1, 2, 3], [4, 5, 6]], geometry) == tuple(
            10 if c == '\n' else int(c) for c in text) + (15,)


def test_prefix_identity_and_ancestry():
    assert prefix_hashes((1, 10, 2))[hashlib.sha256(b'1,10,2').hexdigest()] == 3
    nodes = [node(0, None, None), node(1, 0, 1), node(2, 1, 2), node(3, 0, 9)]
    assert gold_node_lengths(nodes, (1, 2, 15))[1] == {0: 0, 1: 1, 2: 2, 3: None}


@pytest.mark.parametrize('nodes', [
    [node(0, None, None), node(0, None, None)],
    [node(1, 999, 1)], [node(1, 2, 1), node(2, 1, 2)],
])
def test_broken_ancestry_fails(nodes):
    with pytest.raises(RuntimeError):
        gold_node_lengths(nodes, (1, 2))


def test_observed_sibling_probabilities_and_unknown_rank():
    roots = [node(0, None, None, 'expanded', token_position=12)]
    full = roots + [node(t + 1, 0, t, lp=-float(t)) for t in LEGAL]
    a = analyze_cell({'nodes': full}, (1, 2, 15))
    assert a['probabilities'][0]['rank_among_12_legal_tokens'] == 2
    assert a['probabilities'][0]['gold_logprob'] == -1
    assert a['states'][0]['state'] == 'PRUNED'
    assert a['states'][1]['state'] == 'UNOBSERVED_OR_CENSORED'
    assert a['checkpoints']['128']['telemetry_available'] is False
    assert a['root_length'] == 12
    partial = analyze_cell({'nodes': roots + [node(1, 0, 1)]}, (1, 15))
    assert partial['probabilities'][0]['rank_among_12_legal_tokens'] is None


def test_telemetry_complete_rank_and_real_wait_not_counter_difference():
    prefix = hashlib.sha256(b'1').hexdigest()
    trace = {'branch_probabilities': [{
        'parent_node_id': 0, 'prefix_length': 0,
        'full_arc_logprobs': [{'token_id': t, 'logprob': -float(t)} for t in LEGAL],
    }], 'search_trace': [
        {'prefix_hash': prefix, 'frontier_insert_order': 100, 'elapsed_seconds': 2,
         'nodes_expanded_so_far': 120},
        {'prefix_hash': prefix, 'frontier_pop_order': 3, 'elapsed_seconds': 5,
         'nodes_expanded_so_far': 140},
    ]}
    a = analyze_cell({'nodes': [node(0, None, None), node(1, 0, 1, 'retained')],
                      'diagnostic_trace': trace}, (1, 15))
    assert a['states'][0]['state'] == 'EXPANDED'
    assert a['states'][0]['wait_seconds'] == 3
    assert a['checkpoints']['128']['max_popped_gold_prefix_length'] is None
    assert a['checkpoints']['256']['max_popped_gold_prefix_length'] == 1
    assert a['probabilities'][0]['rank_among_12_legal_tokens'] == 2


def test_retained_completed_and_bad_trace():
    nodes = [node(0, None, None), node(1, 0, 1, 'retained')]
    assert analyze_cell({'nodes': nodes}, (1, 15))['first_halt']['state'] == 'RETAINED_NOT_EXPANDED'
    completed = nodes + [node(2, 1, 15, 'completed')]
    assert analyze_cell({'nodes': completed}, (1, 15))['states'][1]['state'] == 'COMPLETED'
    with pytest.raises(RuntimeError, match='PREFIX_LENGTH'):
        analyze_cell({'nodes': nodes, 'diagnostic_trace': {'branch_probabilities': [
            {'parent_node_id': 0, 'prefix_length': 99}]}}, (1, 15))


def test_production_mutated_event_cannot_establish_wait_seconds():
    prefix = hashlib.sha256(b'1').hexdigest()
    cell = {'nodes': [node(0, None, None), node(1, 0, 1, 'expanded')],
            'diagnostic_trace': {'search_trace': [{
                'prefix_hash': prefix, 'frontier_insert_order': 100,
                'frontier_pop_order': 3, 'elapsed_seconds': 5,
            }]}}
    assert analyze_cell(cell, (1, 15))['states'][0]['wait_seconds'] is None


def test_wait_units_and_unknown_clock_cases():
    assert frontier_wait_seconds([]) is None
    assert frontier_wait_seconds([{'frontier_insert_order':99}, {'frontier_pop_order':2}]) is None
    assert frontier_wait_seconds([{'frontier_insert_order':99,'elapsed_seconds':5},
                                  {'frontier_pop_order':2,'elapsed_seconds':3}]) is None
    assert frontier_wait_seconds([{'frontier_insert_order':99,'elapsed_seconds':3},
                                  {'frontier_pop_order':2,'elapsed_seconds':5}]) == 2


def test_task_cluster_means_not_pseudoreplicated_outputs():
    result = task_interval([('a:o0', 0), ('a:o1', 2), ('b:o0', 3)])
    assert result['tasks'] == 2
    assert result['mean_task_delta'] == 2
    assert task_interval([])['status'] == 'UNOBSERVED'
