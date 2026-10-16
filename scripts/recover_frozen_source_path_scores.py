"""CPU-only recovery of observed DFS source-path scores, never B imputation."""
import math


def recover_source_path(nodes, terminal_node_id, expected_token_ids):
    by_id = {}
    for node in nodes:
        node_id = node.get('node_id')
        if node_id in by_id:
            raise ValueError('DUPLICATE_NODE_ID')
        by_id[node_id] = node
    chain, seen = [], set()
    current = terminal_node_id
    while current is not None:
        if current in seen or current not in by_id:
            raise ValueError('BROKEN_OR_CYCLIC_ANCESTRY')
        seen.add(current)
        node = by_id[current]
        if node.get('selected_token') is not None:
            probability = node.get('token_logprob')
            if not isinstance(probability, (int, float)) or not math.isfinite(probability):
                raise ValueError('MISSING_OR_NONFINITE_TOKEN_LOGPROB')
            chain.append((int(node['selected_token']), float(probability)))
        elif node.get('parent_node_id') is not None:
            raise ValueError('TOKENLESS_NONROOT_NODE')
        current = node.get('parent_node_id')
    chain.reverse()
    tokens = [token for token, _ in chain]
    if tokens != list(expected_token_ids) or not tokens or tokens[-1] != 15:
        raise ValueError('EXACT_TOKEN_OR_EOS_MISMATCH')
    nll = -math.fsum(value for _, value in chain)
    stored = by_id[terminal_node_id].get('cumulative_score')
    # Existing tree accumulation uses finite-precision scalar sums. This is
    # a source-evidence consistency check, NOT a B/parity acceptance threshold.
    if (not isinstance(stored, (int, float)) or not math.isfinite(stored)
            or abs(float(stored) - nll) > 1e-4 * len(tokens)):
        raise ValueError('SOURCE_CUMULATIVE_NLL_INCONSISTENT')
    return {'status': 'SOURCE_PATH_RECOVERED', 'token_ids': tokens,
            'token_logprobs': [value for _, value in chain],
            'sequence_nll_from_token_logprobs': nll, 'stored_cumulative_nll': float(stored),
            'mean_token_nll_including_eos': nll / len(tokens),
            'node_ids': _ordered_ids(by_id, terminal_node_id),
            'B_replacement_validated': False,
            'scope': 'Observed source view only; no cross-view scores, B substitution, inference or Gold'}


def _ordered_ids(by_id, terminal):
    result = []
    while terminal is not None:
        result.append(terminal)
        terminal = by_id[terminal].get('parent_node_id')
    return list(reversed(result))
