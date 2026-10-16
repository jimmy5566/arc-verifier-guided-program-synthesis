import gzip
import json
from scripts import index_frozen_b_score_reuse as s


def fixture(root, value=0.1):
    identity = {k: 'same' for k in s.KEYS}
    identity.update(output_id='task:o0', depth=24, scoring_views=list(range(8)))
    unsigned = dict(identity, status='COMPLETE',
                    source_candidates=[{'original_log_likelihood': -value, 'mean_view_nll': value}],
                    view_likelihood_rows=[{}] * 8)
    payload = dict(unsigned, payload_sha256=s.sha(s.canonical(unsigned).encode()))
    path = root / 'SCORING_SHARDS/task_o0_d24.json.gz'
    path.parent.mkdir(parents=True)
    raw = gzip.compress(s.canonical(payload).encode(), mtime=0)
    path.write_bytes(raw)
    receipt = {'status': 'PASS', 'shard_file_sha256': s.sha(raw),
               'payload_sha256': payload['payload_sha256'], 'provenance': identity}
    path.with_suffix('').with_suffix('.sha256.json').write_text(json.dumps(receipt))
    return path


def test_deduplicates_exact_shards_without_changing_parent(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    p = fixture(a)
    fixture(b)
    before = p.read_bytes()
    result = s.build([a, b])
    assert result['reusable_source_units'] == 1
    assert result['duplicate_identical_copies'] == 1
    assert p.read_bytes() == before and result['gpu_used'] is False


def test_never_reuses_conflicting_or_corrupt_scores(tmp_path):
    a, b, c = [tmp_path / n for n in ['a', 'b', 'c']]
    fixture(a)
    fixture(b, 0.2)
    fixture(c).write_bytes(b'bad')
    result = s.build([a, b, c])
    assert result['reusable_source_units'] == 0
    assert len(result['conflicts_not_reused']) == 1
    assert len(result['rejected_not_reused']) == 1
