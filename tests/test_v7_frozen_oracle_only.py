import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts import score_v7_frozen_oracle_only as s


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def archive(root, depth):
    outputs = ['task00:o' + str(i) for i in range(30)] + [f'task{i:02}:o0' for i in range(1, 60)]
    write(root / 'CONTRACT.json', {'ttt_depth': depth, 'gold_loaded': False})
    write(root / 'SOURCE_COMPLETENESS_AUDIT.json', {'status': 'PASS', 'gold_loaded': False,
          'outputs_found': 89, 'canonical_aug8_cells_found': 712})
    write(root / 'RUN_COHORT.json', {'outputs': [{'output_id': oid} for oid in outputs]})
    for oid in outputs:
        p = root / 'tables/candidate_index' / (oid.replace(':', '_') + '.jsonl.gz')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(json.dumps({'output_id': oid, 'depth': depth,
                      'checkpoint': 1024, 'canonical_grid': [[1]]}).encode(), mtime=0))
    hashes = {str(p.relative_to(root)).replace('\\', '/'): s.digest(p.read_bytes())
              for p in root.rglob('*') if p.is_file()}
    write(root / 'manifests/ARCHIVE_HASHES.json', {'status': 'PASS', 'files': hashes})
    write(root / 'manifests/ARCHIVE_HASH_VERIFICATION.json', {'status': 'PASS'})
    return outputs


def test_checked_pool_and_corruption_rejection(tmp_path):
    outputs = archive(tmp_path, 24)
    ids, pool, evidence = s.load_pool(tmp_path, 24)
    assert ids == outputs and len(pool) == 89
    assert len(evidence['scoring_input_hashes']) == 89
    (tmp_path / 'tables/candidate_index/task00_o0.jsonl.gz').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='HASH_MISMATCH'):
        s.load_pool(tmp_path, 24)


def test_main_freezes_inputs_before_gold_and_never_claims_top2(tmp_path):
    for label, depth in [('v7', 24), ('d24', 24), ('d48', 48)]:
        archive(tmp_path / label, depth)
    gold = tmp_path / 'gold.json'
    write(gold, {f'task{i:02}': [[[1]]] * (30 if i == 0 else 1) for i in range(60)})
    expected = s.digest(gold.read_bytes())
    original = Path.read_bytes
    def guarded(path):
        if path == gold:
            assert (tmp_path / 'out/ORACLE_INPUT_FREEZE.json').is_file()
        return original(path)
    args = ['scorer', '--v7', str(tmp_path / 'v7'), '--d24', str(tmp_path / 'd24'),
            '--d48', str(tmp_path / 'd48'), '--solutions', str(gold), '--output', str(tmp_path / 'out'),
            '--source-commit', 'a' * 40]
    with patch('sys.argv', args), patch.object(s, 'GOLD_SHA256', expected), patch.object(Path, 'read_bytes', guarded):
        s.main()
    result = s.read(tmp_path / 'out/ORACLE_SCORECARD.json')
    assert result['output_hits']['V7'] == 89
    assert result['B_Top2'] == 'NOT_RUN' and result['gpu_used'] is False


def test_overlap_requires_actual_hit_sets():
    result = s.overlap({'a', 'b', 'c'}, {'a', 'd'})
    assert result['v7_only_FIX'] == 2 and result['baseline_only_HARM'] == 1
    assert result['covers_all_baseline_hits'] is False
