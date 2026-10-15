import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('archive_ledger', ROOT / 'scripts/freeze_phase3_archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


@pytest.mark.parametrize('representation', ['canonical', 'stored_bytes'])
def test_frozen_ledger_producer_representations(tmp_path, representation):
    data = b'unchanged frozen output\n'
    (tmp_path / 'output.json').write_bytes(data)
    ledger = {'files': {'output.json': hashlib.sha256(data).hexdigest()}, 'includes_raw': False}
    encoded = (json.dumps(ledger, indent=2, sort_keys=True) + '\n').encode()
    (tmp_path / 'HASHES.json').write_bytes(encoded)
    root = hashlib.sha256(encoded if representation == 'stored_bytes' else
                          json.dumps(ledger, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (tmp_path / 'FREEZE.json').write_text(json.dumps({'compact_ledger_sha256': root}))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = archive.verify_source_ledger(tmp_path, 'HASHES.json', 'FREEZE.json')
    assert result['status'] == 'PASS'
    assert result['root_representation'] == ('EXACT_FILE_BYTES' if representation == 'stored_bytes' else 'CANONICAL_JSON')
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    (tmp_path / 'output.json').write_bytes(b'corrupted')
    with pytest.raises(RuntimeError, match='source ledger verification failed'):
        archive.verify_source_ledger(tmp_path, 'HASHES.json', 'FREEZE.json')


def test_canonical_generation_root_does_not_accept_undeclared_byte_representation(tmp_path):
    ledger = {'files': {}}
    encoded = json.dumps(ledger, indent=2).encode()
    (tmp_path / 'GENERATION_HASHES.json').write_bytes(encoded)
    (tmp_path / 'FREEZE.json').write_text(json.dumps({'ledger_sha256': hashlib.sha256(encoded).hexdigest()}))
    with pytest.raises(RuntimeError):
        archive.verify_source_ledger(tmp_path, 'GENERATION_HASHES.json', 'FREEZE.json')
