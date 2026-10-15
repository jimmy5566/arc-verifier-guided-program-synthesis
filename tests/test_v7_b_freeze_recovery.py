import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from scripts import run_v7_ttt24_eval60_b_selector as b
from scripts import run_v7_ttt24_aug8_r1024_eval60 as generation
from scripts.run_v7_eval60_b_freeze_recovery import command_for


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(tmp_path):
    root, training, archive = (tmp_path / n for n in ('adapters', 'training', 'archive'))
    adapter = root / 'task/depth_024'
    write(adapter / 'adapter_config.json', {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM',
          'r': 256, 'lora_alpha': 32, 'use_rslora': True, 'target_modules': list(generation.TTT24_TARGET_MODULES)})
    (adapter / 'adapter_model.safetensors').write_bytes(b'frozen-fixture')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter.iterdir()}
    record = {'adapter_path': str(adapter), 'adapter_files': files,
              'adapter_files_sha256': generation._sha_json(files)}
    manifest = {'status': 'FROZEN', 'target_blind': True, 'task_ids': ['task'], 'records': {'task': record}}
    write(training / 'TTT24_ADAPTER_MANIFEST.json', manifest)
    identities = {'task': generation._manifest_adapter_identity(adapter, record)}
    write(archive / 'ADAPTER_IDENTITIES.json', identities)
    write(archive / 'ADAPTER_IDENTITY_REUSE_RECEIPT.json', {
        'status': 'PASS', 'training_manifest_sha256': b.sha256(training / 'TTT24_ADAPTER_MANIFEST.json'),
        'adapter_identities_sha256': b.sha256(archive / 'ADAPTER_IDENTITIES.json'),
        'adapter_stat_snapshot': generation._adapter_stat_snapshot(root, manifest['records'])})
    return root, training, archive


def test_stable_adapter_does_not_rehash_weights(tmp_path):
    root, training, archive = fixture(tmp_path)
    original = b.sha256
    def bounded_hash(path):
        assert Path(path).suffix != '.safetensors'
        return original(path)
    with patch.object(b, 'sha256', side_effect=bounded_hash):
        result = b.adapter_preflight(training, root, ['task'], archive)
    assert result['verification_mode'] == 'REUSED_EXACT_SHA256_WITH_STABLE_METADATA'


def test_changed_adapter_falls_back_to_hash_and_rejects(tmp_path):
    root, training, archive = fixture(tmp_path)
    (root / 'task/depth_024/adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='TTT24_ADAPTER_HASH_MISMATCH'):
        b.adapter_preflight(training, root, ['task'], archive)


def test_worker_command_cannot_generate_train_or_accept_gold(tmp_path):
    args = argparse.Namespace(**{n: tmp_path / n for n in ('archive', 'training', 'adapter_root',
        'model_path', 'foundation_adapter', 'challenge', 'native_config', 'output')})
    command = command_for(args)
    assert command[1].endswith('run_v7_ttt24_eval60_b_selector.py')
    assert command[-2:] == ['--batch-size', '8']
    assert '--solutions' not in command
