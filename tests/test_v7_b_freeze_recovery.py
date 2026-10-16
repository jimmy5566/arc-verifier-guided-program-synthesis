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


def test_only_changed_metadata_file_is_rehashed(tmp_path):
    import os
    root, training, archive = fixture(tmp_path)
    config = root / 'task/depth_024/adapter_config.json'
    stat = config.stat()
    os.utime(config, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    original = b.sha256
    def bounded_hash(path):
        assert Path(path).suffix != '.safetensors'
        return original(path)
    with patch.object(b, 'sha256', side_effect=bounded_hash):
        result = b.adapter_preflight(training, root, ['task'], archive)
    assert result['rehashed_changed_files'] == ['task/adapter_config.json']


def test_worker_command_cannot_generate_train_or_accept_gold(tmp_path):
    args = argparse.Namespace(**{n: tmp_path / n for n in ('archive', 'training', 'adapter_root',
        'model_path', 'foundation_adapter', 'challenge', 'native_config', 'output')})
    command = command_for(args)
    assert command[1].endswith('run_v7_ttt24_eval60_b_selector.py')
    assert command[-2:] == ['--batch-size', '8']
    assert '--solutions' not in command


def test_recovery_copies_only_identity_matched_immutable_shards(tmp_path):
    from scripts.run_exact_b_selector_phase3_r1024_v1 import write_complete_shard, shard_paths, SCORING_VIEWS
    old, new = tmp_path / 'old', tmp_path / 'new'
    expected = {k: 'fixture' for k in ('candidate_set_sha256', 'candidate_index_sha256',
        'adapter_model_sha256', 'model_identity', 'tokenizer_identity')}
    expected.update(output_id='task:o0', depth=24, scoring_context_window=16384, scoring_views=list(SCORING_VIEWS))
    write_complete_shard(old, 'task:o0', 24, expected,
                         [{'original_log_likelihood': -0.2, 'mean_view_nll': 0.2}],
                         [{} for _ in SCORING_VIEWS], [])
    file, sidecar = shard_paths(old, 'task:o0', 24)
    before = file.read_bytes(), sidecar.read_bytes()
    payload, reused = b.load_or_reuse_shard(new, old, 'task:o0', 24, expected)
    assert reused and payload['status'] == 'COMPLETE'
    assert before == (file.read_bytes(), sidecar.read_bytes())
    wrong = dict(expected, adapter_model_sha256='wrong')
    assert b.load_or_reuse_shard(tmp_path / 'other', old, 'task:o0', 24, wrong) == (None, False)


def test_consecutive_same_task_does_not_reload_weights(tmp_path):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    loader = Mock(return_value={})
    setter = Mock(return_value=SimpleNamespace(unexpected_keys=[]))
    model = SimpleNamespace(base_model=SimpleNamespace(set_adapter=Mock()),
                            parameters=lambda: [], eval=Mock())
    with patch.dict(sys.modules, {'peft': SimpleNamespace(set_peft_model_state_dict=setter),
                                 'safetensors.torch': SimpleNamespace(load_file=loader)}):
        b.activate_task(model, tmp_path / 'task1')
        b.activate_task(model, tmp_path / 'task1')
        assert loader.call_count == 1
        b.activate_task(model, tmp_path / 'task2')
        assert loader.call_count == 2


def test_verified_cached_unit_may_never_fall_back_to_gpu_recompute():
    expected = {'output_id': 'task:o0', 'depth': 24, 'adapter_model_sha256': 'frozen'}
    required = {('task:o0', 24): dict(expected)}
    with pytest.raises(RuntimeError, match='CACHED_UNIT_RECOMPUTE_FORBIDDEN'):
        b.forbid_cached_recompute(expected, None, required)
    with pytest.raises(RuntimeError, match='REQUIRED_REUSE_SCIENTIFIC_IDENTITY_MISMATCH'):
        b.forbid_cached_recompute(dict(expected, adapter_model_sha256='changed'), {}, required)
    b.forbid_cached_recompute(expected, {'status': 'COMPLETE'}, required)
    b.forbid_cached_recompute({'output_id': 'missing:o0', 'depth': 24}, None, required)


def test_required_index_is_hash_bound_and_duplicate_units_rejected(tmp_path):
    index = tmp_path / 'index.json'
    value = {'identity': {'output_id': 'task:o0', 'depth': 24}}
    write(index, {'index': {'a': value}})
    sha = b.sha256(index)
    assert ('task:o0', 24) in b.required_reuse_units(index, sha)
    with pytest.raises(RuntimeError, match='REQUIRED_REUSE_INDEX_HASH_MISMATCH'):
        b.required_reuse_units(index, 'wrong')
    write(index, {'index': {'a': value, 'b': value}})
    with pytest.raises(RuntimeError, match='REQUIRED_REUSE_DUPLICATE_UNIT'):
        b.required_reuse_units(index, b.sha256(index))
