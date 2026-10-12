#!/usr/bin/env python3
"""Materialize target-free prompt token IDs with exact tokenizer-file gates.

Importing this module uses only the standard library.  ``--execute`` deliberately
loads a tokenizer only after every frozen file hash and target-boundary check
passes; it never imports or loads a model, torch, PEFT, or a remote client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = (ROOT / 'experiments/capability_repair_baseline_v1' /
        'connected_components_object_extraction_capability_pilot_v1' /
        'connected_component_count_fit_vs_generation_diagnostic_v1')
DEFAULT_INPUT = BASE / 'COHORT_PROMPT_TASKS_V1.json'
TOKENIZER_IDENTITY = {
    'distribution': 'KAGGLE',
    'repository': 'sorokin/qwen3_4b_grids15_sft139',
    'revision': 'Transformers/bfloat16/1',
    'runtime_path': '/workspace/arc2/models/qwen3_4b_grids15_sft139',
    'tokenizer_files': {
        'tokenizer.json': 'af6b9e855f9db3192bd9152607708ebd7700b40f9fc5f792a26af70150a96cd5',
        'tokenizer_config.json': 'e1912aeba4050616cbfe01b34df3f3df2818207eda0056f6eb87c2336936ea90',
        'vocab.json': '8141a61e47eceec466524bc178bc9c6d5076b7d1b18b456fbd3fdf66cae57f82',
        'special_tokens_map.json': 'f4f79e08d97f4d1c87f8d89264f525c8789da3b73b3bb55d1e12f692f41a7b1b',
    },
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def verify_tokenizer_identity(tokenizer_root: Path, identity: dict[str, Any] = TOKENIZER_IDENTITY) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in identity['tokenizer_files'].items():
        candidate = tokenizer_root / name
        if not candidate.is_file():
            raise RuntimeError(f'TOKENIZER_FILE_MISSING:{name}')
        actual = sha_file(candidate)
        if actual != expected:
            raise RuntimeError(f'TOKENIZER_FILE_SHA_MISMATCH:{name}')
        observed[name] = actual
    return observed


def serialize_grid(grid: list[list[int]]) -> str:
    if not isinstance(grid, list) or not grid or any(not isinstance(row, list) or not row for row in grid):
        raise RuntimeError('PROMPT_GRID_EMPTY_OR_INVALID')
    width = len(grid[0])
    if any(len(row) != width or any(type(cell) is not int or not 0 <= cell <= 9 for cell in row) for row in grid):
        raise RuntimeError('PROMPT_GRID_NOT_RECTANGULAR_DIGITS')
    return '\n'.join(''.join(str(cell) for cell in row) for row in grid)


def native_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    tests = task.get('test')
    if not isinstance(tests, list) or len(tests) != 1 or 'output' in tests[0]:
        raise RuntimeError('PROMPT_TEST_TARGET_BOUNDARY_INVALID')
    train = task.get('train')
    if not isinstance(train, list) or not train:
        raise RuntimeError('PROMPT_TRAIN_LAYOUT_INVALID')
    messages: list[dict[str, str]] = []
    for pair in train:
        if not isinstance(pair, dict) or 'input' not in pair or 'output' not in pair:
            raise RuntimeError('PROMPT_TRAIN_PAIR_INVALID')
        messages.extend((
            {'role': 'user', 'content': serialize_grid(pair['input'])},
            {'role': 'assistant', 'content': serialize_grid(pair['output'])},
        ))
    messages.append({'role': 'user', 'content': serialize_grid(tests[0]['input'])})
    return messages


def load_target_free_records(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding='utf-8'))
    records = document.get('records') if isinstance(document, dict) else None
    if not isinstance(document, dict) or document.get('target_content_present') is not False or not isinstance(records, list):
        raise RuntimeError('PROMPT_DOCUMENT_TARGET_BOUNDARY_INVALID')
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get('cohort_record_id'), str) or not isinstance(record.get('prompt_task_sha256'), str):
            raise RuntimeError('PROMPT_RECORD_SCHEMA_INVALID')
        if record['cohort_record_id'] in seen or not isinstance(record.get('task'), dict):
            raise RuntimeError('PROMPT_RECORD_DUPLICATE_OR_TASK_INVALID')
        if canonical_sha256(record['task']) != record['prompt_task_sha256']:
            raise RuntimeError('PROMPT_TASK_SHA_MISMATCH')
        native_messages(record['task'])
        seen.add(record['cohort_record_id'])
    if len(records) != 72:
        raise RuntimeError('PROMPT_COHORT_COUNT_INVALID')
    return sorted(records, key=lambda item: item['cohort_record_id'])


def _extract_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        encoded = encoded.get('input_ids')
    if hasattr(encoded, 'tolist'):
        encoded = encoded.tolist()
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise RuntimeError('TOKENIZER_BATCH_OUTPUT_UNEXPECTED')
        encoded = encoded[0]
    if not isinstance(encoded, list) or not encoded or any(type(token) is not int for token in encoded):
        raise RuntimeError('TOKENIZER_IDS_INVALID')
    return [int(token) for token in encoded]


def materialize_records(records: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    result = []
    for record in records:
        messages = native_messages(record['task'])
        # Native transport freezes chat-template rendering first.  Passing the
        # resulting text to the tokenizer with add_special_tokens=False avoids
        # a second, implementation-dependent BOS insertion.
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not isinstance(rendered, str) or not rendered:
            raise RuntimeError('CHAT_TEMPLATE_RENDER_INVALID')
        try:
            encoded = tokenizer(rendered, add_special_tokens=False, return_attention_mask=False)
        except TypeError:  # Small deterministic fake tokenizers used by CPU tests.
            encoded = tokenizer(rendered, add_special_tokens=False)
        token_ids = _extract_ids(encoded)
        result.append({
            'cohort_record_id': record['cohort_record_id'],
            'split': record['split'],
            'prompt_task_sha256': record['prompt_task_sha256'],
            'prompt_token_ids': token_ids,
            'prompt_token_ids_sha256': canonical_sha256(token_ids),
        })
    if len({item['cohort_record_id'] for item in result}) != len(result):
        raise RuntimeError('MATERIALIZED_RECORD_DUPLICATE')
    return result


def execute(tokenizer_root: Path, input_path: Path, output_path: Path) -> dict[str, Any]:
    observed = verify_tokenizer_identity(tokenizer_root)
    records = load_target_free_records(input_path)
    from transformers import AutoTokenizer  # Runtime-only tokenizer import; no model import.
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_root), local_files_only=True)
    result = materialize_records(records, tokenizer)
    document = {
        'schema_version': 1,
        'status': 'PROMPT_TOKEN_IDS_MATERIALIZED_TOKENIZER_ONLY',
        'target_content_present': False,
        'model_loaded': False,
        'gpu_used': False,
        'tokenizer_identity': {**TOKENIZER_IDENTITY, 'observed_file_sha256': observed},
        'source_prompt_document_sha256': sha_file(input_path),
        'transport': 'native_chat_template_text_then_tokenizer_add_special_tokens_false',
        'records': result,
    }
    return {'output_sha256': atomic_json(output_path, document), 'records': len(result), 'model_loaded': False, 'gpu_used': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--tokenizer-root', type=Path)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit('REFUSE_TOKENIZER_MATERIALIZATION_WITHOUT_EXPLICIT_EXECUTE')
    if not args.tokenizer_root or not args.output:
        raise SystemExit('TOKENIZER_ROOT_AND_OUTPUT_REQUIRED')
    print(json.dumps(execute(args.tokenizer_root, args.input, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
