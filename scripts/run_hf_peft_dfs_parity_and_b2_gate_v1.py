#!/usr/bin/env python3
"""Target-blind clean-HF/PEFT parity gates for the Regret DFS replacement.

This is intentionally a small gate runner, not an experiment launcher.  It
uses only stock Transformers, PEFT and the frozen adapter/challenge inputs.
No training backend, Gold file, selector, or generation policy is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.hf_peft_backend import (
    adapter_tensor_manifest,
    loaded_adapter_tensor_parity,
    load_hf_peft_inference,
)


TASK_ID = "d59b0160"
OUTPUT_INDEX = 0
TARGET_VIEW = "flip_ud"
FOREIGN_VIEW = "anti_transpose"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_hash(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _assert_challenge_only(path: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("challenge must be a task mapping")
    for task_id, task in raw.items():
        if any("output" in test for test in task.get("test", ())):
            raise RuntimeError(f"test output present in purported challenge: {task_id}")


def _prompt_ids(*, tokenizer: Any, tasks: dict[str, Any], view: str, device: str) -> Any:
    from arc.task import ARCExample, ARCTask
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix
    from inference.nvarc_native_augmentation import NativeAugmentation

    raw_task = tasks[TASK_ID]
    task = ARCTask(raw_task.task_id, tuple(raw_task.train), (ARCExample(raw_task.test[OUTPUT_INDEX].input),))
    augmentation = NativeAugmentation(geometry=view, color_offset=0, pair_order="canonical")
    transformed = augmentation.transform_task(task)
    messages = native_messages_from_training_prefix(
        native_training_message_prefix(transformed), transformed.test[0].input,
    )
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True,
    )
    return encoded["input_ids"].to(device)


def _clone_dynamic_cache(cache: Any) -> Any:
    from transformers.cache_utils import DynamicCache
    from inference.nvarc_turbodfs_dynamic_ready import clone_legacy_cache

    return DynamicCache.from_legacy_cache(clone_legacy_cache(cache))


def _logits_hash(logits: Any) -> str:
    import torch

    value = logits.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return _sha256_bytes(value)


def _root_and_request(*, model: Any, input_ids: Any, token_id: int, cache: Any | None = None, position: int | None = None) -> Any:
    import torch
    from inference.nvarc_turbodfs_dynamic_ready import ready_incremental_forward_kwargs

    if cache is None:
        with torch.no_grad():
            return model(input_ids=input_ids, return_dict=True, use_cache=True)
    with torch.no_grad():
        return model(**ready_incremental_forward_kwargs(
            token_ids=[token_id], position=int(position), cache=cache, device=model.device,
        ))


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from arc.io import load_dataset

    _assert_challenge_only(args.challenge)
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        device=args.device,
        native_config_dir=args.native_config_dir,
    )
    source = adapter_tensor_manifest(args.adapter_path)
    parity = loaded_adapter_tensor_parity(model, source)
    if len(source) != 506 or len(parity) != 506 or not all(row["exact_equal"] for row in parity):
        raise RuntimeError("strict frozen adapter parity failed")
    tasks = load_dataset(args.challenge)
    target_ids = _prompt_ids(tokenizer=tokenizer, tasks=tasks, view=TARGET_VIEW, device=args.device)
    foreign_ids = _prompt_ids(tokenizer=tokenizer, tasks=tasks, view=FOREIGN_VIEW, device=args.device)
    target_root = _root_and_request(model=model, input_ids=target_ids, token_id=0)
    target_cache = _clone_dynamic_cache(target_root.past_key_values)
    target_position = int(target_ids.shape[1])
    # A foreign root prefill is deliberately outside the incremental-history
    # count.  It tests the established S0->S1 transition directly.
    foreign_root = _root_and_request(model=model, input_ids=foreign_ids, token_id=0)
    foreign_cache = _clone_dynamic_cache(foreign_root.past_key_values)
    foreign_position = int(foreign_ids.shape[1])
    for _ in range(args.foreign_incremental_calls):
        foreign_reply = _root_and_request(
            model=model, input_ids=foreign_ids, token_id=0,
            cache=foreign_cache, position=foreign_position,
        )
        foreign_cache = foreign_reply.past_key_values
        foreign_position += 1
    first = _root_and_request(
        model=model, input_ids=target_ids, token_id=0,
        cache=_clone_dynamic_cache(target_cache), position=target_position,
    )
    second = _root_and_request(
        model=model, input_ids=target_ids, token_id=0,
        cache=_clone_dynamic_cache(target_cache), position=target_position,
    )
    first_hash, second_hash = _logits_hash(first.logits), _logits_hash(second.logits)
    if first_hash != second_hash:
        raise RuntimeError("same-request replay is not exact")
    row = {
        "experiment": "HF_PEFT_DFS_PARITY_AND_B2_GATE_V1",
        "task_id": TASK_ID,
        "output_index": OUTPUT_INDEX,
        "target_view": TARGET_VIEW,
        "foreign_view": FOREIGN_VIEW,
        "foreign_incremental_calls": args.foreign_incremental_calls,
        "target_position": target_position,
        "target_token_id": 0,
        "target_prompt_tokens": int(target_ids.shape[1]),
        "foreign_prompt_tokens": int(foreign_ids.shape[1]),
        "full_logits_sha256": first_hash,
        "same_request_replay_sha256": second_hash,
        "same_request_exact": True,
        "ranking": torch.argsort(first.logits[0, -1], descending=True)[:12].detach().cpu().tolist(),
        "adapter_tensor_count": len(source),
        "adapter_exact_match_count": sum(item["exact_equal"] for item in parity),
        "runtime_identity": identity,
        "request_contract": {
            "input_ids": "one continuation token", "position_ids": "explicit absolute position",
            "cache_position": "explicit absolute position", "past_key_values": "Transformers DynamicCache",
            "attention_mask": "omitted because every request is a single unpadded sequence",
            "return_dict": True, "use_cache": True,
        },
    }
    _atomic_json(args.output / f"reentrancy_n{args.foreign_incremental_calls}.json", row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--foreign-incremental-calls", type=int, required=True, choices=(0, 1, 2, 8, 32, 128))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
