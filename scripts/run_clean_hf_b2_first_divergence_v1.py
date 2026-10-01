#!/usr/bin/env python3
"""Target-blind internal localization of Clean-HF B1/B2 numerical divergence.

This diagnostic runs exactly one root incremental request for the real
``identity`` and ``flip_ud`` d59b0160/d24 cells.  It compares a cloned B1
cache with the corresponding row of a real packed B2 cache and records only
tensor hashes, shapes and max absolute differences.  It never opens Gold,
does not run DFS, and never writes model activations or candidate pools.
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

from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import (  # noqa: E402
    _cat_caches,
    _legacy_cache,
    _restore_cache_kind,
    cache_sha256,
    clone_legacy_cache,
    ready_incremental_forward_kwargs,
    start_ready_cell,
)
from scripts.run_clean_hf_parallel_regret_dfs_v1 import (  # noqa: E402
    _assert_challenge_only,
    _cache_transform,
    _config,
    _prompt_ids,
    _verify_frozen_foundation,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha(tensor: Any) -> str:
    import torch

    cpu = tensor.detach().contiguous().cpu()
    # NumPy has no native BF16 dtype on this runtime.  Hash its raw IEEE/BF16
    # words rather than widening values (which could erase a bit-level drift).
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.uint16)
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def _summary(left: Any, right: Any) -> dict[str, Any]:
    import torch
    a, b = left.detach(), right.detach()
    if tuple(a.shape) != tuple(b.shape):
        raise RuntimeError(f"comparison shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}")
    diff = (a.float() - b.float()).abs()
    return {
        "shape": list(a.shape), "dtype": str(a.dtype),
        "b1_sha256": _sha(a), "b2_row_sha256": _sha(b),
        "exact": bool(torch.equal(a, b)),
        "max_abs_difference": float(diff.max().item()),
        "nonzero_elements": int((diff != 0).sum().item()),
    }


def _extract_tensor(value: Any) -> Any:
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from arc.io import load_dataset

    _assert_challenge_only(args.challenge)
    foundation = json.loads(args.adapter_foundation.read_text(encoding="utf-8"))
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path, adapter_path=args.adapter_path, device=args.device,
        native_config_dir=args.native_config_dir,
        frozen_adapter_identity={
            "adapter_sha256": str(foundation["adapter_sha256"]),
            "adapter_config_sha256": str(foundation["adapter_config_sha256"]),
        },
    )
    _verify_frozen_foundation(args.adapter_foundation, identity)
    task = load_dataset(args.challenge)[args.task_id]
    config = _config(1)

    def make(view: str):
        return start_ready_cell(
            model=model,
            input_ids=_prompt_ids(tokenizer=tokenizer, task=task, output_index=args.output_index,
                                  view=view, device=args.device),
            config=config, cell_key=f"{args.task_id}:o{args.output_index}:d{args.depth}:{view}",
            normalize_root_cache=True, root_cache_transform=_cache_transform, cache_strategy="rollback",
        )

    cells = [make("identity"), make("flip_ud")]
    requests = [cell.request for cell in cells]
    if any(request is None for request in requests):
        raise RuntimeError("a diagnostic root cell completed before the first incremental request")
    first, second = requests  # type: ignore[misc]
    if first.cache_key != second.cache_key or first.position != second.position:
        raise RuntimeError("real inputs are B2 incompatible; refusing any padding/synthetic lane")

    # Clone before either forward: this proves B1 and B2 start from byte-identical
    # cache contents without mutating either actual owner used by the B2 path.
    b1_caches = [_restore_cache_kind(clone_legacy_cache(request.cache), request.cache) for request in requests]
    b2_legacy = _cat_caches([request.cache for request in requests])
    b2_cache = _restore_cache_kind(b2_legacy, first.cache)
    layer_modules = {
        name: module for name, module in model.named_modules()
        if any(name.endswith(suffix) for suffix in (
            "layers.0.input_layernorm", "layers.0.self_attn",
            "layers.0.post_attention_layernorm", "layers.0.mlp"))
    }
    captures: dict[str, list[Any]] = {name: [] for name in layer_modules}
    handles = []
    for name, module in layer_modules.items():
        handles.append(module.register_forward_hook(
            lambda _m, _i, output, name=name: captures[name].append(_extract_tensor(output).detach().clone())))
    try:
        b1_outputs = []
        for request, cache in zip(requests, b1_caches, strict=True):
            b1_outputs.append(model(**ready_incremental_forward_kwargs(
                token_ids=[request.token_id], position=request.position, cache=cache, device=model.device,
            ), output_hidden_states=True))
        b2_outputs = model(**ready_incremental_forward_kwargs(
            token_ids=[request.token_id for request in requests], position=first.position,
            cache=b2_cache, device=model.device,
        ), output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()

    # The hooks run twice for each logical module during scalar B1 then once for
    # B2.  We record the identity lane only; the flip lane is analysed too at
    # the model-output/hidden-state level below.
    result: dict[str, Any] = {
        "experiment": "CLEAN_HF_B2_FIRST_DIVERGENCE_V1",
        "target_blind": True, "gold_loaded": False, "unsloth_inference": False,
        "task_id": args.task_id, "output_index": args.output_index, "depth": args.depth,
        "token_ids": [int(request.token_id) for request in requests],
        "position": int(first.position),
        "input_cache_sha256": [cache_sha256(cache) for cache in b1_caches],
        "packed_input_cache_sha256": cache_sha256(b2_cache),
        "lanes": {}, "layer0_identity": {},
    }
    for lane, (b1_output, request) in enumerate(zip(b1_outputs, requests, strict=True)):
        hidden = [_summary(a, b2_outputs.hidden_states[index][lane:lane + 1])
                  for index, a in enumerate(b1_output.hidden_states)]
        first_hidden = next((index for index, row in enumerate(hidden) if not row["exact"]), None)
        logits = _summary(b1_output.logits, b2_outputs.logits[lane:lane + 1])
        result["lanes"][request.cell_key] = {
            "hidden_states": hidden, "first_hidden_divergence_index": first_hidden, "logits": logits,
        }
    for name, values in captures.items():
        if len(values) != 3:
            raise RuntimeError(f"unexpected hook cardinality for {name}: {len(values)}")
        result["layer0_identity"][name] = _summary(values[0], values[2][0:1])
    result["first_divergence"] = {
        key: value["first_hidden_divergence_index"] for key, value in result["lanes"].items()
    }
    _atomic_json(args.output / "CLEAN_HF_B2_FIRST_DIVERGENCE.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--adapter-foundation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
