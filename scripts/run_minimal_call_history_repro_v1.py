#!/usr/bin/env python3
"""Minimal target-blind reproducer for Dynamic-READY incremental call history.

This intentionally executes no decoder search: each condition contains only a
root prefill and the one-token B1 forwards needed to test whether a frozen
``flip_ud`` request changes after one ordinary ``anti_transpose`` forward.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_dynamic_ready import clone_legacy_cache, start_ready_cell
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import adapter_records, atomic_json, load_adapter, no_gold_challenge
from scripts.run_regret_dynamic_ready_v1 import ADAPTER_SHA, DEPTH, OUTPUT_INDEX, TASK_ID, decoder
from scripts.turbodfs_v4_common import assert_native_token_contract, sha256_file

EXPERIMENT = "MINIMAL_CALL_HISTORY_REPRO_V1"
PRIMARY = "flip_ud"
FOREIGN = "anti_transpose"
ARC_TOKENS = tuple(range(11)) + (15,)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_record(value: Any) -> dict[str, Any]:
    detached = value.detach()
    return {
        "shape": [int(item) for item in detached.shape], "dtype": str(detached.dtype),
        "device": str(detached.device), "stride": [int(item) for item in detached.stride()],
        "storage_offset": int(detached.storage_offset()), "contiguous": bool(detached.is_contiguous()),
        "content_sha256": _sha_bytes(detached.float().cpu().contiguous().numpy().tobytes()),
    }


def _cache_record(cache: Any) -> dict[str, Any]:
    """Record accessible cache structure without touching tensors in-place."""
    import torch

    tensors: list[dict[str, Any]] = []
    opaque: list[dict[str, str]] = []
    visited: set[int] = set()

    def walk(value: Any, path: str) -> None:
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, torch.Tensor):
            tensors.append({"path": path, **_tensor_record(value)})
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
            return
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                walk(value[key], f"{path}[{key!r}]")
            return
        # Transformers cache objects use one of these public containers across
        # supported versions.  Do not introspect arbitrary private attributes.
        found = False
        for attr in ("key_cache", "value_cache", "layers", "keys", "values"):
            try:
                child = getattr(value, attr)
            except Exception:
                continue
            found = True
            walk(child, f"{path}.{attr}")
        if not found:
            opaque.append({"path": path, "type": f"{type(value).__module__}.{type(value).__qualname__}"})

    walk(cache, "cache")
    aggregate = hashlib.sha256(json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"python_type": f"{type(cache).__module__}.{type(cache).__qualname__}", "tensor_count": len(tensors),
            "tensors": tensors, "opaque_nodes": opaque, "content_sha256": aggregate}


def _callable_record(value: Any) -> dict[str, Any]:
    source = None
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        pass
    try:
        source_file = inspect.getsourcefile(value)
    except (OSError, TypeError):
        source_file = None
    return {"module": getattr(value, "__module__", None), "qualname": getattr(value, "__qualname__", None),
            "source_file": source_file, "source_sha256": None if source is None else _sha_bytes(source.encode())}


def _backend_record(model: Any) -> dict[str, Any]:
    import torch

    first_attention = None
    try:
        first_attention = model.model.layers[0].self_attn
    except (AttributeError, IndexError, TypeError):
        pass
    package_versions = {}
    for package in ("torch", "transformers", "unsloth", "unsloth-zoo", "peft", "torchao", "triton", "xformers"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "MISSING"
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False)
    return {
        "packages": package_versions, "python": sys.version,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_smi": smi.stdout.strip() if smi.returncode == 0 else None,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_forward": _callable_record(model.forward),
        "model_attn_implementation": getattr(model.config, "_attn_implementation", None),
        "model_use_cache": getattr(model.config, "use_cache", None),
        "first_attention_class": None if first_attention is None else f"{type(first_attention).__module__}.{type(first_attention).__qualname__}",
        "first_attention_forward": None if first_attention is None else _callable_record(first_attention.forward),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _logit_record(logits: Any) -> dict[str, Any]:
    import torch

    row = logits[:, -1].float()
    logprob = torch.log_softmax(row, dim=-1)
    arc_logits = {str(token): float(row[0, token].item()) for token in ARC_TOKENS}
    return {
        "full_logits_sha256": _sha_bytes(row.cpu().contiguous().numpy().tobytes()),
        "arc_logits": arc_logits,
        "arc_logprobs": {str(token): float(logprob[0, token].item()) for token in ARC_TOKENS},
        "arc_ranking": sorted(ARC_TOKENS, key=lambda token: (-arc_logits[str(token)], token)),
    }


def _forward(model: Any, request: dict[str, Any], *, label: str) -> tuple[Any, dict[str, Any]]:
    import torch

    token = torch.tensor([[request["token_id"]]], device=model.device, dtype=torch.long)
    position = torch.tensor([[request["position"]]], device=model.device, dtype=torch.long)
    before = _cache_record(request["cache"])
    with torch.no_grad():
        output = model(input_ids=token, position_ids=position, past_key_values=request["cache"],
                       return_dict=True, use_cache=True)
    after = _cache_record(request["cache"])
    return output, {
        "label": label, "token_id": int(request["token_id"]), "position": int(request["position"]),
        "input": _tensor_record(token), "position_ids": _tensor_record(position),
        "attention_mask": "OMITTED", "cache_position": "OMITTED", "use_cache": True,
        "cache_before": before, "cache_after": after, "logits": _logit_record(output.logits),
    }


def _same(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    import torch

    left = lhs["logits"]; right = rhs["logits"]
    values = [abs(float(left["arc_logits"][str(token)]) - float(right["arc_logits"][str(token)])) for token in ARC_TOKENS]
    return {"full_logits_exact": left["full_logits_sha256"] == right["full_logits_sha256"],
            "arc_ranking_exact": left["arc_ranking"] == right["arc_ranking"], "max_abs_arc_logit_delta": max(values)}


def _load(contract: dict[str, Any], gpu_id: int) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    from scripts import run_eval60_adaptive_inference_joint_v2 as common
    from unsloth import FastLanguageModel

    runtime_args = SimpleNamespace(output=Path(contract["authoritative_root"]), challenge=Path(contract["challenge"]),
        reference_config=Path(contract["reference_config"]), model_path=Path(contract["model_path"]),
        native_config_dir=Path(contract["native_config_dir"]), gpu_id=gpu_id)
    _root, _manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    adapter_sha = load_adapter(model, adapter_records(Path(contract["adapter_manifest"])).get((TASK_ID, DEPTH)))
    if adapter_sha != ADAPTER_SHA:
        raise RuntimeError(f"unexpected adapter SHA: {adapter_sha}")
    task = view_task(tasks[TASK_ID], OUTPUT_INDEX)
    prompts = {}
    for view in (PRIMARY, FOREIGN):
        encoded, _augmentation = common.encoded_view(tokenizer=tokenizer, task=task, view=view, config=generation_config)
        prompts[view] = encoded["input_ids"]
    FastLanguageModel.for_inference(model)
    return model, prompts, {"adapter_sha256": adapter_sha, "native_token_contract": assert_native_token_contract(tokenizer)}


def _request_from_prefill(model: Any, prompt: Any, config: Any, view: str) -> dict[str, Any]:
    """Expose exactly the first legal Dynamic-READY request, without advancing it.

    The request is deliberately *not* a raw logits argmax.  Dynamic-READY's
    frontier-floor and Regret legality rules select it in ``start_ready_cell``.
    Using the actual coroutine request is required for an apples-to-apples B1
    call-history discriminator.
    """
    cell = start_ready_cell(model=model, input_ids=prompt.to(model.device), config=config,
                            cell_key=f"{TASK_ID}:o{OUTPUT_INDEX}:d{DEPTH}:{view}",
                            normalize_root_cache=False, active_time_accounting=True)
    request = cell.request
    if request is None:
        raise RuntimeError(f"{view} prefill yielded no incremental request")
    root_cache = clone_legacy_cache(request.cache)
    return {"token_id": int(request.token_id), "position": int(request.position), "ordinal": int(request.ordinal),
            "cache": root_cache, "prefill_cache": _cache_record(root_cache)}


def _release(model: Any) -> None:
    import torch
    del model
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def _condition_immediate(contract: dict[str, Any], gpu_id: int) -> dict[str, Any]:
    model, prompts, load_meta = _load(contract, gpu_id)
    try:
        config = decoder(contract["caps"])
        flip = _request_from_prefill(model, prompts[PRIMARY], config, PRIMARY)
        frozen = clone_legacy_cache(flip["cache"])
        first, first_row = _forward(model, {**flip, "cache": clone_legacy_cache(frozen)}, label="A_baseline")
        second, second_row = _forward(model, {**flip, "cache": clone_legacy_cache(frozen)}, label="B_immediate_replay")
        return {"load": load_meta, "backend": _backend_record(model), "flip_root": {key: value for key, value in flip.items() if key != "cache"},
                "baseline": first_row, "replay": second_row, "comparison": _same(first_row, second_row)}
    finally:
        _release(model)


def _condition_foreign(contract: dict[str, Any], gpu_id: int) -> dict[str, Any]:
    model, prompts, load_meta = _load(contract, gpu_id)
    try:
        config = decoder(contract["caps"])
        flip = _request_from_prefill(model, prompts[PRIMARY], config, PRIMARY)
        foreign = _request_from_prefill(model, prompts[FOREIGN], config, FOREIGN)
        frozen = clone_legacy_cache(flip["cache"])
        baseline, baseline_row = _forward(model, {**flip, "cache": clone_legacy_cache(frozen)}, label="C_baseline_before_foreign")
        foreign_out, foreign_row = _forward(model, foreign, label="D_one_ordinary_anti_incremental")
        after, after_row = _forward(model, {**flip, "cache": clone_legacy_cache(frozen)}, label="E_flip_after_one_foreign_incremental")
        return {"load": load_meta, "backend": _backend_record(model), "flip_root": {key: value for key, value in flip.items() if key != "cache"},
                "foreign_root": {key: value for key, value in foreign.items() if key != "cache"}, "baseline": baseline_row,
                "foreign": foreign_row, "after_foreign": after_row, "comparison": _same(baseline_row, after_row),
                "foreign_prefill_before_baseline": True}
    finally:
        _release(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--authoritative-root", type=Path, required=True); parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True); parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True); parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gpu-id", type=int, default=0); args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    no_gold_challenge(args.challenge)
    contract = {"experiment": EXPERIMENT, "source_commit": args.source_commit, "target_blind": True, "gold_accessed": False,
        "task_id": TASK_ID, "output_index": OUTPUT_INDEX, "depth": DEPTH, "primary_view": PRIMARY, "foreign_view": FOREIGN,
        "policy": "CUMULATIVE_REGRET_r=4.00", "caps": {"max_new_tokens": 931, "max_score": 1.6094379124341003,
        "max_expanded_nodes": 4096, "max_completed_candidates": 32, "frontier_floor": 1,
        "local_time_limit_seconds": 540.0, "pad_token_id": 13, "arc_tokens": list(ARC_TOKENS)},
        "challenge": str(args.challenge.resolve()), "challenge_sha256": sha256_file(args.challenge.resolve()),
        "authoritative_root": str(args.authoritative_root.resolve()), "reference_config": str(args.reference_config.resolve()),
        "model_path": str(args.model_path.resolve()), "native_config_dir": str(args.native_config_dir.resolve()),
        "adapter_manifest": str(args.adapter_manifest.resolve()), "adapter_manifest_sha256": sha256_file(args.adapter_manifest.resolve()),
        "protocol": "fresh model per condition; A/B immediate replay; C/D/E one foreign incremental; no search"}
    args.output.mkdir(parents=True); atomic_json(args.output / "CONTRACT.json", contract)
    immediate = _condition_immediate(contract, args.gpu_id); atomic_json(args.output / "IMMEDIATE_REPLAY.json", immediate)
    foreign = _condition_foreign(contract, args.gpu_id); atomic_json(args.output / "FOREIGN_CALL_REPLAY.json", foreign)
    decision = {"experiment": EXPERIMENT, "target_blind": True, "gold_accessed": False,
        "immediate_replay_exact": immediate["comparison"]["full_logits_exact"],
        "foreign_call_reproduces_drift": not foreign["comparison"]["full_logits_exact"],
        "foreign_call_changes_arc_ranking": not foreign["comparison"]["arc_ranking_exact"],
        "next": "INSPECT_RUNTIME_STATE" if not foreign["comparison"]["full_logits_exact"] else "DO_NOT_IMPLEMENT_FIX"}
    atomic_json(args.output / "DECISION.json", decision)
    names = ("CONTRACT.json", "IMMEDIATE_REPLAY.json", "FOREIGN_CALL_REPLAY.json", "DECISION.json")
    atomic_json(args.output / "HASHES.json", {"sha256": {name: sha256_file(args.output / name) for name in names}, "gold_accessed": False})
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
