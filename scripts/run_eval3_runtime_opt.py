"""Target-blind RTX 5090 runtime experiment for the frozen Eval3 recipe.

This is deliberately an *experiment* entry point.  It imports the frozen
reference TTT implementation unchanged and varies only generation execution
mechanics (prompt/token cache and micro-batch size).  Every generated view and
deduplicated candidate is frozen before the separate scorer may read targets.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts import run_eval3_reference_ttt as frozen


EXPERIMENT_ID = "5090-blackwell-runtime-opt-v1"
BLACKWELL_UNLEASHED_BACKEND_ID = "blackwell_unleashed_v1"
FROZEN_STATUS = "EVAL3_RUNTIME_OPT_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_hash(task_ids: list[str]) -> str:
    return frozen._task_hash(task_ids)


def _identity(manifest: dict[str, Any], config: dict[str, Any], mode: str, batch_size: int, backend_id: str, task_ids: list[str], generation_execution: str) -> str:
    return _sha256({"experiment": backend_id, "manifest_hash": manifest["task_ids_hash"], "task_ids": task_ids, "config": config, "mode": mode, "generation_micro_batch_size": batch_size, "generation_execution": generation_execution})


@dataclass
class GpuSampler:
    """Low-overhead nvidia-smi sampling for a single, bounded stage."""

    physical_device: int
    interval_seconds: float = 0.25
    samples: list[dict[str, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    ["nvidia-smi", f"--id={self.physical_device}", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=3,
                )
                pieces = completed.stdout.strip().split(",")
                if len(pieces) == 3:
                    self.samples.append({
                        "utilization_pct": float(pieces[0].strip()),
                        "memory_used_mb": float(pieces[1].strip()),
                        "power_w": float(pieces[2].strip()),
                    })
            except (OSError, subprocess.SubprocessError, ValueError):
                # Telemetry failure never changes model execution semantics.
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._poll, name="arc2-gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4)

    def summary(self) -> dict[str, Any]:
        values = [item["utilization_pct"] for item in self.samples]
        memory = [item["memory_used_mb"] for item in self.samples]
        power = [item["power_w"] for item in self.samples]
        def percentile(items: list[float], percent: float) -> float | None:
            if not items:
                return None
            ordered = sorted(items)
            index = int(round((len(ordered) - 1) * percent))
            return ordered[index]
        return {
            "gpu_utilization_samples": len(values),
            "gpu_utilization_avg_pct": (sum(values) / len(values)) if values else None,
            "gpu_utilization_p90_pct": percentile(values, 0.90),
            "gpu_utilization_max_pct": max(values) if values else None,
            "nvidia_smi_memory_max_mb": max(memory) if memory else None,
            "gpu_power_avg_w": (sum(power) / len(power)) if power else None,
            "gpu_power_p90_w": percentile(power, 0.90),
            "gpu_power_max_w": max(power) if power else None,
        }


@contextlib.contextmanager
def _stage(name: str, *, device: int = 0) -> Iterable[dict[str, Any]]:
    """Measure one stage while resetting its CUDA high-water marks."""
    import torch

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    record: dict[str, Any] = {"stage": name, "allocated_start_bytes": int(torch.cuda.memory_allocated(device)), "reserved_start_bytes": int(torch.cuda.memory_reserved(device))}
    # CUDA-visible device 0 is intentionally local to the worker process.
    # nvidia-smi, however, addresses physical devices.  The launcher binds
    # this value before importing torch so GPU1 telemetry cannot silently
    # sample GPU0.
    physical_device = int(os.environ.get("ARC2_PHYSICAL_GPU_ID", str(device)))
    with GpuSampler(physical_device) as sampler:
        try:
            yield record
        finally:
            torch.cuda.synchronize(device)
            record["seconds"] = time.perf_counter() - started
            record["allocated_end_bytes"] = int(torch.cuda.memory_allocated(device))
            record["reserved_end_bytes"] = int(torch.cuda.memory_reserved(device))
            record["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            record["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
            record.update(sampler.summary())


def _mb(value: int) -> float:
    return round(value / (1024 * 1024), 3)


def _normalize_grid(grid: Any) -> list[list[int]] | None:
    if grid is None:
        return None
    return [[int(cell) for cell in row] for row in grid]


def _candidate_key(candidate: dict[str, Any]) -> str:
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))


def _encoded_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Return CPU tensors; putting tensors on device stays inside generation."""
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    return {name: value.detach().cpu() for name, value in encoded.items()}


def _left_pad(items: list[dict[str, Any]], *, pad_token_id: int) -> tuple[dict[str, Any], list[int]]:
    """Left-pad pre-tokenized prompts without changing their token sequence."""
    import torch

    lengths = [int(item["input_ids"].shape[-1]) for item in items]
    width = max(lengths)
    ids: list[Any] = []
    masks: list[Any] = []
    for item, length in zip(items, lengths, strict=True):
        pad = width - length
        ids.append(torch.nn.functional.pad(item["input_ids"][0], (pad, 0), value=pad_token_id))
        source_mask = item.get("attention_mask")
        if source_mask is None:
            source_mask = torch.ones((1, length), dtype=torch.long)
        masks.append(torch.nn.functional.pad(source_mask[0], (pad, 0), value=0))
    return {"input_ids": torch.stack(ids, dim=0), "attention_mask": torch.stack(masks, dim=0)}, lengths


def _generated_suffix(sequence: Any, *, input_width: int, eos_token_id: int | None, pad_token_id: int | None) -> Any:
    """Return one sequence's real continuation, excluding batch completion pad.

    `generate` returns a rectangular tensor.  In a heterogeneous greedy batch,
    shorter examples may be padded after their EOS.  Parsing those pads is
    harmless for the native tokenizer, but counting them as model output is
    not.  This helper preserves EOS itself (as serial generation does) and
    never strips a token before the first EOS.
    """
    suffix = sequence[input_width:]
    if eos_token_id is not None:
        eos_positions = (suffix == int(eos_token_id)).nonzero(as_tuple=False)
        if len(eos_positions):
            return suffix[: int(eos_positions[0].item()) + 1]
    if pad_token_id is not None and pad_token_id != eos_token_id:
        end = int(suffix.shape[-1])
        while end and int(suffix[end - 1].item()) == int(pad_token_id):
            end -= 1
        return suffix[:end]
    return suffix


def _select_cache_rows(cache: Any, indices: Any) -> Any:
    """Retain only unfinished batch rows in a Transformers dynamic cache.

    This is deliberately limited to the documented cache APIs.  Falling back
    to an undocumented tensor walk would make this performance experiment a
    model-state experiment instead.
    """
    if hasattr(cache, "batch_select_indices"):
        result = cache.batch_select_indices(indices)
        return cache if result is None else result
    if hasattr(cache, "reorder_cache"):
        result = cache.reorder_cache(indices)
        return cache if result is None else result
    raise RuntimeError("active compaction requires a documented dynamic-cache row-selection API")


def _greedy_active_compaction(
    *, model: Any, packed: dict[str, Any], max_new_tokens: int, eos_token_id: int | None,
) -> list[Any]:
    """Greedy decode with removal of EOS-completed rows.

    It is an execution-only equivalent of the frozen greedy `generate` call:
    no sampling, no altered maximum length, no altered EOS policy, and no
    candidate-level pruning.  The implementation intentionally refuses cache
    types that cannot select active rows through a public API.
    """
    import torch

    inputs = {name: value.to(model.device) for name, value in packed.items()}
    active_indices = torch.arange(inputs["input_ids"].shape[0], device=model.device)
    outputs: list[list[int]] = [[] for _ in range(int(active_indices.shape[0]))]
    attention_mask = inputs.get("attention_mask")
    with torch.inference_mode():
        response = model(**inputs, use_cache=True, return_dict=True)
        cache = getattr(response, "past_key_values", None)
        if cache is None:
            raise RuntimeError("active compaction requested but model did not return a KV cache")
        logits = response.logits[:, -1, :]
        for _ in range(max_new_tokens):
            next_tokens = torch.argmax(logits, dim=-1)
            keep_positions: list[int] = []
            for position, token in enumerate(next_tokens.detach().cpu().tolist()):
                original = int(active_indices[position].item())
                outputs[original].append(int(token))
                if eos_token_id is None or int(token) != int(eos_token_id):
                    keep_positions.append(position)
            if not keep_positions:
                break
            positions = torch.tensor(keep_positions, dtype=torch.long, device=model.device)
            cache = _select_cache_rows(cache, positions)
            active_indices = active_indices.index_select(0, positions)
            next_input = next_tokens.index_select(0, positions).unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = attention_mask.index_select(0, positions)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=model.device)], dim=-1,
                )
            response = model(input_ids=next_input, attention_mask=attention_mask, past_key_values=cache, use_cache=True, return_dict=True)
            cache = getattr(response, "past_key_values", None)
            if cache is None:
                raise RuntimeError("active compaction lost the KV cache during decode")
            logits = response.logits[:, -1, :]
    return [torch.tensor(values, dtype=torch.long) for values in outputs]


def _verify_kv_cache(model: Any, *, tokenizer: Any) -> dict[str, Any]:
    """Verify a cache object once, independent of target data or decoding."""
    import torch

    config_use_cache = bool(getattr(model.config, "use_cache", False))
    ids = torch.tensor([[int(tokenizer.eos_token_id)]], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        output = model(input_ids=ids, use_cache=True, return_dict=True)
    active = getattr(output, "past_key_values", None) is not None
    del output, ids
    return {"requested_use_cache": True, "model_config_use_cache": config_use_cache, "past_key_values_present": active, "KV_CACHE_ACTIVE": bool(config_use_cache and active)}


def _prepare_requests(
    *, tokenizer: Any, task: Any, config: dict[str, Any], cache_static_inputs: bool,
    augmentations: list[Any] | None = None, augmentation_seed_indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Create ordered generation requests without changing their prompt semantics.

    ``augmentations`` is deliberately optional.  The Eval3 runtime benchmark
    keeps its frozen first-eight behaviour by default.  The 5090 Eval60
    fixed-4+4 runner supplies a subset together with its *historical Aug8
    seed positions*, so portfolio execution can share this batch/static-KV
    implementation without re-numbering the frozen view seeds.
    """
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix
    from inference.nvarc_native_augmentation import bounded_native_augmentations, transform_tasks_for_augmentations

    augmentations = list(augmentations) if augmentations is not None else list(bounded_native_augmentations()[: int(config["generation_augmentation_count"])])
    if augmentation_seed_indices is None:
        augmentation_seed_indices = list(range(len(augmentations)))
    if len(augmentation_seed_indices) != len(augmentations):
        raise ValueError("augmentation seed indices must match the generation views")
    transformed = transform_tasks_for_augmentations(task, augmentations)
    prefixes = [native_training_message_prefix(item) for item in transformed]
    requests: list[dict[str, Any]] = []
    for augmentation_index, (augmentation, seed_index, augmented) in enumerate(zip(augmentations, augmentation_seed_indices, transformed, strict=True)):
        for test_index in range(len(task.test)):
            messages = native_messages_from_training_prefix(prefixes[augmentation_index], augmented.test[test_index].input)
            request = {"augmentation_index": augmentation_index, "seed_index": int(seed_index), "test_index": test_index, "augmentation": augmentation, "messages": messages}
            if cache_static_inputs:
                request["encoded"] = _encoded_prompt(tokenizer, messages)
            requests.append(request)
    return requests, augmentations


def _max_ttt_sequence_tokens(*, tokenizer: Any, task: Any, config: dict[str, Any]) -> int:
    """Telemetry-only inspection of the exact frozen train-only dialogue views."""
    from inference.arc_native_io import ARCNativeInputAdapter

    token_ids = [
        tokenizer(frozen._full_dialogue(item, ARCNativeInputAdapter.serialize_trusted_grid), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        for item in frozen._reference_variants(task)
    ]
    kept = [len(value) for value in token_ids if len(value) <= int(config["max_sequence_length"])]
    if not kept:
        raise RuntimeError("all reference train-only sequences exceed the frozen context bound")
    return max(kept)


def _generate_aug8_runtime(
    *, model: Any, tokenizer: Any, task: Any, config: dict[str, Any], cache_static_inputs: bool, micro_batch_size: int,
    generation_kwargs: dict[str, Any] | None = None, augmentations: list[Any] | None = None,
    augmentation_seed_indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], int, dict[str, Any], list[dict[str, Any]]]:
    """Frozen greedy Aug8 semantics with ordered micro-batched execution."""
    import torch
    from unsloth import FastLanguageModel
    from inference.dynamic_task_scheduler import task_seed
    from inference.nvarc_native import parse_native_grid
    from inference.nvarc_native_candidates import NativeGridCandidate, deduplicate_candidates

    if micro_batch_size not in {1, 2, 4}:
        raise ValueError("generation micro-batch must be one of 1, 2, 4")
    generation_kwargs = dict(generation_kwargs or {})
    requests, augmentations = _prepare_requests(
        tokenizer=tokenizer, task=task, config=config, cache_static_inputs=cache_static_inputs,
        augmentations=augmentations, augmentation_seed_indices=augmentation_seed_indices,
    )
    FastLanguageModel.for_inference(model)
    raw: list[dict[str, Any]] = []
    by_augmentation: dict[int, dict[int, dict[str, Any]]] = {index: {} for index in range(len(augmentations))}
    generated_tokens = 0
    prompt_lengths: list[int] = []
    generated_lengths: list[int] = []
    padded_prompt_tokens = 0
    packed_prompt_tokens = 0
    batch_count = 0
    active_sequence_trace: list[dict[str, Any]] = []
    with _stage("generation") as telemetry:
        for offset in range(0, len(requests), micro_batch_size):
            batch = requests[offset : offset + micro_batch_size]
            encoded = [item.get("encoded") or _encoded_prompt(tokenizer, item["messages"]) for item in batch]
            packed, lengths = _left_pad(encoded, pad_token_id=int(tokenizer.pad_token_id))
            prompt_lengths.extend(lengths)
            packed_width = int(packed["input_ids"].shape[-1])
            padded_prompt_tokens += packed_width * len(batch) - sum(lengths)
            packed_prompt_tokens += packed_width * len(batch)
            batch_count += 1
            if any(length > int(config["generation_context_window"]) for length in lengths):
                raise ValueError(f"generation prompt exceeds context: {max(lengths)}")
            # Greedy decoding does not consume RNG. Preserve the documented seed
            # derivation as explicit provenance for every request nonetheless.
            seeds = [task_seed(task.task_id, int(config["seed"]), f"reference-ttt:{item['seed_index']}:{item['test_index']}") for item in batch]
            for seed in seeds:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(torch.initial_seed())
            began = time.perf_counter()
            batch_generation_kwargs = dict(generation_kwargs)
            active_compaction = bool(batch_generation_kwargs.pop("active_sequence_compaction", False))
            if active_compaction:
                suffixes = _greedy_active_compaction(
                    model=model, packed=packed, max_new_tokens=int(config["max_new_tokens"]), eos_token_id=tokenizer.eos_token_id,
                )
                result = None
            else:
                with torch.inference_mode():
                    result = model.generate(
                        **{name: value.to(model.device) for name, value in packed.items()},
                        max_new_tokens=int(config["max_new_tokens"]), do_sample=False,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True, **batch_generation_kwargs,
                    )
            elapsed = time.perf_counter() - began
            input_width = int(packed["input_ids"].shape[-1])
            batch_generated_lengths: list[int] = []
            for position, item in enumerate(batch):
                suffix = (
                    suffixes[position].detach().cpu()
                    if active_compaction
                    else _generated_suffix(
                        result[position], input_width=input_width,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    ).detach().cpu()
                )
                text = tokenizer.decode(suffix, skip_special_tokens=True)
                parsed = parse_native_grid(text)
                grid = None if parsed is None else item["augmentation"].inverse_grid(parsed)
                value = {
                    "augmentation_index": item["augmentation_index"], "test_index": item["test_index"],
                    "prompt_tokens": lengths[position], "generated_tokens": int(suffix.shape[-1]),
                    "seed": seeds[position], "valid": grid is not None, "grid": _normalize_grid(grid),
                    "batch_size": len(batch), "batch_elapsed_seconds": elapsed,
                }
                raw.append(value); by_augmentation[item["augmentation_index"]][item["test_index"]] = value
                generated_tokens += int(suffix.shape[-1]); generated_lengths.append(int(suffix.shape[-1]))
                batch_generated_lengths.append(int(suffix.shape[-1]))
                del suffix
            max_length = max(batch_generated_lengths, default=0)
            active_sequence_trace.append({
                "batch_index": batch_count - 1,
                "initial_sequence_count": len(batch_generated_lengths),
                "max_generated_tokens": max_length,
                "real_generated_tokens": sum(batch_generated_lengths),
                "padded_finished_slot_tokens": max_length * len(batch_generated_lengths) - sum(batch_generated_lengths),
                "active_sequence_count_by_token": [sum(length > token for length in batch_generated_lengths) for token in range(max_length)],
            })
            del packed, encoded, result
    generated_candidates: list[Any] = []
    invalid = 0
    for augmentation_index, augmentation in enumerate(augmentations):
        outputs = [by_augmentation[augmentation_index][test_index] for test_index in range(len(task.test))]
        if any(not output["valid"] for output in outputs):
            invalid += 1
            continue
        grids = tuple(tuple(tuple(int(cell) for cell in row) for row in output["grid"]) for output in outputs)
        token_count = sum(int(output["generated_tokens"]) for output in outputs)
        elapsed = sum(float(output["batch_elapsed_seconds"]) / int(output["batch_size"]) for output in outputs)
        generated_candidates.append(NativeGridCandidate(augmentation, grids, token_count, elapsed))
    candidates = [candidate.to_dict() for candidate in deduplicate_candidates(generated_candidates)]
    telemetry.update({
        "generated_token_count": generated_tokens,
        "generation_tokens_per_second": generated_tokens / max(float(telemetry["seconds"]), 1e-9),
        "max_generation_prompt_tokens": max(prompt_lengths, default=0),
        "max_generated_tokens": max(generated_lengths, default=0),
        "request_count": len(requests), "micro_batch_size": micro_batch_size,
        "batch_count": batch_count,
        "padding_tokens": padded_prompt_tokens,
        "packed_prompt_tokens": packed_prompt_tokens,
        "padding_ratio": padded_prompt_tokens / max(packed_prompt_tokens, 1),
        "cache_static_inputs": cache_static_inputs,
        "invalid_candidate_count": invalid,
        "active_sequence_trace": active_sequence_trace,
        "finished_slot_waste_tokens": sum(int(item["padded_finished_slot_tokens"]) for item in active_sequence_trace),
        "finished_slot_waste_fraction": (
            sum(int(item["padded_finished_slot_tokens"]) for item in active_sequence_trace)
            / max(sum(int(item["max_generated_tokens"]) * int(item["initial_sequence_count"]) for item in active_sequence_trace), 1)
        ),
    })
    return candidates, invalid, telemetry, raw


def _record_fingerprint(record: dict[str, Any]) -> dict[str, Any]:
    """Target-blind evidence needed for exact per-view and pool comparisons."""
    raw = record.get("raw_views", [])
    candidates = record.get("candidates", [])
    return {
        "raw_views_sha256": _sha256(raw),
        "candidate_pool_sha256": _sha256(candidates),
        "invalid_candidate_count": record.get("invalid_candidate_count"),
        "unique_candidate_count": record.get("unique_candidate_count"),
        "loss_curve_sha256": _sha256(record.get("ttt", {}).get("loss_curve", [])),
    }


def compare_target_blind(serial: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Strict evidence parity; no tolerance, targets, or ranking decisions."""
    results: dict[str, Any] = {"reference_identity": serial["identity"], "candidate_identity": candidate["identity"], "tasks": {}}
    task_ids = serial["task_ids"]
    for task_id in task_ids:
        left, right = serial["records"][task_id], candidate["records"][task_id]
        a, b = _record_fingerprint(left), _record_fingerprint(right)
        task_result = {
            "ttt_loss_agreement": a["loss_curve_sha256"] == b["loss_curve_sha256"],
            "raw_view_candidate_agreement": a["raw_views_sha256"] == b["raw_views_sha256"],
            "candidate_pool_agreement": a["candidate_pool_sha256"] == b["candidate_pool_sha256"],
            "invalid_count_agreement": a["invalid_candidate_count"] == b["invalid_candidate_count"],
            "candidate_count_agreement": a["unique_candidate_count"] == b["unique_candidate_count"],
            "serial": a, "candidate": b,
        }
        task_result["PARITY_SAFE"] = all(value for key, value in task_result.items() if key.endswith("agreement"))
        results["tasks"][task_id] = task_result
    results["PARITY_SAFE"] = all(item["PARITY_SAFE"] for item in results["tasks"].values())
    results["scoring_stage"] = "NOT_APPLICABLE_IN_FROZEN_EVAL3_ANY_OF_K_PROTOCOL"
    results["final_selector_agreement"] = "NOT_APPLICABLE_IN_FROZEN_EVAL3_ANY_OF_K_PROTOCOL"
    return results


def _valid_checkpoint(path: Path, task_id: str, identity: str) -> dict[str, Any] | None:
    try:
        value = _read(path)
    except (OSError, json.JSONDecodeError):
        return None
    record = value.get("record")
    if value.get("identity") != identity or value.get("task_id") != task_id or not isinstance(record, dict):
        return None
    return record if record.get("task_id") == task_id and isinstance(record.get("candidates"), list) and isinstance(record.get("raw_views"), list) else None


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "reference_config", "challenge_path", "model_path", "native_config_dir", "output", "checkpoint_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--mode", choices=("serial", "cache"), required=True)
    parser.add_argument("--generation-micro-batch-size", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--backend-id", default=EXPERIMENT_ID)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--generation-execution", choices=("baseline", "active_compaction", "static_kv", "torch_compile", "static_kv_torch_compile", "active_compaction_static_kv"), default="baseline")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen runtime-optimization candidates")
    manifest, config = _read(args.manifest), _read(args.reference_config)
    manifest_task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "EVAL3_REFERENCE_TTT_COHORT_FROZEN" or len(manifest_task_ids) != 3 or manifest.get("task_ids_hash") != _task_hash(manifest_task_ids):
        raise ValueError("invalid frozen Eval3 manifest")
    task_ids = list(args.task_id) if args.task_id else manifest_task_ids
    if not task_ids or len(set(task_ids)) != len(task_ids) or set(task_ids) - set(manifest_task_ids):
        raise ValueError("requested task IDs must be a unique subset of the frozen Eval3 manifest")
    required = {"rank": 256, "alpha": 32, "ttt_steps": 24, "generation_augmentation_count": 8}
    if {key: config.get(key) for key in required} != required:
        raise ValueError("Eval3 TTT config violates the frozen contract")
    # The parent queue owns physical GPU binding.  Preserve it when supplied;
    # inside every child process the selected physical GPU is cuda:0.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0"); os.environ["TRITON_PTXAS_PATH"] = str(config["ptxas_path"])
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from unsloth import FastLanguageModel
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from arc.io import load_dataset
    from inference.nvarc_native import checkpoint_native_tokenizer

    if not Path(config["ptxas_path"]).is_file():
        raise RuntimeError("the verified PTXAS executable is unavailable")
    torch.cuda.set_device(0)
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 5090" not in gpu_name:
        raise RuntimeError(f"requires RTX 5090 Blackwell candidate environment, got {gpu_name}")
    identity = _identity(manifest, config, args.mode, args.generation_micro_batch_size, args.backend_id, task_ids, args.generation_execution)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True); (args.checkpoint_dir / "tasks").mkdir(exist_ok=True)
    resumed = {task_id: _valid_checkpoint(args.checkpoint_dir / "tasks" / f"{task_id}.json", task_id, identity) for task_id in task_ids} if args.resume else {}
    records = {key: value for key, value in resumed.items() if value is not None}; tasks = load_dataset(args.challenge_path)
    print(json.dumps({"event": "EVAL3_RUNTIME_OPT_TARGET_BLIND_START", "experiment_id": args.backend_id, "mode": args.mode, "generation_micro_batch_size": args.generation_micro_batch_size, "task_ids": task_ids, "resumed": sorted(records), "solutions_opened": False, "gpu": gpu_name, "physical_gpu_id": int(os.environ.get("ARC2_PHYSICAL_GPU_ID", "0"))}, sort_keys=True), flush=True)
    try:
        with _stage("model_load") as load_telemetry:
            model, tokenizer = FastLanguageModel.from_pretrained(model_name=str(args.model_path), full_finetuning=False, load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False, max_seq_length=int(config["max_sequence_length"]))
            native_tokenizer, metadata = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
            if len(tokenizer) != 16 or len(native_tokenizer) != 16 or tokenizer.get_vocab() != native_tokenizer.get_vocab():
                raise RuntimeError("Unsloth tokenizer differs from frozen NVARC native tokenizer")
            tokenizer = native_tokenizer
            model = FastLanguageModel.get_peft_model(model, r=int(config["rank"]), target_modules=list(config["target_modules"]), lora_alpha=int(config["alpha"]), lora_dropout=0.0, bias="none", use_gradient_checkpointing=False, random_state=int(config["seed"]), use_rslora=True, loftq_config=None)
            for _, parameter in model.named_parameters():
                if parameter.dtype == torch.float32:
                    parameter.data = parameter.data.to(torch.bfloat16)
        kv_cache = _verify_kv_cache(model, tokenizer=tokenizer)
        generation_kwargs: dict[str, Any] = {}
        if args.generation_execution in {"static_kv", "static_kv_torch_compile", "active_compaction_static_kv"}:
            generation_kwargs["cache_implementation"] = "static"
        if args.generation_execution in {"active_compaction", "active_compaction_static_kv"}:
            generation_kwargs["active_sequence_compaction"] = True
        default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        frozen_params = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
        if not trainable or not frozen_params:
            raise RuntimeError("invalid official adapter partition")
        adapter_before = {name: frozen._fingerprint(parameter) for name, parameter in trainable[:8]}
        base_fingerprints = {name: frozen._fingerprint(parameter) for name, parameter in frozen_params[:8]}
        for position, task_id in enumerate(task_ids, start=1):
            if task_id in records:
                continue
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); task_started = time.perf_counter()
            print(json.dumps({"event": "EVAL3_RUNTIME_OPT_TASK_START", "task_id": task_id, "task_position": position, "task_total": len(task_ids), "mode": args.mode, "generation_micro_batch_size": args.generation_micro_batch_size}, sort_keys=True), flush=True)
            task = tasks[task_id]
            max_ttt_tokens = _max_ttt_sequence_tokens(tokenizer=tokenizer, task=task, config=config)
            with _stage("ttt") as ttt_stage:
                ttt = frozen._fit_task(model=model, tokenizer=tokenizer, task=task, config=config, default_state=default_state, adapter_before=adapter_before, base_fingerprints=base_fingerprints)
            if not ttt["adapter_updated"] or not ttt["base_model_unchanged"]:
                raise RuntimeError(f"{task_id}: TTT integrity criterion failed")
            generation_model = model
            if args.generation_execution in {"torch_compile", "static_kv_torch_compile"}:
                # Compile only after the frozen TTT update.  TTT itself keeps
                # its exact eager trajectory; this probes generation launch
                # overhead only.  The original PEFT model remains the source
                # of truth for adapter reset and integrity checks.
                generation_model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=False)
            candidates, invalid, generation_stage, raw_views = _generate_aug8_runtime(model=generation_model, tokenizer=tokenizer, task=task, config=config, cache_static_inputs=(args.mode == "cache"), micro_batch_size=args.generation_micro_batch_size, generation_kwargs=generation_kwargs)
            if generation_model is not model:
                del generation_model
            torch.cuda.synchronize()
            whole_peak = {
                "allocated_mb": _mb(max(int(ttt_stage["peak_allocated_bytes"]), int(generation_stage["peak_allocated_bytes"]))),
                "reserved_mb": _mb(max(int(ttt_stage["peak_reserved_bytes"]), int(generation_stage["peak_reserved_bytes"]))),
                "allocated_current_mb": _mb(int(torch.cuda.memory_allocated())),
                "reserved_current_mb": _mb(int(torch.cuda.memory_reserved())),
            }
            record = {
                "task_id": task_id, "status": "SUCCESS" if candidates else "NO_VALID_NATIVE_CANDIDATE", "candidates": candidates, "raw_views": raw_views,
                "generated_candidate_count": 8, "unique_candidate_count": len(candidates), "invalid_candidate_count": invalid,
                "ttt": ttt, "telemetry": {"ttt": ttt_stage, "generation": generation_stage, "scoring": {"stage": "scoring", "status": "NOT_APPLICABLE_IN_FROZEN_EVAL3_ANY_OF_K_PROTOCOL", "seconds": 0.0, "tokens_per_second": None}, "whole_task": {"seconds": time.perf_counter() - task_started, "peak": whole_peak}},
                "KV_CACHE": {**kv_cache, "generation_execution": args.generation_execution, "generation_kwargs": generation_kwargs}, "generation_micro_batch_size": args.generation_micro_batch_size, "cache_static_inputs": args.mode == "cache", "max_ttt_sequence_tokens": max_ttt_tokens, "attention_backend": "xformers_verified_in_5090-blackwell-env-v2", "fail_soft_events": [], "tokenizer": metadata,
            }
            checkpoint = args.checkpoint_dir / "tasks" / f"{task_id}.json"; atomic_write_json(checkpoint, {"identity": identity, "task_id": task_id, "record": record})
            if _valid_checkpoint(checkpoint, task_id, identity) is None:
                raise RuntimeError(f"{task_id}: atomic checkpoint validation failed")
            records[task_id] = record
            print(json.dumps({"event": "EVAL3_RUNTIME_OPT_TASK_FROZEN", "task_id": task_id, "unique_candidates": len(candidates), "invalid": invalid, "seconds": record["telemetry"]["whole_task"]["seconds"], "solutions_opened": False}, sort_keys=True), flush=True)
            set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default"); gc.collect(); torch.cuda.empty_cache()
    finally:
        if "model" in locals():
            del model
        gc.collect()
        if "torch" in locals():
            torch.cuda.empty_cache()
    if set(records) != set(task_ids):
        raise RuntimeError("incomplete Eval3 candidate freeze")
    artifact = {"experiment_id": args.backend_id, "status": FROZEN_STATUS, "mode": args.mode, "generation_execution": args.generation_execution, "generation_micro_batch_size": args.generation_micro_batch_size, "task_ids": task_ids, "task_ids_hash": _task_hash(task_ids), "parent_manifest_task_ids_hash": manifest["task_ids_hash"], "identity": identity, "reference_config": config, "records": {task_id: records[task_id] for task_id in task_ids}, "solutions_opened": False, "scoring_stage": "NOT_APPLICABLE_IN_FROZEN_EVAL3_ANY_OF_K_PROTOCOL", "physical_gpu_id": int(os.environ.get("ARC2_PHYSICAL_GPU_ID", "0"))}
    atomic_write_json(args.output, artifact)
    print(json.dumps({"event": "EVAL3_RUNTIME_OPT_CANDIDATES_FROZEN", "task_count": len(task_ids), "candidate_count": sum(row["unique_candidate_count"] for row in records.values()), "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
