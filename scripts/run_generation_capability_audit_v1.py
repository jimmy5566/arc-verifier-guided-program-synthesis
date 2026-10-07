from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from generation_capability_audit_v1.audit import (
    CONTEXT_WINDOW,
    MAX_NEW_TOKENS,
    MODEL_STATES,
    SOURCE_COMMIT,
    GenerationAuditError,
    aggregate_generation,
    best_checkpoint,
    build_generation_example,
    canonical_sha256,
    classify_novel_gain,
    context_audit_record,
    parse_generated_tokens,
    saturation_status,
    score_record,
    select_calibration_examples,
)


EXPECTED_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
NOVEL_SENTINEL_SHA256 = "a7cb00583eca5f05db4b9dc92107285600b31d6ad3f0c30a199e0810cdd161f8"
REPLAY_SENTINEL_SHA256 = "1a46c1089c5d4b673a72ceac0d7e06f6d4a5c6b23ba9c3f03838929448189cac"
FORBIDDEN_PATH_TERMS = ("holdout", "eval60", "gold", "solution")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8", newline="\n")
    temp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_rows(paths: list[Path], sample_ids: set[str]) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq
    result: dict[str, dict[str, Any]] = {}
    remaining = set(sample_ids)
    for path in paths:
        if not remaining:
            break
        table = pq.read_table(path, filters=[("sample_id", "in", sorted(remaining))])
        for row in table.to_pylist():
            sample_id = str(row["sample_id"])
            result[sample_id] = row
            remaining.discard(sample_id)
    if remaining:
        raise GenerationAuditError(f"SENTINEL_ROWS_MISSING={sorted(remaining)[:8]}")
    return result


def load_examples(args: argparse.Namespace) -> tuple[list[Any], list[Any], dict[str, Any]]:
    novel_sentinel = json.loads((args.pilot_freeze / "NOVEL_VALIDATION_SENTINEL.json").read_text(encoding="utf-8"))
    replay_sentinel = json.loads((args.pilot_freeze / "REPLAY_RETENTION_SENTINEL.json").read_text(encoding="utf-8"))
    if novel_sentinel.get("status") != "FROZEN" or novel_sentinel.get("episode_count") != 128:
        raise GenerationAuditError("NOVEL_SENTINEL_NOT_FROZEN_128")
    if replay_sentinel.get("status") != "FROZEN" or replay_sentinel.get("episode_count") != 64:
        raise GenerationAuditError("REPLAY_SENTINEL_NOT_FROZEN_64")
    if novel_sentinel.get("selection_sha256") != NOVEL_SENTINEL_SHA256:
        raise GenerationAuditError("NOVEL_SENTINEL_IDENTITY_MISMATCH")
    if replay_sentinel.get("selection_sha256") != REPLAY_SENTINEL_SHA256:
        raise GenerationAuditError("REPLAY_SENTINEL_IDENTITY_MISMATCH")
    novel_meta = novel_sentinel["episodes"]
    replay_meta = replay_sentinel["episodes"]
    novel_rows = selected_rows(sorted(args.novel_validation_root.glob("*.parquet")), {str(row["sample_id"]) for row in novel_meta})
    replay_rows = selected_rows([args.replay_shard], {str(row["sample_id"]) for row in replay_meta})
    novel = [build_generation_example(novel_rows[str(meta["sample_id"])], meta) for meta in novel_meta]
    replay = [build_generation_example(replay_rows[str(meta["sample_id"])], meta) for meta in replay_meta]
    audits = [context_audit_record(item, novel_rows[item.sample_id]) for item in novel]
    audits += [context_audit_record(item, replay_rows[item.sample_id]) for item in replay]
    return novel, replay, {
        "status": "PASS" if all(
            row["prompt_is_exact_source_prefix"]
            and row["target_gold_is_exact_disjoint_source_suffix"]
            and row["target_eos_at_boundary"]
            and row["no_later_tokens"]
            and row["gold_target_tokens_in_inference_input_by_source_position"] is False
            for row in audits
        ) else "FAIL",
        "target_policy": "final assistant turn; earlier complete turns retained; final Gold response and later tokens excluded",
        "record_count": len(audits),
        "novel_records": len(novel),
        "replay_records": len(replay),
        "records": audits,
    }


def checkpoint_identity(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer
    from inference.nvarc_native import native_tokenizer_preflight

    model_config = AutoConfig.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=False, use_fast=True)
    tokenizer_check = native_tokenizer_preflight(args.model_path, args.native_config_dir)
    base_files = sorted(
        path for path in args.model_path.iterdir()
        if path.is_file() and (path.suffix in {".json", ".safetensors"} or path.name.endswith(".model"))
    )
    base_hashes = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in base_files}
    checkpoints: dict[str, Any] = {}
    for state in MODEL_STATES[1:]:
        directory = args.checkpoint_root / state
        if not directory.is_dir():
            raise GenerationAuditError(f"CHECKPOINT_UNAVAILABLE={directory}")
        config_path = directory / "adapter_config.json"
        weights = sorted(directory.glob("adapter_model*.safetensors"))
        if not config_path.is_file() or not weights:
            raise GenerationAuditError(f"CHECKPOINT_UNAVAILABLE={directory}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        targets = set(config.get("target_modules") or [])
        if int(config.get("r", -1)) != 64 or int(config.get("lora_alpha", -1)) != 32 or targets != EXPECTED_TARGET_MODULES:
            raise GenerationAuditError(f"CHECKPOINT_LORA_IDENTITY_MISMATCH={state}")
        files = sorted(path for path in directory.iterdir() if path.is_file())
        checkpoints[state] = {
            "path": str(directory),
            "adapter_config": config,
            "files": {path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256_file(path)} for path in files},
        }
    return {
        "status": "PASS",
        "source_commit": SOURCE_COMMIT,
        "base_model_path": str(args.model_path),
        "base_model_type": str(model_config.model_type),
        "base_model_vocab_size": int(model_config.vocab_size),
        "tokenizer_length": len(tokenizer),
        "tokenizer_preflight": tokenizer_check,
        "base_files": base_hashes,
        "checkpoints": checkpoints,
    }


def preflight(args: argparse.Namespace) -> int:
    if args.output.exists() and any(args.output.iterdir()):
        raise GenerationAuditError(f"NONEMPTY_OUTPUT_EXISTS={args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (args.novel_validation_root, args.replay_shard):
        lowered = str(path).replace("\\", "/").lower()
        if any(term in lowered for term in FORBIDDEN_PATH_TERMS):
            raise GenerationAuditError(f"FORBIDDEN_DATA_PATH={path}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"],
        cwd=Path(__file__).resolve().parents[1], check=False,
    ).returncode == 0
    if not ancestor:
        raise GenerationAuditError("SOURCE_COMMIT_NOT_ANCESTOR")
    identity = checkpoint_identity(args)
    novel, replay, context = load_examples(args)
    if context["status"] != "PASS":
        raise GenerationAuditError("GENERATION_CONTEXT_AUDIT_FAIL")
    atomic_json(args.output / "CHECKPOINT_IDENTITY_AUDIT.json", identity)
    atomic_json(args.output / "GENERATION_CONTEXT_AUDIT.json", context)
    atomic_json(args.output / "GENERATION_CONFIG.json", {
        "status": "FROZEN",
        "source_commit": SOURCE_COMMIT,
        "source_commit_is_ancestor": ancestor,
        "model_states": list(MODEL_STATES),
        "novel_count": len(novel),
        "replay_count": len(replay),
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS, "temperature": None},
        "context_window": CONTEXT_WINDOW,
        "target_policy": context["target_policy"],
        "optimizer_steps": 0,
        "backward_calls": 0,
        "optimizer_state_created": False,
    })
    atomic_json(args.output / "NO_TRAINING_AUDIT.json", {
        "status": "PASS", "optimizer_steps": 0, "backward_calls": 0, "optimizer_state_created": False,
    })
    print(json.dumps({"status": "PASS", "novel": len(novel), "replay": len(replay), "checkpoints": len(identity["checkpoints"])}))
    return 0


def load_model(model_path: Path, adapter: Path | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.config.use_cache = True
    return model


def generate_state(args: argparse.Namespace) -> int:
    import torch
    if args.state not in MODEL_STATES:
        raise GenerationAuditError(f"UNKNOWN_STATE={args.state}")
    novel, replay, context = load_examples(args)
    if context["status"] != "PASS":
        raise GenerationAuditError("GENERATION_CONTEXT_AUDIT_FAIL")
    adapter = None if args.state == "base" else args.checkpoint_root / args.state
    model = load_model(args.model_path, adapter)
    torch.cuda.reset_peak_memory_stats(0)
    records: list[dict[str, Any]] = []
    requested_batch_size = args.batch_size or 1
    if requested_batch_size < 1:
        raise GenerationAuditError("INVALID_GENERATION_BATCH_SIZE")
    max_batched_prompt_tokens = args.max_batched_prompt_tokens
    if max_batched_prompt_tokens < 1:
        raise GenerationAuditError("INVALID_MAX_BATCHED_PROMPT_TOKENS")
    oom_fallbacks: list[dict[str, Any]] = []
    started = time.perf_counter()
    for cohort, items in (("novel", novel), ("replay", replay)):
        pending = sorted(items, key=lambda example: (len(example.prompt_ids), example.sample_id))
        cursor = 0
        while cursor < len(pending):
            batch_count = min(requested_batch_size, len(pending) - cursor)
            while batch_count > 1:
                width = max(len(example.prompt_ids) for example in pending[cursor:cursor + batch_count])
                if width * batch_count <= max_batched_prompt_tokens:
                    break
                batch_count -= 1
            batch = pending[cursor:cursor + batch_count]
            while True:
                width = max(len(example.prompt_ids) for example in batch)
                prompt = torch.full((len(batch), width), 13, dtype=torch.long, device="cuda:0")
                mask = torch.zeros_like(prompt)
                for row_index, example in enumerate(batch):
                    length = len(example.prompt_ids)
                    prompt[row_index, width - length:] = torch.tensor(example.prompt_ids, dtype=torch.long, device="cuda:0")
                    mask[row_index, width - length:] = 1
                one_started = time.perf_counter()
                try:
                    with torch.inference_mode():
                        output = model.generate(
                            input_ids=prompt,
                            attention_mask=mask,
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            num_beams=1,
                            eos_token_id=15,
                            pad_token_id=13,
                            use_cache=True,
                        )
                    torch.cuda.synchronize(0)
                    batch_seconds = time.perf_counter() - one_started
                    break
                except torch.cuda.OutOfMemoryError:
                    failed_size = len(batch)
                    del prompt, mask
                    gc.collect()
                    torch.cuda.empty_cache()
                    if failed_size == 1:
                        raise
                    batch = batch[:max(1, failed_size // 2)]
                    oom_fallbacks.append({
                        "cohort": cohort,
                        "cursor": cursor,
                        "failed_batch_size": failed_size,
                        "fallback_batch_size": len(batch),
                    })
            for row_index, example in enumerate(batch):
                raw_suffix = tuple(int(value) for value in output[row_index, width:].detach().cpu().tolist())
                if 15 in raw_suffix:
                    suffix = raw_suffix[:raw_suffix.index(15) + 1]
                else:
                    suffix = raw_suffix
                parsed = parse_generated_tokens(suffix, example.gold_grid, hit_max_new_tokens=len(suffix) >= MAX_NEW_TOKENS and 15 not in suffix)
                record = score_record(example, parsed) | {
                    "checkpoint": args.state,
                    "cohort": cohort,
                    "prompt_tokens": len(example.prompt_ids),
                    "generated_token_ids": list(suffix),
                    "generated_token_count": len(suffix),
                    "terminated_by_eos": parsed.terminated_by_eos,
                    "generation_seconds": batch_seconds,
                    "requested_batch_size": requested_batch_size,
                    "actual_batch_size": len(batch),
                    "batch_prompt_width": width,
                    "gold_grid": [list(row) for row in example.gold_grid],
                    "predicted_grid": None if parsed.grid is None else [list(row) for row in parsed.grid],
                }
                records.append(record)
                print(json.dumps({"event": "GENERATION_PROGRESS", "checkpoint": args.state, "cohort": cohort, "completed": len(records), "total": len(novel) + len(replay), "sample_id": example.sample_id, "classification": parsed.classification, "exact": record["exact_grid"], "actual_batch_size": len(batch)}), flush=True)
            atomic_json(args.output / f"{args.state}.partial.json", {
                "status": "INCOMPLETE", "checkpoint": args.state, "completed": len(records), "total": len(novel) + len(replay), "requested_batch_size": requested_batch_size, "max_batched_prompt_tokens": max_batched_prompt_tokens, "oom_fallbacks": oom_fallbacks, "records": records,
            })
            cursor += len(batch)
            del output, prompt, mask
    payload = {
        "status": "COMPLETE",
        "checkpoint": args.state,
        "adapter": None if adapter is None else str(adapter),
        "record_count": len(records),
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "requested_batch_size": requested_batch_size,
        "max_batched_prompt_tokens": max_batched_prompt_tokens,
        "oom_fallbacks": oom_fallbacks,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "optimizer_state_created": False,
        "records": records,
    }
    atomic_json(args.output / f"{args.state}.json", payload)
    partial = args.output / f"{args.state}.partial.json"
    if partial.exists():
        partial.unlink()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def calibration(args: argparse.Namespace) -> int:
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    novel, replay, context = load_examples(args)
    if context["status"] != "PASS":
        raise GenerationAuditError("GENERATION_CONTEXT_AUDIT_FAIL")
    selected = select_calibration_examples(novel, replay)

    class FirstTokenTiming(LogitsProcessor):
        def __init__(self, started: float) -> None:
            self.started = started
            self.first_call: float | None = None

        def __call__(self, input_ids, scores):
            if self.first_call is None:
                self.first_call = time.perf_counter()
            return scores

    load_started = time.perf_counter()
    model = load_model(args.model_path, args.checkpoint_root / "tokens_2000000")
    model_load_seconds = time.perf_counter() - load_started
    rows: list[dict[str, Any]] = []
    for cohort, example, stratum in selected:
        prompt = torch.tensor([example.prompt_ids], dtype=torch.long, device="cuda:0")
        mask = torch.ones_like(prompt)
        started = time.perf_counter()
        timing = FirstTokenTiming(started)
        with torch.inference_mode():
            output = model.generate(
                input_ids=prompt,
                attention_mask=mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                eos_token_id=15,
                pad_token_id=13,
                use_cache=True,
                logits_processor=LogitsProcessorList([timing]),
            )
        ended = time.perf_counter()
        suffix = tuple(int(value) for value in output[0, prompt.shape[-1]:].detach().cpu().tolist())
        if timing.first_call is None:
            raise GenerationAuditError("CALIBRATION_TIMING_HOOK_NOT_CALLED")
        prefill_seconds = timing.first_call - started
        decode_seconds = ended - timing.first_call
        decode_steps = max(0, len(suffix) - 1)
        parsed = parse_generated_tokens(suffix, example.gold_grid, hit_max_new_tokens=len(suffix) >= MAX_NEW_TOKENS and 15 not in suffix)
        rows.append({
            "cohort": cohort,
            "stratum": stratum,
            "sample_id": example.sample_id,
            "source": example.source,
            "family": example.family,
            "prompt_tokens": len(example.prompt_ids),
            "generated_tokens": len(suffix),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_steps_after_first_token": decode_steps,
            "decode_tokens_per_second": None if decode_steps == 0 or decode_seconds <= 0 else decode_steps / decode_seconds,
            "total_seconds": ended - started,
            "parse_classification": parsed.classification,
            "exact_grid": score_record(example, parsed)["exact_grid"],
        })
        print(json.dumps({"event": "CALIBRATION_PROGRESS", "completed": len(rows), "total": 20, **{key: rows[-1][key] for key in ("sample_id", "cohort", "stratum", "prompt_tokens", "generated_tokens", "total_seconds")}}), flush=True)
        del output, prompt, mask
    novel_mean = sum(row["total_seconds"] for row in rows if row["cohort"] == "novel") / 12
    replay_mean = sum(row["total_seconds"] for row in rows if row["cohort"] == "replay") / 8
    estimated_generation_seconds = 5 * (128 * novel_mean + 64 * replay_mean)
    estimated_total_seconds = estimated_generation_seconds + 5 * model_load_seconds
    payload = {
        "status": "STOP_RUNTIME_EXCEEDS_6H" if estimated_total_seconds > 6 * 3600 else "PASS_FULL_AUDIT_AUTHORIZED",
        "checkpoint": "tokens_2000000",
        "selection_policy": "Novel min/median/max per family; Replay min/max per source",
        "representative_generation_count": len(rows),
        "novel_family_count": len({row["family"] for row in rows if row["cohort"] == "novel"}),
        "replay_source_count": len({row["source"] for row in rows if row["cohort"] == "replay"}),
        "model_load_seconds": model_load_seconds,
        "mean_novel_seconds": novel_mean,
        "mean_replay_seconds": replay_mean,
        "mean_seconds_per_generation_unweighted": sum(row["total_seconds"] for row in rows) / len(rows),
        "estimated_full_generation_seconds": estimated_generation_seconds,
        "estimated_full_total_seconds": estimated_total_seconds,
        "estimated_full_total_hours": estimated_total_seconds / 3600,
        "stop_threshold_hours": 6,
        "timing_definition": "prefill ends at first unchanged logits-processor call; decode counts steps after first-token logits",
        "generation_semantics_unchanged": True,
        "rows": rows,
    }
    atomic_json(args.output / "CALIBRATION_20.json", payload)
    print(json.dumps({key: payload[key] for key in ("status", "estimated_full_total_seconds", "estimated_full_total_hours")}, sort_keys=True))
    return 2 if payload["status"] == "STOP_RUNTIME_EXCEEDS_6H" else 0


def batch_capacity(args: argparse.Namespace) -> int:
    """Measure inference capacity without changing frozen generation semantics."""
    import torch

    if args.batch_size is None or args.batch_size < 1:
        raise GenerationAuditError("BATCH_SIZE_REQUIRED")
    novel, replay, context = load_examples(args)
    if context["status"] != "PASS":
        raise GenerationAuditError("GENERATION_CONTEXT_AUDIT_FAIL")
    examples = {example.sample_id: example for example in novel + replay}
    if args.sample_id not in examples:
        raise GenerationAuditError(f"CAPACITY_SAMPLE_NOT_FOUND={args.sample_id}")
    example = examples[args.sample_id]
    adapter = args.checkpoint_root / "tokens_2000000"
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / f"BATCH_{args.batch_size}_CAPACITY.json"

    model = load_model(args.model_path, adapter)
    prompt = torch.tensor(
        [example.prompt_ids] * args.batch_size,
        dtype=torch.long,
        device="cuda:0",
    )
    mask = torch.ones_like(prompt)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=prompt,
                attention_mask=mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                eos_token_id=15,
                pad_token_id=13,
                use_cache=True,
            )
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - started
        suffixes = [
            tuple(int(value) for value in row[prompt.shape[-1]:].detach().cpu().tolist())
            for row in output
        ]
        generated = [len(suffix) for suffix in suffixes]
        payload = {
            "status": "PASS",
            "checkpoint": "tokens_2000000",
            "batch_size": args.batch_size,
            "sample_id": example.sample_id,
            "selection": "frozen calibration lower-median prompt; replicated for hardware capacity only",
            "prompt_tokens_per_sequence": len(example.prompt_ids),
            "generated_tokens_per_sequence": generated,
            "total_generated_tokens": sum(generated),
            "elapsed_seconds": elapsed,
            "generated_tokens_per_second": sum(generated) / elapsed,
            "all_batch_outputs_identical": len(set(suffixes)) == 1,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "generation_semantics": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": MAX_NEW_TOKENS,
                "eos_token_id": 15,
                "pad_token_id": 13,
                "use_cache": True,
            },
            "scientific_audit_metric": False,
            "optimizer_steps": 0,
            "gold_accessed": False,
        }
    except torch.cuda.OutOfMemoryError as exc:
        elapsed = time.perf_counter() - started
        payload = {
            "status": "OOM",
            "checkpoint": "tokens_2000000",
            "batch_size": args.batch_size,
            "sample_id": example.sample_id,
            "prompt_tokens_per_sequence": len(example.prompt_ids),
            "elapsed_seconds": elapsed,
            "exception_type": type(exc).__name__,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "generation_semantics_unchanged": True,
            "scientific_audit_metric": False,
            "optimizer_steps": 0,
            "gold_accessed": False,
        }
    atomic_json(receipt_path, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def batch_parity(args: argparse.Namespace) -> int:
    """Require token-exact parity before using batched scientific generation."""
    import torch

    novel, replay, context = load_examples(args)
    if context["status"] != "PASS":
        raise GenerationAuditError("GENERATION_CONTEXT_AUDIT_FAIL")
    selected = [example for _, example, _ in select_calibration_examples(novel, replay)]
    examples = {example.sample_id: example for example in novel + replay}
    if args.sample_id not in examples:
        raise GenerationAuditError(f"CAPACITY_SAMPLE_NOT_FOUND={args.sample_id}")
    anchor = examples[args.sample_id]
    model = load_model(args.model_path, args.checkpoint_root / "tokens_2000000")
    args.output.mkdir(parents=True, exist_ok=True)

    def generate_batch(batch: list[Any]) -> list[tuple[int, ...]]:
        width = max(len(example.prompt_ids) for example in batch)
        prompt = torch.full((len(batch), width), 13, dtype=torch.long, device="cuda:0")
        mask = torch.zeros_like(prompt)
        for row_index, example in enumerate(batch):
            length = len(example.prompt_ids)
            prompt[row_index, width - length:] = torch.tensor(example.prompt_ids, dtype=torch.long, device="cuda:0")
            mask[row_index, width - length:] = 1
        with torch.inference_mode():
            output = model.generate(
                input_ids=prompt,
                attention_mask=mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                eos_token_id=15,
                pad_token_id=13,
                use_cache=True,
            )
        torch.cuda.synchronize(0)
        suffixes = []
        for row in output:
            raw = tuple(int(value) for value in row[width:].detach().cpu().tolist())
            suffixes.append(raw[:raw.index(15) + 1] if 15 in raw else raw)
        del output, prompt, mask
        return suffixes

    def digest(tokens: tuple[int, ...]) -> str:
        return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")).hexdigest()

    def mismatch(left: tuple[int, ...], right: tuple[int, ...]) -> dict[str, Any] | None:
        if left == right:
            return None
        shared = min(len(left), len(right))
        first = next((index for index in range(shared) if left[index] != right[index]), shared)
        return {
            "first_divergence": first,
            "batch1_length": len(left),
            "batched_length": len(right),
            "batch1_sha256": digest(left),
            "batched_sha256": digest(right),
        }

    singles: dict[str, tuple[int, ...]] = {}
    for index, example in enumerate(selected, 1):
        singles[example.sample_id] = generate_batch([example])[0]
        print(json.dumps({"event": "PARITY_BATCH1_PROGRESS", "completed": index, "total": len(selected), "sample_id": example.sample_id}), flush=True)

    anchor_batch32 = generate_batch([anchor] * 32)
    anchor_mismatches = [mismatch(singles[anchor.sample_id], tokens) for tokens in anchor_batch32]
    anchor_mismatches = [item for item in anchor_mismatches if item is not None]

    batched: dict[str, tuple[int, ...]] = {}
    actual_batch_sizes: list[int] = []
    pending = sorted(selected, key=lambda example: (len(example.prompt_ids), example.sample_id))
    cursor = 0
    while cursor < len(pending):
        count = min(32, len(pending) - cursor)
        while count > 1:
            width = max(len(example.prompt_ids) for example in pending[cursor:cursor + count])
            if width * count <= args.max_batched_prompt_tokens:
                break
            count -= 1
        batch = pending[cursor:cursor + count]
        suffixes = generate_batch(batch)
        actual_batch_sizes.append(len(batch))
        for example, tokens in zip(batch, suffixes, strict=True):
            batched[example.sample_id] = tokens
        cursor += len(batch)

    representative_mismatches = []
    for example in selected:
        detail = mismatch(singles[example.sample_id], batched[example.sample_id])
        if detail is not None:
            representative_mismatches.append({"sample_id": example.sample_id, **detail})
    status = "PASS" if not anchor_mismatches and not representative_mismatches else "FAIL"
    payload = {
        "status": status,
        "checkpoint": "tokens_2000000",
        "comparison": "batch1_vs_max_batch32_token_exact",
        "anchor_sample_id": anchor.sample_id,
        "anchor_actual_batch_size": 32,
        "anchor_comparisons": 32,
        "anchor_mismatch_count": len(anchor_mismatches),
        "anchor_mismatches": anchor_mismatches,
        "representative_count": len(selected),
        "representative_actual_batch_sizes": actual_batch_sizes,
        "representative_mismatch_count": len(representative_mismatches),
        "representative_mismatches": representative_mismatches,
        "max_batched_prompt_tokens": args.max_batched_prompt_tokens,
        "generation_semantics_unchanged": True,
        "gold_used_for_generation_or_comparison": False,
        "optimizer_steps": 0,
    }
    atomic_json(args.output / "BATCH1_VS_BATCH32_PARITY.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if status == "PASS" else 4


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return None if denominator == 0 else numerator / denominator


def _next_phase(novel_class: str, replay_delta: float, saturation: str, final_valid_rate: float) -> str:
    if novel_class == "REGRESSION":
        return "F_STOP_DUE_TO_REGRESSION"
    if replay_delta < -0.05:
        return "D_CHANGE_CURRICULUM"
    if novel_class == "NO_CLEAR_GENERATIVE_GAIN":
        return "E_INVESTIGATE_GENERATION_FORMAT_GAP" if final_valid_rate < 0.90 else "C_CHANGE_TRAINING_OBJECTIVE"
    if saturation == "GENERATION_SATURATION_BEFORE_2M":
        return "B_STOP_AT_2M_SATURATED"
    return "A_EXTEND_TO_5M"


def finalize(args: argparse.Namespace) -> int:
    points: list[dict[str, Any]] = []
    raw_hashes: dict[str, Any] = {}
    for state in MODEL_STATES:
        path = args.output / f"{state}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE" or payload.get("record_count") != 192:
            raise GenerationAuditError(f"GENERATION_STATE_INCOMPLETE={state}")
        if payload.get("optimizer_steps") != 0 or payload.get("backward_calls") != 0 or payload.get("optimizer_state_created") is not False:
            raise GenerationAuditError(f"TRAINING_ACTIVITY_DETECTED={state}")
        novel_records = [row for row in payload["records"] if row["cohort"] == "novel"]
        replay_records = [row for row in payload["records"] if row["cohort"] == "replay"]
        points.append({
            "checkpoint": state,
            "novel": aggregate_generation(novel_records),
            "replay": aggregate_generation(replay_records),
            "wall_seconds": payload["wall_seconds"],
            "peak_vram_allocated_bytes": payload["peak_vram_allocated_bytes"],
        })
        raw_hashes[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    losses = json.loads((args.pilot_freeze / "LEARNING_CURVE.json").read_text(encoding="utf-8"))["points"]
    loss_by_state = {
        "base": losses[0], "tokens_500000": losses[1], "tokens_1000000": losses[2],
        "tokens_1500000": losses[3], "tokens_2000000": losses[4],
    }
    comparison = []
    for point in points:
        state = point["checkpoint"]
        comparison.append({
            "checkpoint": state,
            "novel_teacher_forced_macro_loss": loss_by_state[state]["novel_macro_family_loss"],
            "novel_exact_generation_accuracy": point["novel"]["macro_family_exact_grid_accuracy"],
            "replay_teacher_forced_macro_loss": loss_by_state[state]["replay_macro_family_loss"],
            "replay_exact_generation_accuracy": point["replay"]["macro_family_exact_grid_accuracy"],
        })
    correlations = {
        "novel_loss_vs_macro_exact_pearson": _pearson([row["novel_teacher_forced_macro_loss"] for row in comparison], [row["novel_exact_generation_accuracy"] for row in comparison]),
        "replay_loss_vs_macro_exact_pearson": _pearson([row["replay_teacher_forced_macro_loss"] for row in comparison], [row["replay_exact_generation_accuracy"] for row in comparison]),
        "descriptive_only_no_causal_claim": True,
    }
    base, final = points[0], points[-1]
    novel_class = classify_novel_gain(base["novel"]["macro_family_exact_grid_accuracy"], final["novel"]["macro_family_exact_grid_accuracy"])
    saturation = saturation_status(points)
    replay_delta = final["replay"]["macro_family_exact_grid_accuracy"] - base["replay"]["macro_family_exact_grid_accuracy"]
    replay_status = "REPLAY_GENERATION_PRESERVED_OR_IMPROVED" if replay_delta >= -0.03 else "REPLAY_GENERATION_REGRESSION"
    best = best_checkpoint(points)
    recommendation = _next_phase(novel_class, replay_delta, saturation, final["novel"]["valid_format_rate"])
    artifact = args.artifact
    if artifact is None:
        raise GenerationAuditError("ARTIFACT_PATH_REQUIRED")
    artifact.mkdir(parents=True, exist_ok=True)
    pre_identity = json.loads((args.output / "CHECKPOINT_IDENTITY_AUDIT.json").read_text(encoding="utf-8"))
    post_identity = checkpoint_identity(args)
    if pre_identity.get("base_files") != post_identity.get("base_files") or pre_identity.get("checkpoints") != post_identity.get("checkpoints"):
        raise GenerationAuditError("CHECKPOINT_IDENTITY_CHANGED_DURING_AUDIT")
    pre_identity["post_generation_identity_verification"] = "PASS"
    atomic_json(args.output / "CHECKPOINT_IDENTITY_AUDIT.json", pre_identity)
    for name in ("CHECKPOINT_IDENTITY_AUDIT.json", "GENERATION_CONTEXT_AUDIT.json", "GENERATION_CONFIG.json", "NO_TRAINING_AUDIT.json"):
        source = args.output / name
        atomic_text(artifact / name, source.read_text(encoding="utf-8"))
    atomic_json(artifact / "NOVEL_GENERATION_RESULTS.json", {"status": "PASS", "points": [{"checkpoint": row["checkpoint"], **row["novel"]} for row in points]})
    atomic_json(artifact / "REPLAY_GENERATION_RESULTS.json", {"status": "PASS", "points": [{"checkpoint": row["checkpoint"], **row["replay"]} for row in points]})
    atomic_json(artifact / "LOSS_VS_GENERATION_COMPARISON.json", {"status": "PASS", "points": comparison, "correlations": correlations})
    with (artifact / "GENERATION_LEARNING_CURVE.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = list(comparison[0])
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(comparison)
    atomic_json(artifact / "GENERATION_DATA_ACCESS_AUDIT.json", {
        "status": "PASS", "novel_validation_accessed": True, "replay_retention_accessed": True,
        "novel_holdout_accessed": False, "eval60_accessed": False, "kaggle_gold_accessed": False,
    })
    atomic_json(artifact / "GENERATION_PREDICTION_HASHES.json", {"status": "PASS", "raw_predictions_git_ignored": True, "files": raw_hashes})
    incremental = []
    for previous, current in zip(points, points[1:], strict=True):
        incremental.append({"from": previous["checkpoint"], "to": current["checkpoint"], "novel_macro_exact_delta": current["novel"]["macro_family_exact_grid_accuracy"] - previous["novel"]["macro_family_exact_grid_accuracy"]})
    gate = {
        "status": novel_class,
        "NOVEL_GENERATION_RESULT": novel_class,
        "REPLAY_GENERATION_RETENTION": replay_status,
        "replay_macro_exact_delta_vs_base": replay_delta,
        "GENERATION_SATURATION_STATUS": saturation,
        "incremental_novel_macro_exact_gain": incremental,
        "best_generation_checkpoint": best["checkpoint"],
        "next_phase_recommendation": recommendation,
        "diagnostic_thresholds_not_significance_claims": True,
    }
    atomic_json(artifact / "GENERATION_CAPABILITY_GATE.json", gate)
    atomic_text(artifact / "REPORT.md", "\n".join([
        "# Generation Capability Audit V1", "", f"- status: {novel_class}",
        f"- best generation checkpoint: {best['checkpoint']}", f"- replay retention: {replay_status}",
        f"- saturation: {saturation}", f"- recommendation: {recommendation}",
        "- Novel Holdout accessed: false", "- Eval60 accessed: false", "- Kaggle Gold accessed: false",
        "- optimizer steps: 0", "",
    ]))
    hashes = {}
    for path in sorted(artifact.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.json":
            hashes[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(artifact / "SHA256SUMS.json", {"status": "PASS", "files": hashes})
    print(json.dumps(gate, sort_keys=True))
    return 0


def controller(args: argparse.Namespace) -> int:
    for state in MODEL_STATES:
        command = [
            sys.executable, str(Path(__file__).resolve()), "--mode", "generate", "--state", state,
            "--output", str(args.output), "--pilot-freeze", str(args.pilot_freeze),
            "--model-path", str(args.model_path), "--checkpoint-root", str(args.checkpoint_root),
            "--novel-validation-root", str(args.novel_validation_root), "--replay-shard", str(args.replay_shard),
            "--native-config-dir", str(args.native_config_dir),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise GenerationAuditError(f"GENERATION_WORKER_FAILED={state}:{completed.returncode}")
    return finalize(args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("preflight", "calibration", "batch-capacity", "batch-parity", "generate", "finalize", "controller"), required=True)
    p.add_argument("--state")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-batched-prompt-tokens", type=int, default=21568)
    p.add_argument("--sample-id", default="1d_arc:1d_padded_fill_3:native-source-episode-v1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--artifact", type=Path)
    p.add_argument("--pilot-freeze", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--novel-validation-root", type=Path, required=True)
    p.add_argument("--replay-shard", type=Path, required=True)
    p.add_argument("--native-config-dir", type=Path, required=True)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.mode == "preflight":
        return preflight(args)
    if args.mode == "generate":
        return generate_state(args)
    if args.mode == "calibration":
        return calibration(args)
    if args.mode == "batch-capacity":
        return batch_capacity(args)
    if args.mode == "batch-parity":
        return batch_parity(args)
    if args.mode == "finalize":
        return finalize(args)
    return controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
