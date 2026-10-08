from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from capability_pilot_2m_v1.pilot import (
    CHECKPOINT_THRESHOLDS,
    CONTEXT,
    DATASET_FINGERPRINT,
    REPLAY_POOL,
    REPLAY_SHA256,
    SOURCE_COMMIT,
    TOKEN_BUDGET,
    PilotGateError,
    aggregate_losses,
    build_training_schedule,
    checkpoint_crossings,
    classify_results,
    distribution_audit,
    frozen_training_config,
    next_phase_recommendation,
    relative_changes,
    select_best_safe_checkpoint,
    select_replay_retention_sentinel,
    select_validation_sentinel,
    validate_allowed_data_paths,
)
from gpu_benchmark_3090_v1.benchmark import assert_lora_partition, require_single_cuda_device, sha256_file
from gpu_throughput_opt_3090_v1.benchmark import CONFIG_D


EXPECTED_TRAINABLE = 132_120_576


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


def write_optimizer_accounting(path: Path | None, value: dict[str, Any]) -> None:
    """Publish durable optimizer-only accounting for the launcher.

    This path is unset for historical/replay execution.  It is deliberately
    written only when an optimizer step is active or complete, so imports,
    model loading, validation, and wrapper work cannot become GPU charges.
    """
    if path is not None:
        atomic_json(path, value)


def load_metadata(paths: list[Path]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    columns = ["sample_id", "generator_family", "source", "split", "final_training_role", "sequence_length", "supervised_token_count"]
    rows: list[dict[str, Any]] = []
    for path in paths:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        loaded = pq.read_table(path, columns=[name for name in columns if name in available]).to_pylist()
        for row in loaded:
            row.setdefault("split", "train")
        rows.extend(loaded)
    return rows


def filtered_rows(paths: list[Path], wanted: set[str]) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq
    found: dict[str, dict[str, Any]] = {}
    remaining = set(wanted)
    for path in paths:
        if not remaining:
            break
        table = pq.read_table(path, columns=["sample_id", "input_ids", "labels"], filters=[("sample_id", "in", sorted(remaining))])
        for row in table.to_pylist():
            sample_id = str(row["sample_id"])
            found[sample_id] = row
            remaining.discard(sample_id)
    if remaining:
        raise PilotGateError(f"SELECTED_ROWS_MISSING={sorted(remaining)[:8]}")
    return found


def selection_content_hashes(payload: dict[str, Any], paths: list[Path]) -> None:
    wanted = {str(row["sample_id"]) for row in payload["episodes"]}
    raw = filtered_rows(paths, wanted)
    for row in payload["episodes"]:
        item = raw[str(row["sample_id"])]
        canonical = json.dumps({"input_ids": item["input_ids"], "labels": item["labels"]}, separators=(",", ":"))
        row["token_label_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    payload["selection_sha256"] = hashlib.sha256(
        json.dumps(payload["episodes"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare(args: argparse.Namespace) -> int:
    validate_allowed_data_paths(args.novel_train_root, args.novel_validation_root, args.replay_shard)
    if args.output.exists() and any(args.output.iterdir()):
        raise PilotGateError(f"NONEMPTY_OUTPUT_EXISTS={args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    if sha256_file(args.replay_shard) != REPLAY_SHA256:
        raise PilotGateError("REPLAY_SHA256_MISMATCH")
    fingerprint = json.loads(args.novel_fingerprint.read_text(encoding="utf-8"))
    if fingerprint.get("status") != "FROZEN" or fingerprint.get("fingerprint_sha256") != DATASET_FINGERPRINT:
        raise PilotGateError("NOVEL_FINGERPRINT_MISMATCH")
    gate = json.loads(args.scientific_gate.read_text(encoding="utf-8"))
    if gate.get("OFFICIAL_SYSTEMATICITY_PROTOCOL_READY") is not True or gate.get("GPU_BENCHMARK_READY") is not True:
        raise PilotGateError("SCIENTIFIC_GATE_NOT_READY")

    novel_paths = sorted(args.novel_train_root.glob("*.parquet"))
    validation_paths = sorted(args.novel_validation_root.glob("*.parquet"))
    if len(novel_paths) != 84 or len(validation_paths) != 9:
        raise PilotGateError(f"NOVEL_SHARD_COUNTS={len(novel_paths)},{len(validation_paths)}")
    novel = load_metadata(novel_paths)
    validation = load_metadata(validation_paths)
    replay = load_metadata([args.replay_shard])
    validation_sentinel = select_validation_sentinel(validation)
    replay_sentinel = select_replay_retention_sentinel(replay)
    selection_content_hashes(validation_sentinel, validation_paths)
    selection_content_hashes(replay_sentinel, [args.replay_shard])
    retention_ids = [str(row["sample_id"]) for row in replay_sentinel["episodes"]]
    schedule = build_training_schedule(novel, replay, retention_ids)
    if {row["split"] for row in novel} != {"train"}:
        raise PilotGateError("NOVEL_TRAIN_SPLIT_CONTAMINATION")
    if any(term in str(row.get("final_training_role", "")) for row in novel + replay for term in ("HOLDOUT", "HARD_EXCLUDE", "QUARANTINE")):
        raise PilotGateError("FORBIDDEN_ROLE_IN_LOADED_TRAINING_INPUT")
    checks = {
        "novel_holdout_rows": 0,
        "novel_validation_rows": 0,
        "hard_exclude_rows": 0,
        "quarantine_rows": 0,
        "retention_sentinel_rows_in_schedule": len(set(retention_ids) & {str(row["sample_id"]) for row in schedule["episodes"]}),
        "schedule_complete_optimizer_steps": len(schedule["episodes"]) % 4 == 0,
        "token_budget_reached": schedule["actual_transformer_tokens"] >= TOKEN_BUDGET,
    }
    status = "PASS" if all(value is True or value == 0 for value in checks.values()) else "FAIL"
    atomic_json(args.output / "TRAINING_CONFIG.json", frozen_training_config())
    atomic_json(args.output / "NOVEL_VALIDATION_SENTINEL.json", validation_sentinel)
    atomic_json(args.output / "REPLAY_RETENTION_SENTINEL.json", replay_sentinel)
    atomic_json(args.output / "TRAINING_SCHEDULE_2M.json", schedule)
    atomic_text(args.output / "TRAINING_SCHEDULE_2M.sha256", f"{sha256_file(args.output / 'TRAINING_SCHEDULE_2M.json')}  TRAINING_SCHEDULE_2M.json\n")
    atomic_json(args.output / "PRETRAIN_DATA_POLICY_AUDIT.json", {
        "status": status,
        "checks": checks,
        "novel_train_rows_inspected": len(novel),
        "novel_validation_rows_inspected_for_sentinel_only": len(validation),
        "replay_rows_inspected": len(replay),
        "configured_paths": {
            "novel_train": str(args.novel_train_root),
            "novel_validation": str(args.novel_validation_root),
            "replay": str(args.replay_shard),
        },
        "forbidden_paths_configurable": False,
    })
    if status != "PASS":
        raise PilotGateError("PRETRAIN_DATA_POLICY_FAIL")
    print(json.dumps({"status": "PASS", "steps": schedule["optimizer_steps"], "tokens": schedule["actual_transformer_tokens"]}))
    return 0


def collate_one(row: dict[str, Any], device):
    import torch
    ids = [int(value) for value in row["input_ids"]]
    labels = [int(value) for value in row["labels"]]
    if len(ids) > CONTEXT or len(ids) != len(labels):
        raise PilotGateError("CONTEXT_OR_LABEL_MISMATCH")
    return (
        torch.tensor([ids], dtype=torch.long, device=device),
        torch.tensor([labels], dtype=torch.long, device=device),
    )


def build_model(model_path: Path, adapter: Path | None = None):
    import random
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    random.seed(2_000_031)
    torch.manual_seed(2_000_031)
    torch.cuda.manual_seed_all(2_000_031)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    if adapter is None:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model = get_peft_model(model, LoraConfig(
            r=64, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
    else:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return model


def load_all_selected(args: argparse.Namespace, schedule: dict[str, Any], validation: dict[str, Any], retention: dict[str, Any]):
    novel_ids = {str(row["sample_id"]) for row in schedule["episodes"] if row["pool"] != REPLAY_POOL}
    replay_ids = {str(row["sample_id"]) for row in schedule["episodes"] if row["pool"] == REPLAY_POOL}
    validation_ids = {str(row["sample_id"]) for row in validation["episodes"]}
    retention_ids = {str(row["sample_id"]) for row in retention["episodes"]}
    novel = filtered_rows(sorted(args.novel_train_root.glob("*.parquet")), novel_ids)
    replay = filtered_rows([args.replay_shard], replay_ids | retention_ids)
    validation_raw = filtered_rows(sorted(args.novel_validation_root.glob("*.parquet")), validation_ids)
    train = {("NOVEL", key): value for key, value in novel.items()} | {("REPLAY", key): value for key, value in replay.items()}
    return train, validation_raw, replay


def evaluate(model, metadata: list[dict[str, Any]], raw: dict[str, dict[str, Any]], device) -> dict[str, Any]:
    import torch
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for meta in metadata:
            values = raw[str(meta["sample_id"])]
            input_ids, labels = collate_one(values, device)
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = float(outputs.loss.detach().float().cpu())
            supervised = sum(int(value) != -100 for value in values["labels"][1:])
            if not math.isfinite(loss) or supervised <= 0:
                raise PilotGateError("NONFINITE_EVALUATION")
            records.append({
                "sample_id": str(meta["sample_id"]), "source": str(meta["source"]),
                "family": str(meta["family"]), "loss": loss,
                "nll_sum": loss * supervised, "supervised_tokens": supervised,
            })
    return aggregate_losses(records)


def lora_parameter_norm(model) -> float:
    import torch
    squares = torch.zeros((), device="cuda:0", dtype=torch.float64)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" in name:
            squares += parameter.detach().double().square().sum()
    return float(squares.sqrt().cpu())


def gradient_norm(params) -> float:
    import torch
    squares = torch.zeros((), device="cuda:0", dtype=torch.float64)
    for parameter in params:
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise PilotGateError("NONFINITE_GRADIENT")
            squares += parameter.grad.detach().double().square().sum()
    return float(squares.sqrt().cpu())


def point(training_tokens: int, optimizer_steps: int, novel: dict[str, Any], replay: dict[str, Any], base_novel: float, base_replay: float, train: dict[str, Any] | None):
    changes = relative_changes(base_novel, novel["macro_family_average_loss"], base_replay, replay["macro_family_average_loss"])
    return {
        "training_tokens": training_tokens,
        "optimizer_steps": optimizer_steps,
        "novel": novel,
        "replay": replay,
        "novel_macro_family_loss": novel["macro_family_average_loss"],
        "novel_source_macro_loss": novel["source_macro_loss"],
        "novel_micro_loss": novel["micro_average_loss"],
        "replay_macro_family_loss": replay["macro_family_average_loss"],
        "replay_source_macro_loss": replay["source_macro_loss"],
        "replay_micro_loss": replay["micro_average_loss"],
        **changes,
        "training": train,
    }


def gpu_identity(output: Path) -> None:
    import importlib.metadata
    import torch
    require_single_cuda_device(torch.cuda.device_count(), torch.cuda.get_device_name(0))
    if not torch.cuda.is_bf16_supported():
        raise PilotGateError("BF16_UNAVAILABLE")
    atomic_json(output / "GPU_RUNTIME_PREFLIGHT.json", {
        "status": "PASS", "gpu": torch.cuda.get_device_name(0), "device_count": torch.cuda.device_count(),
        "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "bf16_supported": True, "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"), "peft": importlib.metadata.version("peft"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"), "python": platform.python_version(),
    })


def train_worker(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import bitsandbytes as bnb
    import torch
    validate_allowed_data_paths(args.novel_train_root, args.novel_validation_root, args.replay_shard)
    args.runtime.mkdir(parents=True, exist_ok=False)
    args.checkpoints.mkdir(parents=True, exist_ok=False)
    gpu_identity(args.runtime)
    schedule = json.loads(args.freeze.joinpath("TRAINING_SCHEDULE_2M.json").read_text(encoding="utf-8"))
    validation = json.loads(args.freeze.joinpath("NOVEL_VALIDATION_SENTINEL.json").read_text(encoding="utf-8"))
    retention = json.loads(args.freeze.joinpath("REPLAY_RETENTION_SENTINEL.json").read_text(encoding="utf-8"))
    if sha256_file(args.replay_shard) != REPLAY_SHA256:
        raise PilotGateError("REPLAY_SHA256_MISMATCH")
    train_rows, validation_raw, replay_raw = load_all_selected(args, schedule, validation, retention)
    model = build_model(args.model_path)
    partition = assert_lora_partition(model.named_parameters())
    if partition["trainable_parameters"] != EXPECTED_TRAINABLE:
        raise PilotGateError("TRAINABLE_PARAMETER_COUNT_MISMATCH")
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    device = torch.device("cuda:0")
    disable = model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()
    with disable:
        base_novel_metrics = evaluate(model, validation["episodes"], validation_raw, device)
        base_replay_metrics = evaluate(model, retention["episodes"], replay_raw, device)
    baseline = {
        "status": "PASS", "model_state": "SFT139_BASE_WITH_LORA_DISABLED_BEFORE_FIRST_UPDATE",
        "novel_validation": base_novel_metrics, "replay_retention": base_replay_metrics,
    }
    atomic_json(args.runtime / "BASELINE_CAPABILITY_METRICS.json", baseline)
    points = [point(0, 0, base_novel_metrics, base_replay_metrics,
                    base_novel_metrics["macro_family_average_loss"], base_replay_metrics["macro_family_average_loss"], None)]
    optimizer = bnb.optim.PagedAdamW8bit(params, lr=5e-5)
    model.train()
    torch.cuda.reset_peak_memory_stats(0)
    trace: list[dict[str, Any]] = []
    tokens = supervised_tokens = 0
    train_seconds = 0.0
    accounting_raw = os.environ.get("ARC2_GPU_ACCOUNTING_STATE")
    accounting_path = Path(accounting_raw) if accounting_raw else None
    accounting = {"schema_version": 1, "status": "READY", "completed_optimizer_training_seconds": 0.0, "last_completed_optimizer_step": 0, "active_optimizer_step_started_monotonic_ns": None}
    run_started = time.perf_counter()
    last_group_loss = None
    for step_index in range(schedule["optimizer_steps"]):
        group_meta = schedule["episodes"][step_index * 4:(step_index + 1) * 4]
        group = []
        for meta in group_meta:
            key = "REPLAY" if meta["pool"] == REPLAY_POOL else "NOVEL"
            group.append((meta, train_rows[(key, str(meta["sample_id"]))]))
        group_supervised = [sum(int(value) != -100 for value in raw["labels"][1:]) for _, raw in group]
        total_group_supervised = sum(group_supervised)
        if total_group_supervised <= 0:
            raise PilotGateError("NO_SUPERVISED_TOKENS")
        lr = 5e-5 * min(1.0, (step_index + 1) / 3.0)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        step_monotonic_started = time.monotonic_ns()
        accounting.update({"status": "ACTIVE_OPTIMIZER_STEP", "active_optimizer_step_started_monotonic_ns": step_monotonic_started})
        write_optimizer_accounting(accounting_path, accounting)
        group_loss = 0.0
        previous_tokens = tokens
        for (meta, raw), count in zip(group, group_supervised):
            input_ids, labels = collate_one(raw, device)
            loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
            if not bool(torch.isfinite(loss)):
                raise PilotGateError("NONFINITE_LOSS")
            weight = count / total_group_supervised
            (loss * weight).backward()
            group_loss += float(loss.detach().float().cpu()) * weight
            tokens += int(meta["sequence_length"])
            supervised_tokens += count
        grad_norm = gradient_norm(params)
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_started
        train_seconds += step_seconds
        accounting.update({"status": "READY", "completed_optimizer_training_seconds": train_seconds, "last_completed_optimizer_step": step_index + 1, "active_optimizer_step_started_monotonic_ns": None})
        write_optimizer_accounting(accounting_path, accounting)
        last_group_loss = group_loss
        trace.append({
            "optimizer_step": step_index + 1, "processed_tokens": tokens,
            "supervised_tokens": supervised_tokens, "training_loss": group_loss,
            "gradient_norm": grad_norm, "learning_rate": lr,
            "lora_parameter_norm": lora_parameter_norm(model),
            "step_seconds": step_seconds, "tokens_per_sec": sum(int(x["sequence_length"]) for x in group_meta) / step_seconds,
        })
        for threshold in checkpoint_crossings(previous_tokens, tokens):
            checkpoint = args.checkpoints / f"tokens_{threshold}"
            checkpoint.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(checkpoint)
            novel_metrics = evaluate(model, validation["episodes"], validation_raw, device)
            replay_metrics = evaluate(model, retention["episodes"], replay_raw, device)
            current = point(tokens, step_index + 1, novel_metrics, replay_metrics,
                            base_novel_metrics["macro_family_average_loss"], base_replay_metrics["macro_family_average_loss"], trace[-1])
            current["checkpoint_threshold"] = threshold
            current["checkpoint_path"] = str(checkpoint)
            points.append(current)
            atomic_json(args.runtime / f"EVAL_{threshold}.json", current)
            if current["replay_loss_change"] > 0.50:
                raise PilotGateError("EMERGENCY_DIVERGENCE_STOP")
            model.train()
        atomic_json(args.runtime / "TRAINING_PROGRESS.json", {
            "status": "RUNNING", "optimizer_steps": step_index + 1, "actual_transformer_tokens": tokens,
            "actual_supervised_tokens": supervised_tokens, "last_training_loss": group_loss,
        })
    wall = time.perf_counter() - run_started
    if tokens != schedule["actual_transformer_tokens"]:
        raise PilotGateError("REALIZED_TOKEN_COUNT_MISMATCH")
    result = {
        "status": "PASS", "actual_transformer_tokens": tokens, "actual_supervised_tokens": supervised_tokens,
        "optimizer_steps": schedule["optimizer_steps"], "training_wall_seconds": train_seconds,
        "total_wall_seconds": wall, "average_training_tokens_per_sec": tokens / train_seconds,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "finite_losses": all(math.isfinite(float(row["training_loss"])) for row in trace),
        "finite_gradients": all(math.isfinite(float(row["gradient_norm"])) for row in trace),
        "points": points, "trace": trace, "parameter_partition": partition,
    }
    atomic_json(args.runtime / "TRAINING_RESULT.json", result)
    write_optimizer_accounting(accounting_path, {**accounting, "status": "COMPLETE"})
    atomic_json(args.runtime / "TRAINING_PROGRESS.json", {"status": "COMPLETE", **{key: result[key] for key in ("actual_transformer_tokens", "actual_supervised_tokens", "optimizer_steps")}})
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def reload_worker(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    model = build_model(args.model_path, args.adapter)
    raw = filtered_rows(sorted(args.novel_validation_root.glob("*.parquet")), {args.sample_id})[args.sample_id]
    model.eval()
    with torch.inference_mode():
        input_ids, labels = collate_one(raw, torch.device("cuda:0"))
        loss = float(model(input_ids=input_ids, labels=labels, use_cache=False).loss.detach().float().cpu())
    atomic_json(args.result, {"status": "PASS", "sample_id": args.sample_id, "loss": loss})
    return 0


def finalize(args: argparse.Namespace) -> int:
    result = json.loads(args.runtime.joinpath("TRAINING_RESULT.json").read_text(encoding="utf-8"))
    schedule = json.loads(args.freeze.joinpath("TRAINING_SCHEDULE_2M.json").read_text(encoding="utf-8"))
    points = result["points"]
    best = select_best_safe_checkpoint(points)
    selection = {"status": "PASS", "selection_rule": "lowest Novel macro-family loss subject to replay loss change <= 5%", "best_safe_checkpoint": best}
    atomic_json(args.runtime / "CHECKPOINT_SELECTION.json", selection)
    columns = ["training_tokens", "optimizer_steps", "novel_macro_family_loss", "novel_source_macro_loss", "novel_micro_loss", "novel_gain_relative", "replay_macro_family_loss", "replay_source_macro_loss", "replay_micro_loss", "replay_loss_change", "training_loss", "tokens_per_sec"]
    curve = []
    for item in points:
        train = item.get("training") or {}
        curve.append({**{key: item.get(key) for key in columns[:10]}, "training_loss": train.get("training_loss"), "tokens_per_sec": train.get("tokens_per_sec")})
    atomic_json(args.runtime / "LEARNING_CURVE.json", {"status": "PASS", "points": curve})
    with (args.runtime / "LEARNING_CURVE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(curve)
    atomic_json(args.runtime / "TRAINING_DISTRIBUTION_AUDIT.json", distribution_audit(schedule["episodes"]))
    last = points[-1]
    classes = classify_results(last["novel_gain_relative"], last["replay_loss_change"])
    gate = {
        "status": classes["gate"], "NOVEL_CAPABILITY_RESULT": classes["novel_capability_result"],
        "REPLAY_RETENTION_RESULT": classes["replay_retention_result"],
        "novel_gain_relative": last["novel_gain_relative"], "replay_loss_change": last["replay_loss_change"],
        "interpretation_thresholds_are_diagnostic_not_significance_claims": True,
        "next_phase_recommendation": next_phase_recommendation(classes["gate"], last["novel_gain_relative"], last["replay_loss_change"]),
        "long_training_beyond_2m_started": False,
    }
    atomic_json(args.runtime / "CAPABILITY_PILOT_GATE.json", gate)
    atomic_json(args.runtime / "CAPABILITY_PILOT_DATA_ACCESS_AUDIT.json", {
        "status": "PASS", "novel_train_accessed": True, "replay_train_accessed": True,
        "novel_validation_accessed": True, "replay_retention_accessed": True,
        "novel_holdout_accessed": False, "eval60_accessed": False, "kaggle_gold_accessed": False,
    })
    atomic_json(args.runtime / "BEST_CHECKPOINT_RELOAD_AUDIT.json", {"status": "PENDING_FRESH_PROCESS" if best else "NOT_APPLICABLE_NO_SAFE_CHECKPOINT"})
    report = f"""# Capability Pilot 2M V1\n\n- status: {gate['status']}\n- processed transformer tokens: {result['actual_transformer_tokens']}\n- optimizer steps: {result['optimizer_steps']}\n- best safe checkpoint: {best['checkpoint_path'] if best else 'NONE'}\n- Novel Holdout accessed: false\n- Eval60 accessed: false\n- Kaggle Gold accessed: false\n- long training beyond 2M started: false\n"""
    atomic_text(args.runtime / "REPORT.md", report)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("prepare", "train", "reload", "finalize"), required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--runtime", type=Path)
    p.add_argument("--checkpoints", type=Path)
    p.add_argument("--model-path", type=Path)
    p.add_argument("--novel-train-root", type=Path)
    p.add_argument("--novel-validation-root", type=Path)
    p.add_argument("--replay-shard", type=Path)
    p.add_argument("--novel-fingerprint", type=Path)
    p.add_argument("--scientific-gate", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--sample-id")
    p.add_argument("--result", type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.mode == "prepare": return prepare(args)
    if args.mode == "train": return train_worker(args)
    if args.mode == "reload": return reload_worker(args)
    return finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
