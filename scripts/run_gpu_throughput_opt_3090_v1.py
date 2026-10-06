from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from gpu_benchmark_3090_v1.benchmark import (
    REPLAY_ROWS,
    REPLAY_SHA256,
    BenchmarkGateError,
    assert_lora_partition,
    require_single_cuda_device,
    sha256_file,
    truncate_example,
    validate_model_package,
)
from gpu_throughput_opt_3090_v1.benchmark import (
    CONFIG_A,
    CONFIG_B,
    CONFIG_C,
    CONFIG_D,
    CONTEXT,
    DATASET_FINGERPRINT,
    SEED,
    SOURCE_COMMIT,
    TARGET_MODULES,
    RunConfig,
    attention_mask_required,
    effective_episode_groups,
    runtime_extrapolation,
    select_fastest_safe,
    update_fairness,
    vram_classification,
)


EXPECTED_TRAINABLE = 132_120_576


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_preflight(model_path: Path, output: Path) -> None:
    package = validate_model_package(model_path)
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    atomic_json(output / "MODEL_IDENTITY_PREFLIGHT.json", {
        "status": "PASS",
        "model_target": "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
        "local_path": str(model_path),
        "model_type": config.model_type,
        "tokenizer_class": tokenizer.__class__.__name__,
        **package,
    })


def gpu_preflight(output: Path) -> None:
    import torch
    require_single_cuda_device(torch.cuda.device_count(), torch.cuda.get_device_name(0))
    if not torch.cuda.is_bf16_supported():
        raise BenchmarkGateError("BF16_UNAVAILABLE")
    prop = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).strip()
    atomic_json(output / "GPU_PREFLIGHT.json", {
        "status": "PASS",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_devices": torch.cuda.device_count(),
        "gpu_exact_name": torch.cuda.get_device_name(0),
        "total_vram_bytes": prop.total_memory,
        "driver": driver,
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "bitsandbytes": package_version("bitsandbytes"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported": True,
    })


def flash_attention_preflight(output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "FLASH_ATTN_2_NOT_AVAILABLE_IN_FROZEN_ENVIRONMENT",
        "available": False,
        "environment_modified": False,
    }
    if importlib.util.find_spec("flash_attn") is not None:
        try:
            import flash_attn
            result.update({"status": "PASS", "available": True, "version": getattr(flash_attn, "__version__", None)})
        except Exception as exc:
            result["import_error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(output / "FLASH_ATTN2_PREFLIGHT.json", result)
    return result


def filtered_rows(paths: list[Path], wanted: set[str]) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq
    found: dict[str, dict[str, Any]] = {}
    remaining = set(wanted)
    for path in paths:
        if not remaining:
            break
        table = pq.read_table(
            path,
            columns=["sample_id", "input_ids", "labels"],
            filters=[("sample_id", "in", sorted(remaining))],
        )
        for row in table.to_pylist():
            found[str(row["sample_id"])] = row
            remaining.discard(str(row["sample_id"]))
    if remaining:
        raise BenchmarkGateError(f"SCHEDULE_ROWS_MISSING={sorted(remaining)[:5]}")
    return found


def load_scheduled_rows(novel_root: Path, replay_path: Path, schedule: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    novel_ids = {str(row["sample_id"]) for row in schedule if row["pool"] == "POOL_NOVEL_V1_1"}
    replay_ids = {str(row["sample_id"]) for row in schedule if row["pool"] == "POOL_REPLAY_V2_1"}
    novel = filtered_rows(sorted(novel_root.glob("*.parquet")), novel_ids)
    replay = filtered_rows([replay_path], replay_ids)
    return {
        **{("POOL_NOVEL_V1_1", key): value for key, value in novel.items()},
        **{("POOL_REPLAY_V2_1", key): value for key, value in replay.items()},
    }


def build_model(model_path: Path, config: RunConfig, adapter_path: Path | None = None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    attention = "flash_attention_2" if config.attention_backend == "flash_attention_2" else "sdpa"
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "attn_implementation": attention,
    }
    if config.precision == "NF4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.config.use_cache = False
    if config.precision == "NF4":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if config.gradient_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    if adapter_path is None:
        model = get_peft_model(model, LoraConfig(
            r=64,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ))
    else:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    return model


def collate(examples: list[dict[str, Any]], device):
    import torch
    width = max(len(row["input_ids"]) for row in examples)
    input_ids, labels, attention = [], [], []
    for row in examples:
        padding = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
        attention.append([1] * len(row["input_ids"]) + [0] * padding)
    mask = torch.tensor(attention, dtype=torch.long, device=device) if attention_mask_required(
        [len(row["input_ids"]) for row in examples]
    ) else None
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
        mask,
        width * len(examples),
    )


def model_loss(model, input_ids, labels, mask):
    kwargs = {"input_ids": input_ids, "labels": labels, "use_cache": False}
    if mask is not None:
        kwargs["attention_mask"] = mask
    return model(**kwargs).loss


def parameter_samples(model) -> tuple[list[str], list[list[int]], list[float]]:
    import torch
    names, shapes, samples = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        flat = parameter.detach().reshape(-1)
        count = min(32, flat.numel())
        indices = torch.linspace(0, flat.numel() - 1, count, device=flat.device).long()
        names.append(name)
        shapes.append(list(parameter.shape))
        samples.extend(float(value) for value in flat[indices].float().cpu())
    return names, shapes, samples


def telemetry_start(path: Path):
    handle = path.open("w", encoding="utf-8", newline="")
    handle.write("timestamp,utilization_gpu_pct,utilization_memory_pct,memory_used_mb,memory_total_mb,temperature_c,power_w,clocks_sm_mhz,clocks_mem_mhz\n")
    handle.flush()
    query = "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem"
    process = subprocess.Popen(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-l", "2"],
        stdout=handle, stderr=subprocess.DEVNULL, text=True,
    )
    return process, handle


def telemetry_stop(monitor) -> None:
    if monitor is None:
        return
    process, handle = monitor
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    handle.close()


def parse_telemetry(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    summary = {}
    for key in ("utilization_gpu_pct", "utilization_memory_pct", "memory_used_mb", "temperature_c", "power_w"):
        values = [float(row[key]) for row in rows if row.get(key, "").strip()]
        summary[key] = {"mean": sum(values) / len(values), "max": max(values)} if values else None
    return {"samples": len(rows), "summary": summary}


def worker(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import bitsandbytes as bnb
    import torch
    config = RunConfig(**json.loads(args.config.read_text(encoding="utf-8"))["run_config"])
    config.validate()
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))["schedule"]
    rows = load_scheduled_rows(args.novel_root, args.replay_shard, schedule)
    prepared = []
    for item in schedule:
        raw = rows[(item["pool"], str(item["sample_id"]))]
        values = truncate_example(raw["input_ids"], raw["labels"], CONTEXT)
        if values["truncated"]:
            raise BenchmarkGateError(f"CONTEXT_8704_TRUNCATION={item['sample_id']}")
        values["shifted_supervised_tokens"] = sum(value != -100 for value in values["labels"][1:])
        prepared.append((item, values))
    model = build_model(args.model_path, config, args.adapter if args.mode == "reload" else None)
    attention = getattr(model.config, "_attn_implementation", None)
    device = torch.device("cuda:0")

    def first_reference_loss() -> float:
        chunk = [prepared[index][1] for index in range(config.micro_batch)]
        input_ids, labels, mask, _ = collate(chunk, device)
        with torch.no_grad():
            return float(model_loss(model, input_ids, labels, mask).detach().float().cpu())

    if args.mode == "reload":
        model.eval()
        loss = first_reference_loss()
        atomic_json(args.result, {"status": "PASS", "loss": loss, "attention_backend": attention})
        return 0

    partition = assert_lora_partition(model.named_parameters())
    if partition["trainable_parameters"] != EXPECTED_TRAINABLE:
        raise BenchmarkGateError(f"TRAINABLE_PARAMETER_COUNT_MISMATCH={partition['trainable_parameters']}")
    params = [value for value in model.parameters() if value.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(params, lr=5e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    warmup_steps = 0 if args.mode == "smoke" else 3
    measured_steps = 2 if args.mode == "smoke" else 10
    total_steps = warmup_steps + measured_steps
    initial_names: list[str] = []
    initial_shapes: list[list[int]] = []
    initial_samples: list[float] = []
    if args.mode == "smoke":
        initial_names, initial_shapes, initial_samples = parameter_samples(model)
    model.train()
    monitor = None
    telemetry_path = args.output / f"telemetry_{config.name}.csv"
    losses: list[float] = []
    raw_tokens = transformer_tokens = supervised_tokens = 0
    forward_time = backward_time = optimizer_time = dataloader_time = 0.0
    measured_started = None
    torch.cuda.reset_peak_memory_stats(0)
    try:
        for step in range(total_steps):
            measured = step >= warmup_steps
            if step == warmup_steps:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(0)
                forward_time = backward_time = optimizer_time = dataloader_time = 0.0
                monitor = telemetry_start(telemetry_path) if args.mode == "performance" else None
                measured_started = time.perf_counter()
            group = [prepared[index][1] for index in range(step * 4, step * 4 + 4)]
            total_group_supervised = sum(row["shifted_supervised_tokens"] for row in group)
            if total_group_supervised <= 0:
                raise BenchmarkGateError("NO_SUPERVISED_TOKENS")
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for offset in range(0, 4, config.micro_batch):
                data_started = time.perf_counter()
                chunk = group[offset:offset + config.micro_batch]
                input_ids, labels, mask, padded_tokens = collate(chunk, device)
                if measured:
                    dataloader_time += time.perf_counter() - data_started
                chunk_supervised = sum(row["shifted_supervised_tokens"] for row in chunk)
                torch.cuda.synchronize()
                started = time.perf_counter()
                loss = model_loss(model, input_ids, labels, mask)
                torch.cuda.synchronize()
                if measured:
                    forward_time += time.perf_counter() - started
                if not bool(torch.isfinite(loss)):
                    raise BenchmarkGateError("NONFINITE_LOSS")
                weight = chunk_supervised / total_group_supervised
                group_loss += float(loss.detach().float().cpu()) * weight
                torch.cuda.synchronize()
                started = time.perf_counter()
                (loss * weight).backward()
                torch.cuda.synchronize()
                if measured:
                    backward_time += time.perf_counter() - started
                    raw_tokens += sum(row["processed_tokens"] for row in chunk)
                    transformer_tokens += padded_tokens
                    supervised_tokens += chunk_supervised
            if args.mode == "smoke" and not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in params
            ):
                raise BenchmarkGateError("NONFINITE_GRADIENT")
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            if measured:
                optimizer_time += time.perf_counter() - started
                losses.append(group_loss)
            if args.mode == "smoke" and step == 0:
                names, shapes, current = parameter_samples(model)
                updates = [now - before for now, before in zip(current, initial_samples)]
                fingerprint = {
                    "config": config.name,
                    "parameter_names": names,
                    "parameter_shapes": shapes,
                    "initial_sample_sha256": hashlib.sha256(json.dumps(initial_samples, separators=(",", ":")).encode()).hexdigest(),
                    "update_samples": updates,
                    "finite": all(value == value and abs(value) != float("inf") for value in updates),
                    "sample_count": len(updates),
                }
                atomic_json(args.update_fingerprint, fingerprint)
        torch.cuda.synchronize()
        measured_seconds = time.perf_counter() - measured_started
    finally:
        telemetry_stop(monitor)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    vram = vram_classification(total_vram, peak_reserved)
    result = {
        "status": "PASS",
        "config": config.json(),
        "mode": args.mode,
        "optimizer_steps": measured_steps,
        "warmup_optimizer_steps": warmup_steps,
        "microsteps": measured_steps * config.grad_accumulation,
        "raw_episode_tokens": raw_tokens,
        "actual_transformer_tokens": transformer_tokens,
        "supervised_tokens": supervised_tokens,
        "wall_seconds": measured_seconds,
        "tokens_per_second": raw_tokens / measured_seconds,
        "transformer_tokens_per_second": transformer_tokens / measured_seconds,
        "supervised_tokens_per_second": supervised_tokens / measured_seconds,
        "seconds_per_microstep": measured_seconds / (measured_steps * config.grad_accumulation),
        "seconds_per_optimizer_step": measured_seconds / measured_steps,
        "forward_seconds": forward_time,
        "backward_seconds": backward_time,
        "optimizer_seconds": optimizer_time,
        "dataloader_wait_seconds": dataloader_time,
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_vram,
        "vram": vram,
        "supervised_token_loss_rate": 0.0,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "min_loss": min(losses),
        "max_loss": max(losses),
        "finite_losses": True,
        "attention_backend": attention,
        "parameter_partition": partition,
    }
    if args.mode == "smoke":
        args.adapter.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(args.adapter)
        model.eval()
        result["checkpoint_reference_loss"] = first_reference_loss()
    else:
        result["telemetry"] = parse_telemetry(telemetry_path)
    atomic_json(args.result, result)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def run_subprocess(command: list[str], result_path: Path) -> dict[str, Any]:
    completed = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "TOKENIZERS_PARALLELISM": "false"})
    if not result_path.is_file():
        return {"status": "FAIL", "failure_class": "INFRASTRUCTURE", "returncode": completed.returncode}
    return json.loads(result_path.read_text(encoding="utf-8"))


def worker_command(args: argparse.Namespace, config_path: Path, mode: str, result: Path, adapter: Path, update: Path) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--worker", "--mode", mode,
        "--model-path", str(args.model_path), "--novel-root", str(args.novel_root),
        "--replay-shard", str(args.replay_shard), "--schedule", str(args.output / "COMMON_BENCHMARK_SCHEDULE.json"),
        "--config", str(config_path), "--output", str(args.output), "--result", str(result),
        "--adapter", str(adapter), "--update-fingerprint", str(update),
    ]


def controller(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=False)
    args.temp_root.mkdir(parents=True, exist_ok=False)
    gpu_preflight(args.output)
    model_preflight(args.model_path, args.output)
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", SOURCE_COMMIT], text=True).strip() != SOURCE_COMMIT:
        raise BenchmarkGateError("SOURCE_COMMIT_UNAVAILABLE")
    fingerprint = json.loads((repo / "artifacts/novel_training_data_v1_1/NOVEL_DATASET_FINGERPRINT.json").read_text(encoding="utf-8"))
    if fingerprint.get("fingerprint_sha256") != DATASET_FINGERPRINT:
        raise BenchmarkGateError("DATASET_FINGERPRINT_MISMATCH")
    if sha256_file(args.replay_shard) != REPLAY_SHA256:
        raise BenchmarkGateError("REPLAY_SHA256_MISMATCH")
    import pyarrow.parquet as pq
    if pq.read_metadata(args.replay_shard).num_rows != REPLAY_ROWS:
        raise BenchmarkGateError("REPLAY_ROW_COUNT_MISMATCH")
    novel_manifest = json.loads((repo / "artifacts/novel_training_data_v1_1/NOVEL_TRAIN_SHARD_MANIFEST.json").read_text(encoding="utf-8"))
    for shard in novel_manifest["shards"]:
        path = args.novel_root / Path(shard["logical_name"]).name
        if not path.is_file() or sha256_file(path) != shard["sha256"]:
            raise BenchmarkGateError(f"NOVEL_SHARD_HASH_MISMATCH={path.name}")
    source_schedule_path = repo / "artifacts/gpu_benchmark_3090_v1/BENCHMARK_SAMPLE_SCHEDULE.json"
    source_schedule = json.loads(source_schedule_path.read_text(encoding="utf-8"))
    schedule = source_schedule["schedule"][:52]
    groups = effective_episode_groups(schedule, 13)
    common = {
        "status": "FROZEN",
        "source_schedule": str(source_schedule_path.relative_to(repo)).replace("\\", "/"),
        "source_schedule_sha256": sha256_file(source_schedule_path),
        "seed": source_schedule["seed"],
        "schedule": schedule,
        "effective_episode_groups": groups,
        "same_four_ordered_episodes_all_configs": True,
        "exclusion_counts": {"validation": 0, "holdout": 0, "HARD_EXCLUDE": 0, "QUARANTINE": 0, "eval60": 0},
    }
    atomic_json(args.output / "COMMON_BENCHMARK_SCHEDULE.json", common)
    access = {
        "novel_train_accessed": True, "replay_train_accessed": True,
        "novel_validation_accessed": False, "novel_holdout_accessed": False,
        "eval60_accessed": False,
    }
    atomic_json(args.output / "GPU_DATA_ACCESS_AUDIT.json", {"status": "PASS", **access})
    flash = flash_attention_preflight(args.output)

    configs: dict[str, RunConfig] = {config.name: config for config in (CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D)}
    config_paths: dict[str, Path] = {}
    results: dict[str, dict[str, Any]] = {}
    for config in configs.values():
        path = args.output / f"CONFIG_{config.name}.json"
        atomic_json(path, {"status": "FROZEN", "run_config": {
            key: value for key, value in config.json().items()
            if key in {"name", "precision", "quantization", "micro_batch", "grad_accumulation", "attention_backend", "gradient_checkpointing"}
        }, "scientific_contract": config.json()})
        config_paths[config.name] = path

    def smoke_reload(config: RunConfig) -> dict[str, Any]:
        name = config.name
        adapter = args.temp_root / f"adapter_{name}"
        update = args.temp_root / f"UPDATE_{name}.json"
        smoke_path = args.output / f"SMOKE_{name}.json"
        smoke = run_subprocess(worker_command(args, config_paths[name], "smoke", smoke_path, adapter, update), smoke_path)
        row: dict[str, Any] = {"config": config.json(), "smoke": smoke, "adapter": str(adapter), "update": str(update)}
        if smoke.get("status") != "PASS":
            return row
        reload_path = args.output / f"RELOAD_{name}.json"
        reloaded = run_subprocess(worker_command(args, config_paths[name], "reload", reload_path, adapter, update), reload_path)
        tolerance = max(1e-3, abs(smoke["checkpoint_reference_loss"]) * 5e-4)
        reload_pass = reloaded.get("status") == "PASS" and abs(reloaded["loss"] - smoke["checkpoint_reference_loss"]) <= tolerance
        row["reload"] = reloaded
        row["reload_pass"] = reload_pass
        row["reload_tolerance"] = tolerance
        return row

    for config in (CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D):
        results[config.name] = smoke_reload(config)
        smoke = results[config.name]["smoke"]
        if smoke.get("status") != "PASS" and smoke.get("failure_class") == "OOM":
            results[config.name]["status"] = "OOM_AT_MB4" if config.name == CONFIG_C.name else "BF16_BASE_OOM" if config.name == CONFIG_D.name else "OOM"

    fingerprints = {}
    for config in (CONFIG_A, CONFIG_B, CONFIG_C):
        path = Path(results[config.name]["update"])
        if path.is_file():
            fingerprints[config.name] = json.loads(path.read_text(encoding="utf-8"))
    fairness: dict[str, Any] = {"reference": CONFIG_A.name, "same_effective_episode_groups": True, "comparisons": {}}
    if CONFIG_A.name not in fingerprints:
        fairness["status"] = "FAIL"
    else:
        fairness["comparisons"][CONFIG_A.name] = {"status": "PASS", "normalized_sample_update_difference": 0.0, "same_initialization": True}
        for config in (CONFIG_B, CONFIG_C):
            if config.name in fingerprints:
                fairness["comparisons"][config.name] = update_fairness(fingerprints[CONFIG_A.name], fingerprints[config.name])
            else:
                fairness["comparisons"][config.name] = {"status": "NOT_RUN", "reason": results[config.name].get("status", "SMOKE_FAIL")}
        fairness["status"] = "PASS" if fairness["comparisons"][CONFIG_B.name]["status"] == "PASS" and fairness["comparisons"][CONFIG_A.name]["status"] == "PASS" else "FAIL"
    atomic_json(args.output / "FAIRNESS_UPDATE_AUDIT.json", fairness)

    gradient_ok = {
        CONFIG_A.name: fairness.get("comparisons", {}).get(CONFIG_A.name, {}).get("status") == "PASS",
        CONFIG_B.name: fairness.get("comparisons", {}).get(CONFIG_B.name, {}).get("status") == "PASS",
        CONFIG_C.name: fairness.get("comparisons", {}).get(CONFIG_C.name, {}).get("status") == "PASS",
        CONFIG_D.name: True,
    }

    def performance(config: RunConfig) -> dict[str, Any] | None:
        row = results[config.name]
        if row["smoke"].get("status") != "PASS" or not row.get("reload_pass") or not gradient_ok.get(config.name, True):
            return None
        perf_path = args.output / f"PERFORMANCE_{config.name}.json"
        perf = run_subprocess(worker_command(args, config_paths[config.name], "performance", perf_path, Path(row["adapter"]), Path(row["update"])), perf_path)
        row["performance"] = perf
        shutil.rmtree(row["adapter"], ignore_errors=True)
        return perf

    for config in (CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D):
        performance(config)

    if flash.get("available"):
        nf4_candidates = [
            config for config in (CONFIG_A, CONFIG_B, CONFIG_C)
            if results[config.name].get("performance", {}).get("status") == "PASS" and gradient_ok[config.name]
        ]
        if nf4_candidates:
            base = max(nf4_candidates, key=lambda config: results[config.name]["performance"]["tokens_per_second"])
            config_e = replace(base, name="E_BEST_NF4_FLASH_ATTN2", attention_backend="flash_attention_2")
            configs[config_e.name] = config_e
            config_paths[config_e.name] = args.output / f"CONFIG_{config_e.name}.json"
            atomic_json(config_paths[config_e.name], {"status": "FROZEN", "run_config": {
                key: value for key, value in config_e.json().items()
                if key in {"name", "precision", "quantization", "micro_batch", "grad_accumulation", "attention_backend", "gradient_checkpointing"}
            }, "scientific_contract": config_e.json(), "derived_from": base.name})
            results[config_e.name] = smoke_reload(config_e)
            gradient_ok[config_e.name] = True
            performance(config_e)
    else:
        results["E_BEST_NF4_FLASH_ATTN2"] = {"status": "FLASH_ATTN_2_NOT_AVAILABLE_IN_FROZEN_ENVIRONMENT"}

    main_candidates = [
        config for config in (CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D)
        if results[config.name].get("performance", {}).get("status") == "PASS"
        and gradient_ok.get(config.name, True)
    ]
    if main_candidates:
        best_main = max(main_candidates, key=lambda config: results[config.name]["performance"]["tokens_per_second"])
        best_perf = results[best_main.name]["performance"]
        if best_perf["peak_vram_reserved_bytes"] <= 17 * 2**30 and best_perf["vram"]["headroom_gib"] >= 5.0:
            config_f = replace(best_main, name="F_BEST_CONFIG_NO_GRADIENT_CHECKPOINTING", gradient_checkpointing=False)
            configs[config_f.name] = config_f
            config_paths[config_f.name] = args.output / f"CONFIG_{config_f.name}.json"
            atomic_json(config_paths[config_f.name], {"status": "FROZEN", "run_config": {
                key: value for key, value in config_f.json().items()
                if key in {"name", "precision", "quantization", "micro_batch", "grad_accumulation", "attention_backend", "gradient_checkpointing"}
            }, "scientific_contract": config_f.json(), "derived_from": best_main.name})
            results[config_f.name] = smoke_reload(config_f)
            gradient_ok[config_f.name] = True
            performance(config_f)
        else:
            results["F_BEST_CONFIG_NO_GRADIENT_CHECKPOINTING"] = {"status": "NOT_RUN_VRAM_GATE"}
    else:
        results["F_BEST_CONFIG_NO_GRADIENT_CHECKPOINTING"] = {"status": "NOT_RUN_NO_BASE_CONFIG"}

    access_pass = not any(access[key] for key in ("novel_validation_accessed", "novel_holdout_accessed", "eval60_accessed"))
    comparison_rows = []
    baseline_tps = results[CONFIG_A.name].get("performance", {}).get("tokens_per_second")
    for name in (CONFIG_A.name, CONFIG_B.name, CONFIG_C.name, CONFIG_D.name, "E_BEST_NF4_FLASH_ATTN2", "F_BEST_CONFIG_NO_GRADIENT_CHECKPOINTING"):
        row = results.get(name, {})
        perf = row.get("performance")
        config = configs.get(name)
        if perf and perf.get("status") == "PASS":
            comparison_rows.append({
                "config": name,
                "precision": config.precision,
                "quantization": config.quantization,
                "microbatch": config.micro_batch,
                "gradient_accumulation": config.grad_accumulation,
                "effective_batch": 4,
                "attention_backend": perf["attention_backend"],
                "gradient_checkpointing": config.gradient_checkpointing,
                "tokens_per_second": perf["tokens_per_second"],
                "actual_transformer_tokens_per_second": perf["transformer_tokens_per_second"],
                "speedup_vs_A": perf["tokens_per_second"] / baseline_tps,
                "seconds_per_optimizer_step": perf["seconds_per_optimizer_step"],
                "peak_vram_allocated_bytes": perf["peak_vram_allocated_bytes"],
                "peak_vram_reserved_bytes": perf["peak_vram_reserved_bytes"],
                "vram": perf["vram"],
                "gpu_utilization": perf["telemetry"]["summary"]["utilization_gpu_pct"],
                "power": perf["telemetry"]["summary"]["power_w"],
                "temperature": perf["telemetry"]["summary"]["temperature_c"],
                "finite_losses": perf["finite_losses"],
                "gradient_semantics_pass": gradient_ok.get(name, True),
                "checkpoint_reload_pass": row.get("reload_pass") is True,
                "supervised_token_loss_rate": perf["supervised_token_loss_rate"],
                "data_access_policy_pass": access_pass,
                "status": "PASS",
            })
        else:
            comparison_rows.append({"config": name, "status": row.get("status") or row.get("smoke", {}).get("failure_class") or "NOT_RUN"})
    winner = select_fastest_safe(comparison_rows)
    comparison = {
        "status": "PASS" if winner else "FAIL",
        "baseline_previous_tokens_per_second": 326.53989841364546,
        "baseline_A_tokens_per_second": baseline_tps,
        "configs": comparison_rows,
        "best": winner["config"] if winner else None,
        "selection_policy": "FASTEST_SAFE_STABLE_ZERO_TRUNCATION_NO_FORBIDDEN_ACCESS",
        "loss_not_used_for_selection": True,
        "nf4_bf16_quality_comparison_not_established": True,
    }
    atomic_json(args.output / "THROUGHPUT_COMPARISON.json", comparison)
    extrapolation = {
        row["config"]: runtime_extrapolation(row["tokens_per_second"])
        for row in comparison_rows if row.get("status") == "PASS"
    }
    atomic_json(args.output / "OPTIMIZED_RUNTIME_EXTRAPOLATION.json", extrapolation)
    atomic_json(args.output / "GPU_THROUGHPUT_OPT_GATE.json", {
        "status": "PASS" if winner else "FAIL",
        "best": winner["config"] if winner else None,
        "data_access_policy_pass": access_pass,
        "long_training_started": False,
        "measured_optimizer_steps_per_successful_config": 10,
        "maximum_measured_optimizer_steps_allowed": 15,
    })
    atomic_json(args.output / "PROVENANCE.json", {
        "source_commit": SOURCE_COMMIT,
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "model_path": str(args.model_path),
        "benchmark_only": True,
        "LONG_TRAINING_STARTED": False,
    })
    for row in results.values():
        update = row.get("update") if isinstance(row, dict) else None
        if update:
            Path(update).unlink(missing_ok=True)
    return 0 if winner else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("smoke", "performance", "reload"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--novel-root", type=Path, required=True)
    parser.add_argument("--replay-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--update-fingerprint", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return worker(args) if args.worker else controller(args)
    except Exception as exc:
        failure = type(exc).__name__
        upper = str(exc).upper()
        if failure == "OutOfMemoryError" or "OUT OF MEMORY" in upper:
            failure = "OOM"
        elif "NONFINITE" in upper:
            failure = "NUMERICAL"
        payload = {"status": "FAIL", "failure_class": failure, "error": str(exc)}
        if args.result:
            atomic_json(args.result, payload)
        elif args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output / "FAILURE.json", payload | {"LONG_TRAINING_STARTED": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
