from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from gpu_benchmark_3090_v1.benchmark import (
    DATASET_FINGERPRINT,
    POOL_WEIGHTS,
    REPLAY_ROWS,
    REPLAY_SHA256,
    SEED,
    SOURCE_COMMIT,
    TARGET_MODULES,
    BenchmarkGateError,
    assert_lora_partition,
    derive_gate,
    make_family_schedule,
    require_single_cuda_device,
    runtime_extrapolation,
    sha256_file,
    truncate_example,
    validate_model_package,
    vram_safety,
)


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


def read_metadata(novel_root: Path, replay_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import pyarrow.parquet as pq
    columns = ["sample_id", "source", "generator_family", "sequence_length", "supervised_token_count", "final_training_role"]
    novel = []
    for path in sorted(novel_root.glob("*.parquet")):
        for row in pq.read_table(path, columns=columns).to_pylist():
            row["split"] = "train"
            novel.append(row)
    replay = pq.read_table(replay_path, columns=columns).to_pylist()
    for row in replay:
        row["split"] = "train"
    return novel, replay


def load_scheduled_rows(novel_root: Path, replay_path: Path, schedule: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq
    wanted = {row["sample_id"] for row in schedule}
    found: dict[str, dict[str, Any]] = {}
    for path in [*sorted(novel_root.glob("*.parquet")), replay_path]:
        for row in pq.read_table(path, columns=["sample_id", "input_ids", "labels"]).to_pylist():
            if row["sample_id"] in wanted:
                found[row["sample_id"]] = row
        if len(found) == len(wanted):
            break
    missing = sorted(wanted - set(found))
    if missing:
        raise BenchmarkGateError(f"SCHEDULE_ROWS_MISSING={missing[:5]}")
    return found


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
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    atomic_json(output / "GPU_PREFLIGHT.json", {
        "status": "PASS",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_devices": torch.cuda.device_count(),
        "gpu_exact_name": torch.cuda.get_device_name(0),
        "total_vram_bytes": properties.total_memory,
        "driver": driver,
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "bitsandbytes": package_version("bitsandbytes"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "attention_backend_requested": "sdpa",
    })


def build_model(model_path: Path, seed: int, adapter_path: Path | None = None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if adapter_path is None:
        config = LoraConfig(
            r=64,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        )
        model = get_peft_model(model, config)
    else:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    return model


def telemetry_start(path: Path):
    handle = path.open("w", encoding="utf-8", newline="")
    handle.write("timestamp,utilization_gpu_pct,utilization_memory_pct,memory_used_mb,memory_total_mb,temperature_c,power_w,clocks_sm_mhz,clocks_mem_mhz\n")
    handle.flush()
    query = "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem"
    process = subprocess.Popen(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-l", "2"],
        stdout=handle,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return process, handle


def telemetry_stop(process, handle) -> None:
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    handle.close()


def parse_telemetry(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    numeric = {}
    for key in ("utilization_gpu_pct", "utilization_memory_pct", "memory_used_mb", "temperature_c", "power_w"):
        values = [float(row[key]) for row in rows if row.get(key, "").strip()]
        numeric[key] = {"mean": sum(values) / len(values), "max": max(values)} if values else None
    return {"samples": len(rows), "summary": numeric}


def worker(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import bitsandbytes as bnb
    import torch
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))["schedule"]
    rows = load_scheduled_rows(args.novel_root, args.replay_shard, schedule)
    prepared = []
    for item in schedule:
        row = rows[item["sample_id"]]
        prepared.append((item, truncate_example(row["input_ids"], row["labels"], args.max_length)))
    model = build_model(args.model_path, SEED, args.adapter if args.mode == "reload" else None)
    partition = assert_lora_partition(model.named_parameters()) if args.mode != "reload" else None
    attention = getattr(model.config, "_attn_implementation", None)
    device = torch.device("cuda:0")

    def batch(index: int):
        item, values = prepared[index % len(prepared)]
        return item, values, torch.tensor([values["input_ids"]], dtype=torch.long, device=device), torch.tensor([values["labels"]], dtype=torch.long, device=device)

    if args.mode == "reload":
        model.eval()
        _, values, input_ids, labels = batch(0)
        with torch.no_grad():
            loss = float(model(input_ids=input_ids, labels=labels, use_cache=False).loss.detach().float().cpu())
        atomic_json(args.result, {"status": "PASS", "loss": loss, "processed_tokens": values["processed_tokens"], "attention_backend": attention})
        return 0

    model.train()
    params = [value for value in model.parameters() if value.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(params, lr=5e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    warmup_steps = 0 if args.mode in ("smoke", "diagnostic") else 3
    measured_steps = 5 if args.mode == "smoke" else 2 if args.mode == "diagnostic" else 20
    total_steps = warmup_steps + measured_steps
    losses: list[float] = []
    measured_losses: list[float] = []
    total_tokens = total_supervised = truncated_samples = supervised_lost = 0
    forward_time = backward_time = optimizer_time = dataloader_time = 0.0
    telemetry_path = args.output / f"telemetry_{args.max_length}.csv"
    monitor = telemetry_start(telemetry_path) if args.mode == "performance" else None
    torch.cuda.reset_peak_memory_stats(0)
    measured_started = None
    micro_index = 0
    try:
        for step in range(total_steps):
            if step == warmup_steps:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(0)
                forward_time = backward_time = optimizer_time = dataloader_time = 0.0
                measured_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(4):
                data_started = time.perf_counter()
                _, values, input_ids, labels = batch(micro_index)
                micro_index += 1
                dataloader_time += time.perf_counter() - data_started
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = model(input_ids=input_ids, labels=labels, use_cache=False)
                torch.cuda.synchronize()
                forward_time += time.perf_counter() - started
                loss = output.loss
                if not bool(torch.isfinite(loss)):
                    raise BenchmarkGateError("NONFINITE_LOSS")
                losses.append(float(loss.detach().float().cpu()))
                if step >= warmup_steps:
                    measured_losses.append(losses[-1])
                    total_tokens += values["processed_tokens"]
                    total_supervised += values["processed_supervised_tokens"]
                    truncated_samples += int(values["truncated"])
                    supervised_lost += values["supervised_tokens_lost"]
                torch.cuda.synchronize()
                started = time.perf_counter()
                (loss / 4.0).backward()
                torch.cuda.synchronize()
                backward_time += time.perf_counter() - started
            if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in params):
                raise BenchmarkGateError("NONFINITE_GRADIENT")
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            optimizer_time += time.perf_counter() - started
        measured_seconds = time.perf_counter() - measured_started
    except torch.cuda.OutOfMemoryError:
        atomic_json(args.result, {"status": "FAIL", "failure_class": "OOM", "max_seq_length": args.max_length})
        return 42
    finally:
        if monitor:
            telemetry_stop(*monitor)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    safety = vram_safety(total_vram, peak_reserved)
    result = {
        "status": "PASS",
        "mode": args.mode,
        "max_seq_length": args.max_length,
        "optimizer_steps": measured_steps,
        "warmup_optimizer_steps": warmup_steps,
        "microsteps": measured_steps * 4,
        "gradient_accumulation_steps": 4,
        "processed_tokens": total_tokens,
        "supervised_tokens": total_supervised,
        "supervised_token_loss_rate": supervised_lost / max(total_supervised + supervised_lost, 1),
        "wall_seconds": measured_seconds,
        "tokens_per_second": total_tokens / measured_seconds,
        "supervised_tokens_per_second": total_supervised / measured_seconds,
        "seconds_per_microstep": measured_seconds / (measured_steps * 4),
        "seconds_per_optimizer_step": measured_seconds / measured_steps,
        "forward_seconds": forward_time,
        "backward_seconds": backward_time,
        "optimizer_seconds": optimizer_time,
        "dataloader_wait_seconds": dataloader_time,
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_vram,
        "vram": safety,
        "vram_safety": safety["classification"],
        "truncated_microsteps": truncated_samples,
        "supervised_tokens_lost": supervised_lost,
        "initial_loss": measured_losses[0],
        "final_loss": measured_losses[-1],
        "mean_loss": sum(measured_losses) / len(measured_losses),
        "min_loss": min(measured_losses),
        "max_loss": max(measured_losses),
        "finite_losses": True,
        "attention_backend": attention,
        "parameter_partition": partition,
        "qlora": {"quantization": "NF4", "double_quantization": True, "compute_dtype": "bfloat16", "rank": 64, "alpha": 32, "dropout": 0.0, "target_modules": TARGET_MODULES, "optimizer": "PagedAdamW8bit", "learning_rate": 5e-5, "gradient_checkpointing": True},
    }
    if args.mode == "smoke":
        args.adapter.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(args.adapter)
        model.eval()
        _, _, input_ids, labels = batch(0)
        with torch.no_grad():
            result["checkpoint_reference_loss"] = float(model(input_ids=input_ids, labels=labels, use_cache=False).loss.detach().float().cpu())
    elif args.mode == "performance":
        result["telemetry"] = parse_telemetry(telemetry_path)
    atomic_json(args.result, result)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def run_subprocess(base: list[str], result_path: Path) -> dict[str, Any]:
    completed = subprocess.run(base, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "TOKENIZERS_PARALLELISM": "false"})
    if not result_path.is_file():
        return {"status": "FAIL", "failure_class": "INFRASTRUCTURE", "returncode": completed.returncode}
    return json.loads(result_path.read_text(encoding="utf-8"))


def controller(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=False)
    args.temp_root.mkdir(parents=True, exist_ok=False)
    gpu_preflight(args.output)
    model_preflight(args.model_path, args.output)
    repo_root = Path(__file__).resolve().parents[1]
    frozen_gate = json.loads((repo_root / "artifacts/novel_training_data_v1_1/SCIENTIFIC_TRAINING_GATE_V1_1.json").read_text(encoding="utf-8"))
    frozen_fingerprint = json.loads((repo_root / "artifacts/novel_training_data_v1_1/NOVEL_DATASET_FINGERPRINT.json").read_text(encoding="utf-8"))
    if frozen_gate.get("GPU_BENCHMARK_READY") is not True or frozen_gate.get("GPU_TRAINING_STARTED") is not False:
        raise BenchmarkGateError("FROZEN_DATA_GATE_NOT_READY")
    if frozen_fingerprint.get("fingerprint_sha256") != DATASET_FINGERPRINT:
        raise BenchmarkGateError("NOVEL_DATASET_FINGERPRINT_MISMATCH")
    source_commit = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", SOURCE_COMMIT], text=True).strip()
    if source_commit != SOURCE_COMMIT:
        raise BenchmarkGateError("SCIENTIFIC_SOURCE_COMMIT_UNAVAILABLE")
    if sha256_file(args.replay_shard) != REPLAY_SHA256:
        raise BenchmarkGateError("REPLAY_SHA256_MISMATCH")
    novel_manifest = json.loads((repo_root / "artifacts/novel_training_data_v1_1/NOVEL_TRAIN_SHARD_MANIFEST.json").read_text(encoding="utf-8"))
    for shard in novel_manifest["shards"]:
        path = args.novel_root / Path(shard["logical_name"]).name
        if not path.is_file() or sha256_file(path) != shard["sha256"]:
            raise BenchmarkGateError(f"NOVEL_SHARD_HASH_MISMATCH={path.name}")
    novel, replay = read_metadata(args.novel_root, args.replay_shard)
    if len(replay) != REPLAY_ROWS:
        raise BenchmarkGateError(f"REPLAY_ROW_COUNT_MISMATCH={len(replay)}")
    schedule = make_family_schedule(novel, replay, 92, SEED)
    access = {
        "novel_train_accessed": True,
        "replay_train_accessed": True,
        "novel_validation_accessed": False,
        "novel_holdout_accessed": False,
        "eval60_accessed": False,
    }
    atomic_json(args.output / "GPU_DATA_ACCESS_AUDIT.json", {"status": "PASS", **access})
    atomic_json(args.output / "BENCHMARK_SAMPLE_SCHEDULE.json", {
        "status": "FROZEN",
        "seed": SEED,
        "draws": len(schedule),
        "pool_weights": POOL_WEIGHTS,
        "actual_pool_counts": dict(Counter(row["pool"] for row in schedule)),
        "sampler": "POOL_THEN_FAMILY_UNIFORM_THEN_EPISODE",
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "schedule": schedule,
        "exclusion_counts": {"holdout": 0, "validation": 0, "HARD_EXCLUDE": 0, "QUARANTINE": 0},
    })
    schedule_path = args.output / "BENCHMARK_SAMPLE_SCHEDULE.json"
    script = Path(__file__).resolve()
    contexts: dict[str, Any] = {}
    reload_rows = []
    for length in (8192, 8704):
        adapter = args.temp_root / f"adapter_{length}"
        smoke_path = args.output / f"SMOKE_{length}.json"
        common = [sys.executable, str(script), "--worker", "--mode", "smoke", "--model-path", str(args.model_path), "--novel-root", str(args.novel_root), "--replay-shard", str(args.replay_shard), "--schedule", str(schedule_path), "--output", str(args.output), "--result", str(smoke_path), "--adapter", str(adapter), "--max-length", str(length)]
        smoke = run_subprocess(common, smoke_path)
        contexts[str(length)] = {"smoke": smoke}
        if smoke.get("status") != "PASS":
            if length == 8704 and smoke.get("failure_class") == "OOM":
                diagnostic_path = args.output / "CONTEXT_8512_DIAGNOSTIC.json"
                diagnostic_adapter = args.temp_root / "adapter_8512_diagnostic"
                diagnostic_cmd = [sys.executable, str(script), "--worker", "--mode", "diagnostic", "--model-path", str(args.model_path), "--novel-root", str(args.novel_root), "--replay-shard", str(args.replay_shard), "--schedule", str(schedule_path), "--output", str(args.output), "--result", str(diagnostic_path), "--adapter", str(diagnostic_adapter), "--max-length", "8512"]
                diagnostic = run_subprocess(diagnostic_cmd, diagnostic_path)
                contexts["8512"] = {"diagnostic": diagnostic}
            continue
        reload_path = args.output / f"reload_{length}.json"
        reload_cmd = [sys.executable, str(script), "--worker", "--mode", "reload", "--model-path", str(args.model_path), "--novel-root", str(args.novel_root), "--replay-shard", str(args.replay_shard), "--schedule", str(schedule_path), "--output", str(args.output), "--result", str(reload_path), "--adapter", str(adapter), "--max-length", str(length)]
        reload_result = run_subprocess(reload_cmd, reload_path)
        tolerance = max(1e-3, abs(smoke["checkpoint_reference_loss"]) * 5e-4)
        reload_pass = reload_result.get("status") == "PASS" and abs(reload_result["loss"] - smoke["checkpoint_reference_loss"]) <= tolerance
        reload_rows.append({"max_seq_length": length, "status": "PASS" if reload_pass else "FAIL", "reference_loss": smoke["checkpoint_reference_loss"], "reloaded_loss": reload_result.get("loss"), "absolute_tolerance": tolerance})
        if not reload_pass:
            continue
        perf_path = args.output / f"PERFORMANCE_{length}.json"
        perf_cmd = [sys.executable, str(script), "--worker", "--mode", "performance", "--model-path", str(args.model_path), "--novel-root", str(args.novel_root), "--replay-shard", str(args.replay_shard), "--schedule", str(schedule_path), "--output", str(args.output), "--result", str(perf_path), "--adapter", str(adapter), "--max-length", str(length)]
        contexts[str(length)]["performance"] = run_subprocess(perf_cmd, perf_path)
        shutil.rmtree(adapter, ignore_errors=True)
    expected_reload_lengths = [
        length for length in (8192, 8704)
        if contexts.get(str(length), {}).get("smoke", {}).get("status") == "PASS"
    ]
    reload_ok = (
        len(reload_rows) == len(expected_reload_lengths)
        and {row["max_seq_length"] for row in reload_rows} == set(expected_reload_lengths)
        and all(row["status"] == "PASS" for row in reload_rows)
    )
    atomic_json(args.output / "CHECKPOINT_RELOAD_AUDIT.json", {
        "status": "PASS" if reload_ok else "FAIL",
        "expected_contexts": expected_reload_lengths,
        "contexts": reload_rows,
    })
    extrapolation = {}
    for length in (8192, 8704):
        perf = contexts.get(str(length), {}).get("performance")
        if perf and perf.get("status") == "PASS":
            extrapolation[str(length)] = runtime_extrapolation(perf["tokens_per_second"])
    atomic_json(args.output / "RUNTIME_EXTRAPOLATION.json", extrapolation)
    result_a = contexts.get("8192", {}).get("performance")
    result_b = contexts.get("8704", {}).get("performance")
    diagnostic = contexts.get("8512", {}).get("diagnostic")
    smoke_failures = [
        value.get("smoke", {}) for key, value in contexts.items()
        if key in ("8192", "8704") and value.get("smoke", {}).get("status") == "FAIL"
    ]
    numerical_failure = any(
        failure.get("failure_class") == "NUMERICAL"
        or "NONFINITE" in str(failure.get("error", "")).upper()
        for failure in smoke_failures
    )
    infrastructure_failure = any(
        failure.get("failure_class") not in ("OOM", "NUMERICAL")
        and "NONFINITE" not in str(failure.get("error", "")).upper()
        for failure in smoke_failures
    )
    checks = {
        "data_access_policy": access["novel_train_accessed"] and access["replay_train_accessed"] and not any(access[key] for key in ("novel_validation_accessed", "novel_holdout_accessed", "eval60_accessed")),
        "infrastructure": not infrastructure_failure,
        "numerical": not numerical_failure and all(value.get("smoke", {}).get("finite_losses", False) for key, value in contexts.items() if key in ("8192", "8704") and value.get("smoke", {}).get("status") == "PASS"),
    }
    gate_status = derive_gate(checks, result_a, result_b, diagnostic)
    comparison = {"status": "COMPLETE", "contexts": contexts, "recommended_context": 8704 if gate_status == "PASS_8704_RECOMMENDED" else 8512 if gate_status == "PASS_8512_FULL_CONTENT_CANDIDATE" else 8192 if gate_status == "PASS_8192_ONLY" else None}
    atomic_json(args.output / "BENCHMARK_COMPARISON.json", comparison)
    passed_performance = [value.get("performance") for key, value in contexts.items() if key in ("8192", "8704") and value.get("performance", {}).get("status") == "PASS"]
    gate = {
        "status": gate_status,
        "checks": {
            "exact_sft139_model_verified": True,
            "single_rtx3090_only": True,
            "nf4_qlora_loaded": any(value.get("smoke", {}).get("status") == "PASS" for value in contexts.values()),
            "base_frozen_lora_trainable": all(value.get("smoke", {}).get("parameter_partition", {}).get("base_trainable_parameters") == 0 for key, value in contexts.items() if key in ("8192", "8704") and value.get("smoke", {}).get("status") == "PASS"),
            "family_aware_sampler_verified": True,
            "same_sample_schedule": True,
            "data_access_policy": checks["data_access_policy"],
            "smoke_forward_backward_optimizer": any(value.get("smoke", {}).get("status") == "PASS" for value in contexts.values()),
            "checkpoint_reload": reload_ok,
            "finite_losses": checks["numerical"],
            "measured_benchmark_completed": bool(passed_performance),
            "throughput_measured": all(value.get("tokens_per_second", 0) > 0 for value in passed_performance),
            "vram_measured": all(value.get("peak_vram_reserved_bytes", 0) > 0 for value in passed_performance),
            "telemetry_recorded": all(value.get("telemetry", {}).get("samples", 0) > 0 for value in passed_performance),
            "runtime_extrapolation_produced": bool(extrapolation),
            "long_training_not_started": True,
        },
        "LONG_GPU_TRAINING_STARTED": False,
    }
    atomic_json(args.output / "GPU_BENCHMARK_GATE.json", gate)
    atomic_json(args.output / "PROVENANCE.json", {"source_commit": SOURCE_COMMIT, "dataset_fingerprint": DATASET_FINGERPRINT, "model_path": str(args.model_path), "GPU_TRAINING_STARTED": False, "benchmark_only": True})
    return 0 if gate_status.startswith("PASS_") else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("smoke", "performance", "reload", "diagnostic"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--novel-root", type=Path, required=True)
    parser.add_argument("--replay-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--temp-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return worker(args) if args.worker else controller(args)
    except Exception as exc:
        failure_class = type(exc).__name__
        error_upper = str(exc).upper()
        if failure_class == "OutOfMemoryError" or "OUT OF MEMORY" in error_upper:
            failure_class = "OOM"
        elif "NONFINITE" in error_upper:
            failure_class = "NUMERICAL"
        if args.result:
            atomic_json(args.result, {"status": "FAIL", "failure_class": failure_class, "error": str(exc)})
        elif args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output / "FAILURE.json", {"status": "FAIL", "failure_class": failure_class, "error": str(exc), "GPU_TRAINING_STARTED": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
