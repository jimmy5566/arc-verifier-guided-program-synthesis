"""Run Qwen3-8B V2 task-level preflight or 1/2/4 scaling, solution-blind."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import CheckpointIdentity, atomic_write_json, inspect_hardware, sha256_file
from inference.qwen3_transformers_parallel_runner import Qwen3TransformersParallelRunner


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_identity(frozen_path: Path, frozen: dict[str, Any], parameter_mode: str, model_path: Path) -> CheckpointIdentity:
    macro = frozen["macro_dsl"]
    model_config = model_path / "config.json"
    if not model_config.is_file():
        raise FileNotFoundError(f"attached local model config absent: {model_config}")
    return CheckpointIdentity(
        "LLM_PROGRAM_SYNTHESIS_V2",
        frozen["model"]["model_source"],
        sha256_file(model_config),
        hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        macro["prompt_version"],
        macro["schema_version"],
        parameter_mode,
    )


def summarize_condition(outcome: dict[str, Any], task_count: int) -> dict[str, Any]:
    reports = outcome["worker_reports"]
    return {
        "wall_seconds": outcome["wall_seconds"],
        "tasks_per_hour": task_count / max(outcome["wall_seconds"], 0.001) * 3600,
        "worker_to_gpu_mapping": outcome["worker_to_gpu_mapping"],
        "duplicate_task_count": outcome["duplicate_task_count"],
        "worker_reports": reports,
        "all_tasks_checkpointed": outcome["task_count"] == task_count,
        "worker_metrics": {
            str(report["worker_id"]): report["metrics"]
            for report in reports
        },
    }


def run_condition(*, workers: int, task_ids: list[str], model_path: Path, challenge_path: Path, frozen_path: Path, frozen: dict[str, Any], checkpoints_root: Path, parameter_mode: str, include_merged: bool = False) -> dict[str, Any]:
    identity = build_identity(frozen_path, frozen, parameter_mode, model_path)
    runner = Qwen3TransformersParallelRunner(
        identity=identity,
        checkpoint_root=checkpoints_root / f"{workers}w_{parameter_mode}",
        worker_count=workers,
    )
    outcome = runner.run(
        task_ids=task_ids,
        model_path=model_path,
        challenge_path=challenge_path,
        frozen_config_path=frozen_path,
        parameter_mode=parameter_mode,
    )
    summary = summarize_condition(outcome, len(task_ids)) | {"checkpoint_identity": identity.to_dict()}
    return summary | {"_merged": outcome} if include_merged else summary


def frozen_pilot_checkpoint(*, config: dict[str, Any], frozen_path: Path, parameter_mode: str, outcome: dict[str, Any]) -> dict[str, Any]:
    """Freeze one complete, solution-blind prediction set for later scoring."""
    records = outcome["_merged"]["records"]
    declared = set(config["task_ids"])
    if set(records) != declared or len(records) != len(declared):
        raise RuntimeError("pilot prediction freeze rejected: task checkpoint set is incomplete or mismatched")
    return {
        "experiment_id": "LLM_PROGRAM_SYNTHESIS_V2",
        "stage": "PILOT_PREDICTIONS_FROZEN",
        "condition": config["config_id"],
        "parameter_mode": parameter_mode,
        "task_config": config,
        "frozen_config": str(frozen_path),
        "frozen_config_sha256": outcome["checkpoint_identity"]["experiment_config_sha256"],
        "protocol": "training challenges only during inference; no solution file was opened before this complete prediction freeze",
        "records": records,
        "task_count": len(records),
        "complete": True,
        "runtime_seconds": sum(float(record.get("inference_seconds", 0.0)) for record in records.values()),
        "wall_seconds": outcome["wall_seconds"],
        "total_prompt_tokens": sum(int(record.get("prompt_tokens", 0)) for record in records.values()),
        "total_output_tokens": sum(int(record.get("output_tokens", 0)) for record in records.values()),
        "worker_to_gpu_mapping": outcome["worker_to_gpu_mapping"],
        "worker_reports": outcome["worker_reports"],
        "duplicate_task_count": outcome["duplicate_task_count"],
        "prediction_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("load-only-gate", "four-worker-preflight", "scaling", "pilot"), required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    args = parser.parse_args()
    hardware = inspect_hardware()
    config, frozen = read_json(args.task_config), read_json(args.frozen_config)
    task_ids = list(config["task_ids"])
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("configured task IDs must be unique")
    if not args.challenge_path.is_file():
        raise FileNotFoundError(f"challenge input is absent: {args.challenge_path}")
    if hardware.status.value != "SUCCESS":
        result = {"status": "INVALID_GPU_RUNTIME", "hardware": hardware.to_dict(), "inference_started": False}
    elif args.mode == "load-only-gate":
        try:
            identity = build_identity(args.frozen_config, frozen, config.get("parameter_mode", "symbolic"), args.model_path)
            outcome = Qwen3TransformersParallelRunner(
                identity=identity,
                checkpoint_root=args.checkpoints_root / "load_only",
                worker_count=4,
            ).run(
                task_ids=[],
                model_path=args.model_path,
                challenge_path=args.challenge_path,
                frozen_config_path=args.frozen_config,
                parameter_mode=config.get("parameter_mode", "symbolic"),
                load_only=True,
            )
            expected_mapping = {str(index): index for index in range(4)}
            reports = outcome["worker_reports"]
            ready = len(reports) == 4 and all(report["model_load_seconds"] <= 300 for report in reports)
            mapping_ok = outcome["worker_to_gpu_mapping"] == expected_mapping
            result = {
                "experiment_id": "LLM_PROGRAM_SYNTHESIS_V2",
                "stage": "QWEN3_8B_FRESH_4_WORKER_LOAD_GATE",
                "protocol": "single-process sequential safetensors warm-up; workers load one after another; no ARC task or solution was executed",
                "hardware": hardware.to_dict(),
                "status": "SUCCESS" if ready and mapping_ok else "FAILED",
                "outcome": outcome,
                "worker_ready": ready,
                "mapping_ok": mapping_ok,
            }
            if not ready or not mapping_ok:
                result["failure_type"] = "LOAD_GATE_INVARIANT_FAILED"
        except Exception as exc:
            result = {"experiment_id": "LLM_PROGRAM_SYNTHESIS_V2", "stage": "QWEN3_8B_FRESH_4_WORKER_LOAD_GATE", "protocol": "single-process sequential safetensors warm-up; workers load one after another; no ARC task or solution was executed", "hardware": hardware.to_dict(), "status": "FAILED", "failure_type": type(exc).__name__, "failure": str(exc)}
    elif args.mode == "four-worker-preflight":
        selected = task_ids[:4]
        if len(selected) != 4:
            raise ValueError("four-worker preflight requires at least four configured tasks")
        try:
            outcome = run_condition(workers=4, task_ids=selected, model_path=args.model_path, challenge_path=args.challenge_path, frozen_path=args.frozen_config, frozen=frozen, checkpoints_root=args.checkpoints_root, parameter_mode=config.get("parameter_mode", "symbolic"))
            reports = outcome["worker_reports"]
            expected_mapping = {str(index): index for index in range(4)}
            mapping_ok = outcome["worker_to_gpu_mapping"] == expected_mapping
            per_worker_ok = all(
                report["metrics"]["tasks_completed"] == 1
                and report["metrics"]["completion_tokens"] > 0
                and report["metrics"]["gpu_peak"]["vram_used_mib"] > report["metrics"]["gpu_before"]["vram_used_mib"]
                and report["cuda_visible_devices"] == str(report["worker_id"])
                and report["logical_device"] == "cuda:0"
                for report in reports
            )
            if not mapping_ok or not per_worker_ok or outcome["duplicate_task_count"] != 0:
                raise RuntimeError(f"four-worker preflight invariant failed: mapping_ok={mapping_ok}, per_worker_ok={per_worker_ok}, duplicate_task_count={outcome['duplicate_task_count']}")
            result = {
                "experiment_id": "LLM_PROGRAM_SYNTHESIS_V2",
                "stage": "QWEN3_8B_FOUR_WORKER_ARC_PREFLIGHT",
                "protocol": "four distinct competition training challenges only; no solution file opened",
                "hardware": hardware.to_dict(),
                "task_ids": selected,
                "status": "SUCCESS",
                "outcome": outcome,
            }
        except Exception as exc:
            result = {"experiment_id": "LLM_PROGRAM_SYNTHESIS_V2", "stage": "QWEN3_8B_FOUR_WORKER_ARC_PREFLIGHT", "protocol": "four distinct competition training challenges only; no solution file opened", "hardware": hardware.to_dict(), "status": "FAILED", "failure_type": type(exc).__name__, "failure": str(exc)}
    elif args.mode == "scaling":
        try:
            conditions = {
                str(workers): run_condition(workers=workers, task_ids=task_ids, model_path=args.model_path, challenge_path=args.challenge_path, frozen_path=args.frozen_config, frozen=frozen, checkpoints_root=args.checkpoints_root, parameter_mode=config.get("parameter_mode", "symbolic"))
                for workers in (1, 2, 4)
            }
            one = conditions["1"]["wall_seconds"]
            conditions["2"]["speedup_vs_1"] = one / conditions["2"]["wall_seconds"]
            conditions["4"]["speedup_vs_1"] = one / conditions["4"]["wall_seconds"]
            conditions["4"]["parallel_efficiency_4"] = conditions["4"]["speedup_vs_1"] / 4
            result = {"experiment_id": "LLM_PROGRAM_SYNTHESIS_V2", "stage": "QWEN3_8B_1_2_4_GPU_SCALING_SMOKE", "protocol": "identical fixed competition training challenges only; no solution file opened", "hardware": hardware.to_dict(), "task_ids": task_ids, "status": "SUCCESS", "conditions": conditions}
        except Exception as exc:
            result = {"experiment_id": "LLM_PROGRAM_SYNTHESIS_V2", "stage": "QWEN3_8B_1_2_4_GPU_SCALING_SMOKE", "protocol": "identical fixed competition training challenges only; no solution file opened", "hardware": hardware.to_dict(), "status": "FAILED", "failure_type": type(exc).__name__, "failure": str(exc)}
    else:
        if len(task_ids) != 50:
            raise ValueError("the frozen Qwen3-8B pilot must declare exactly 50 task IDs")
        try:
            symbolic = run_condition(workers=4, task_ids=task_ids, model_path=args.model_path, challenge_path=args.challenge_path, frozen_path=args.frozen_config, frozen=frozen, checkpoints_root=args.checkpoints_root, parameter_mode="symbolic", include_merged=True)
            direct = run_condition(workers=4, task_ids=task_ids, model_path=args.model_path, challenge_path=args.challenge_path, frozen_path=args.frozen_config, frozen=frozen, checkpoints_root=args.checkpoints_root, parameter_mode="direct", include_merged=True)
            frozen_symbolic = frozen_pilot_checkpoint(config=config, frozen_path=args.frozen_config, parameter_mode="symbolic", outcome=symbolic)
            frozen_direct = frozen_pilot_checkpoint(config=config, frozen_path=args.frozen_config, parameter_mode="direct", outcome=direct)
            symbolic_path = args.output.parent / "LLM_PROGRAM_SYNTHESIS_V2_PILOT_50_QWEN3_8B_SYMBOLIC_FROZEN.json"
            direct_path = args.output.parent / "LLM_PROGRAM_SYNTHESIS_V2_PILOT_50_QWEN3_8B_DIRECT_FROZEN.json"
            atomic_write_json(symbolic_path, frozen_symbolic)
            atomic_write_json(direct_path, frozen_direct)
            result = {
                "experiment_id": "LLM_PROGRAM_SYNTHESIS_V2",
                "stage": "PILOT_PREDICTIONS_FROZEN",
                "status": "SUCCESS",
                "protocol": "both declared conditions completed on training challenges; neither runner has a solution input",
                "task_ids": task_ids,
                "condition_files": {"symbolic": symbolic_path.name, "direct": direct_path.name},
                "conditions": {
                    "symbolic": {key: value for key, value in symbolic.items() if key != "_merged"},
                    "direct": {key: value for key, value in direct.items() if key != "_merged"},
                },
            }
        except Exception as exc:
            result = {"experiment_id": "LLM_PROGRAM_SYNTHESIS_V2", "stage": "PILOT_PREDICTIONS_FROZEN", "status": "FAILED", "failure_type": type(exc).__name__, "failure": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, result)
    print(json.dumps({"status": result["status"], "stage": result.get("stage"), "failure_type": result.get("failure_type")}), flush=True)
    if result["status"] != "SUCCESS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
