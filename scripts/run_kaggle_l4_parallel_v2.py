"""Offline 1/2/4-worker scaling and four-L4 V2 pilot runner.

No training solution is opened here.  Scoring remains exclusively in the
existing gated V2 finalizer after all required condition checkpoints freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import (
    CheckpointIdentity,
    HardwareReport,
    KaggleL4ParallelRunner,
    OfflineOllamaServer,
    RunnerStatus,
    TaskCheckpointStore,
    atomic_write_json,
    inspect_hardware,
    locate_offline_model,
    materialize_offline_ollama,
)
from llm.macro_dsl import macro_program_json_schema


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_identity(frozen_path: Path, frozen: dict[str, Any], parameter_mode: str, model_manifest: str) -> CheckpointIdentity:
    macro = frozen["macro_dsl"]
    return CheckpointIdentity("LLM_PROGRAM_SYNTHESIS_V2", frozen["model"]["model"], model_manifest, hashlib.sha256(frozen_path.read_bytes()).hexdigest(), macro["prompt_version"], macro["schema_version"], parameter_mode)


def write_gate_report(path: Path, hardware: HardwareReport, artifact: dict[str, Any], *, status: str | None = None, detail: str | None = None) -> None:
    atomic_write_json(path, {"experiment_id": "KAGGLE_L4_PARALLEL_BENCHMARK", "status": status or (hardware.status.value if hardware.status != RunnerStatus.SUCCESS else artifact.get("status")), "hardware": hardware.to_dict(), "model_artifact": artifact, "inference_started": False, "detail": detail})


def prepare_offline_ollama(*, artifact, model_store: Path, bootstrap_dir: Path, model_identifier: str) -> OfflineOllamaServer:
    """Import GGUF into an isolated local Ollama store with no network request."""
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    source = Path(artifact.modelfile_path).read_text(encoding="utf-8")
    rewritten = []
    replaced = False
    for line in source.splitlines():
        if line.strip().upper().startswith("FROM "):
            rewritten.append(f"FROM {artifact.model_path}")
            replaced = True
        else:
            rewritten.append(line)
    if not replaced:
        raise RuntimeError("offline Modelfile has no FROM directive")
    modelfile = bootstrap_dir / "Qwen3-14B-Q4_K_M.Modelfile"
    modelfile.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    server = OfflineOllamaServer(binary=Path(artifact.ollama_binary_path), models_root=model_store, endpoint="http://127.0.0.1:11429", gpu_id=0, log_path=bootstrap_dir / "ollama_bootstrap.log")
    server.start()
    subprocess.run([artifact.ollama_binary_path, "create", model_identifier, "-f", str(modelfile)], env=server.environment, check=True, timeout=600)
    return server


def verify_model_preflight(*, endpoint: str, frozen: dict[str, Any]) -> dict[str, Any]:
    """Verify the exact offline model/schema interface before task inference."""
    payload = {"model": frozen["model"]["model"], "stream": False, "format": "json", "prompt": "Return JSON with an empty hypotheses array.", "options": {"temperature": frozen["sampling"]["temperature"], "top_p": frozen["sampling"]["top_p"], "num_predict": 32, "num_ctx": frozen["model"]["context_window"], "seed": frozen["sampling"]["seed"]}}
    request = Request(f"{endpoint}/api/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    started = perf_counter()
    try:
        with urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
    if not isinstance(body.get("response"), str):
        raise RuntimeError(f"malformed Ollama preflight response: {body}")
    return {"elapsed_seconds": perf_counter() - started, "prompt_tokens": body.get("prompt_eval_count"), "completion_tokens": body.get("eval_count")}


def aggregate_checkpoint(*, output: Path, config: dict[str, Any], frozen_path: Path, parameter_mode: str, merged: dict[str, Any]) -> None:
    records = merged["records"]
    statuses = [candidate["status"] for record in records.values() for candidate in record.get("candidate_results", [])]
    result = {"experiment_id": "LLM_PROGRAM_SYNTHESIS_V2", "condition": config["config_id"], "parameter_mode": parameter_mode, "task_config": config, "frozen_config": str(frozen_path), "frozen_config_sha256": merged["identity"]["experiment_config_sha256"], "protocol": "training challenges only; no solutions opened before every declared prediction is frozen", "records": records, "task_count": len(records), "complete": len(records) == len(config["task_ids"]), "runtime_seconds": sum(float(record.get("inference_seconds", 0)) for record in records.values()), "total_prompt_tokens": sum(int(record.get("prompt_tokens", 0)) for record in records.values()), "total_output_tokens": sum(int(record.get("output_tokens", 0)) for record in records.values()), "worker_to_gpu_mapping": merged["worker_to_gpu_mapping"], "worker_reports": merged["worker_reports"], "duplicate_task_count": merged["duplicate_task_count"]}
    atomic_write_json(output, result)


def run_condition(*, config: dict[str, Any], frozen_path: Path, frozen: dict[str, Any], parameter_mode: str, artifact, root: Path, checkpoints_root: Path, workers: int) -> dict[str, Any]:
    identity = build_identity(frozen_path, frozen, parameter_mode, artifact.manifest_sha256 or "")
    condition_root = checkpoints_root / f"{config['config_id']}_{parameter_mode}_{workers}w"
    runner = KaggleL4ParallelRunner(identity=identity, checkpoint_root=condition_root, worker_count=workers)
    started = perf_counter()
    merged = runner.run_v2(task_ids=config["task_ids"], project_root=root, task_config_path=Path(config["_path"]), frozen_config_path=frozen_path, parameter_mode=parameter_mode, artifact=artifact)
    wall_seconds = perf_counter() - started
    provider_failed_task_ids = sorted(
        task_id
        for task_id, record in merged["records"].items()
        if any(candidate.get("status") == "PROVIDER_FAILED" for candidate in record.get("candidate_results", []))
    )
    return {"worker_count": workers, "wall_seconds": wall_seconds, "tasks_per_minute": len(config["task_ids"]) / max(wall_seconds, 0.001) * 60, "merged": merged, "checkpoint_root": str(condition_root), "provider_failed_task_ids": provider_failed_task_ids}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("benchmark", "pilot"), required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--parameter-mode", choices=("symbolic", "direct"), default="symbolic")
    parser.add_argument("--input-root", type=Path, required=True, help="Read-only directory containing the separately obtained model/runtime artifact.")
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    root = Path.cwd()
    config, frozen = read_json(args.task_config), read_json(args.frozen_config)
    config["_path"] = str(args.task_config)
    hardware = inspect_hardware()
    gate_result = args.output_root / "results" / "KAGGLE_L4_PARALLEL_BENCHMARK.json"
    if hardware.status != RunnerStatus.SUCCESS:
        write_gate_report(gate_result, hardware, {"status": RunnerStatus.INVALID_GPU_RUNTIME.value})
        print(json.dumps({"status": RunnerStatus.INVALID_GPU_RUNTIME.value, "hardware": hardware.to_dict()}))
        return
    artifact = locate_offline_model(args.input_root, required_quantization=frozen["model"]["quantization"])
    if artifact.status != RunnerStatus.SUCCESS:
        write_gate_report(gate_result, hardware, artifact.to_dict())
        print(json.dumps({"status": RunnerStatus.MODEL_ARTIFACT_MISSING.value, "artifact": artifact.to_dict()}))
        return
    artifact = materialize_offline_ollama(artifact, args.output_root / "checkpoints" / "llm_v2_kaggle" / "offline_runtime")
    model_store = args.output_root / "checkpoints" / "llm_v2_kaggle" / "ollama_models"
    bootstrap_server = prepare_offline_ollama(artifact=artifact, model_store=model_store, bootstrap_dir=args.output_root / "checkpoints" / "llm_v2_kaggle" / "bootstrap", model_identifier=frozen["model"]["model"])
    try:
        preflight = verify_model_preflight(endpoint=bootstrap_server.endpoint, frozen=frozen)
    except Exception as exc:
        write_gate_report(gate_result, hardware, artifact.to_dict(), status="MODEL_BACKEND_INVALID", detail=f"{type(exc).__name__}: {exc}")
        print(json.dumps({"status": "MODEL_BACKEND_INVALID", "reason": f"{type(exc).__name__}: {exc}"}))
        return
    finally:
        bootstrap_server.stop()
    checkpoints_root = args.output_root / "checkpoints" / "llm_v2_kaggle"
    if args.mode == "benchmark":
        scaling: dict[str, Any] = {}
        for workers in (1, 2, 4):
            outcome = run_condition(config=config, frozen_path=args.frozen_config, frozen=frozen, parameter_mode=args.parameter_mode, artifact=artifact, root=root, checkpoints_root=checkpoints_root, workers=workers)
            scaling[str(workers)] = {key: value for key, value in outcome.items() if key != "merged"} | {"worker_reports": outcome["merged"]["worker_reports"], "worker_to_gpu_mapping": outcome["merged"]["worker_to_gpu_mapping"], "duplicate_task_count": outcome["merged"]["duplicate_task_count"]}
            if outcome["provider_failed_task_ids"]:
                write_gate_report(gate_result, hardware, artifact.to_dict(), status="MODEL_BACKEND_INVALID", detail=f"provider failures in {workers}-worker condition: {outcome['provider_failed_task_ids']}")
                print(json.dumps({"status": "MODEL_BACKEND_INVALID", "worker_count": workers, "provider_failed_task_ids": outcome["provider_failed_task_ids"]}))
                return
        one = scaling["1"]["wall_seconds"]
        for workers in (2, 4):
            scaling[str(workers)]["speedup_vs_1"] = one / scaling[str(workers)]["wall_seconds"]
        scaling["4"]["parallel_efficiency"] = scaling["4"]["speedup_vs_1"] / 4
        atomic_write_json(gate_result, {"experiment_id": "KAGGLE_L4_PARALLEL_BENCHMARK", "status": "COMPLETE", "hardware": hardware.to_dict(), "model_artifact": artifact.to_dict(), "model_preflight": preflight, "parameter_mode": args.parameter_mode, "scaling": scaling})
    else:
        outcome = run_condition(config=config, frozen_path=args.frozen_config, frozen=frozen, parameter_mode=args.parameter_mode, artifact=artifact, root=root, checkpoints_root=checkpoints_root, workers=4)
        if outcome["provider_failed_task_ids"]:
            write_gate_report(gate_result, hardware, artifact.to_dict(), status="MODEL_BACKEND_INVALID", detail=f"provider failures in pilot: {outcome['provider_failed_task_ids']}")
            print(json.dumps({"status": "MODEL_BACKEND_INVALID", "provider_failed_task_ids": outcome["provider_failed_task_ids"]}))
            return
        aggregate_checkpoint(output=args.output_root / "checkpoints" / f"LLM_PROGRAM_SYNTHESIS_V2_PILOT_50_{args.parameter_mode.upper()}.json", config=config, frozen_path=args.frozen_config, parameter_mode=args.parameter_mode, merged=outcome["merged"])
        print(json.dumps({"status": "PILOT_PREDICTIONS_FROZEN", "wall_seconds": outcome["wall_seconds"], "worker_to_gpu_mapping": outcome["merged"]["worker_to_gpu_mapping"]}))


if __name__ == "__main__":
    main()
