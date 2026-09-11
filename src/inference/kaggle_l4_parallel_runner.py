"""Offline, task-parallel runner infrastructure for the Kaggle 4xL4 runtime.

The module intentionally keeps task-level model parallelism out of scope: every
worker owns one GPU and one local model endpoint.  It is safe to import on a
CPU development machine; hardware and artifact gates run only when requested.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import shutil
import math
from multiprocessing import get_context
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from queue import Empty
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence


class RunnerStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID_GPU_RUNTIME = "INVALID_GPU_RUNTIME"
    MODEL_ARTIFACT_MISSING = "MODEL_ARTIFACT_MISSING"
    CHECKPOINT_CONFIG_MISMATCH = "CHECKPOINT_CONFIG_MISMATCH"
    MODEL_ERROR = "MODEL_ERROR"
    PARSE_ERROR = "PARSE_ERROR"
    TIMEOUT = "TIMEOUT"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    CHECKPOINT_ERROR = "CHECKPOINT_ERROR"
    MODEL_LOAD_STALL = "MODEL_LOAD_STALL"


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    vram_mib: int
    driver_version: str


@dataclass(frozen=True)
class HardwareReport:
    status: RunnerStatus
    gpus: tuple[GPUInfo, ...]
    cuda_available: bool
    cuda_runtime: str | None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "gpus": [asdict(gpu) for gpu in self.gpus], "gpu_count": len(self.gpus), "cuda_available": self.cuda_available, "cuda_runtime": self.cuda_runtime, "reason": self.reason}


@dataclass(frozen=True)
class ModelArtifact:
    status: RunnerStatus
    model_path: str | None
    modelfile_path: str | None
    ollama_binary_path: str | None
    total_bytes: int
    manifest_sha256: str | None
    tokenizer: str | None
    quantization: str | None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"status": self.status.value}


@dataclass(frozen=True)
class WorkerAssignment:
    worker_id: int
    gpu_id: int
    task_ids: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointIdentity:
    experiment_id: str
    model_identifier: str
    model_config_sha256: str
    experiment_config_sha256: str
    prompt_version: str
    schema_version: str
    parameter_mode: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a complete JSON record before atomically making it visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def inspect_hardware(command: Callable[..., str] = subprocess.check_output) -> HardwareReport:
    """Inspect the allocated GPUs without importing a framework or loading a model."""
    try:
        output = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        return HardwareReport(RunnerStatus.INVALID_GPU_RUNTIME, (), False, None, f"nvidia-smi unavailable: {exc}")
    gpus: list[GPUInfo] = []
    for index, line in enumerate(item for item in output.splitlines() if item.strip()):
        values = [item.strip() for item in line.split(",")]
        if len(values) != 3:
            return HardwareReport(RunnerStatus.INVALID_GPU_RUNTIME, (), False, None, f"unparseable nvidia-smi row: {line!r}")
        try:
            gpus.append(GPUInfo(index, values[0], int(values[1]), values[2]))
        except ValueError:
            return HardwareReport(RunnerStatus.INVALID_GPU_RUNTIME, (), False, None, f"unparseable VRAM value: {line!r}")
    try:
        runtime = command(["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"], text=True, stderr=subprocess.STDOUT).splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        runtime = None
    is_l4x4 = len(gpus) == 4 and all(re.search(r"\bL4\b", gpu.name, flags=re.IGNORECASE) for gpu in gpus)
    if not is_l4x4:
        return HardwareReport(RunnerStatus.INVALID_GPU_RUNTIME, tuple(gpus), bool(gpus), runtime, "expected exactly four NVIDIA L4 GPUs")
    return HardwareReport(RunnerStatus.SUCCESS, tuple(gpus), True, runtime)


def _stable_manifest(paths: Iterable[Path], root: Path) -> tuple[str, int]:
    digest, total = hashlib.sha256(), 0
    for path in sorted(paths):
        stat = path.stat()
        digest.update(f"{path.relative_to(root).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        total += stat.st_size
    return digest.hexdigest(), total


def locate_offline_model(input_root: Path, *, required_quantization: str = "Q4_K_M") -> ModelArtifact:
    """Find the frozen Qwen GGUF and offline Ollama binary among attached inputs.

    A GGUF embeds its tokenizer, so its presence is recorded explicitly rather
    than pretending that a separate tokenizer.json is required for this backend.
    """
    if not input_root.exists():
        return ModelArtifact(RunnerStatus.MODEL_ARTIFACT_MISSING, None, None, None, 0, None, None, None, f"input root absent: {input_root}")
    ggufs = sorted(path for path in input_root.rglob("*.gguf") if required_quantization.lower() in path.name.lower() and "qwen3" in path.name.lower())
    modelfiles = sorted(path for path in input_root.rglob("Modelfile") if path.is_file())
    binaries = sorted(path for path in input_root.rglob("ollama") if path.is_file())
    if not ggufs:
        return ModelArtifact(RunnerStatus.MODEL_ARTIFACT_MISSING, None, None, str(binaries[0]) if binaries else None, 0, None, None, required_quantization, f"missing Qwen3 {required_quantization} GGUF in attached inputs")
    model = ggufs[0]
    modelfile = next((path for path in modelfiles if path.parent == model.parent), modelfiles[0] if modelfiles else None)
    if modelfile is None:
        return ModelArtifact(RunnerStatus.MODEL_ARTIFACT_MISSING, str(model), None, str(binaries[0]) if binaries else None, model.stat().st_size, None, "embedded_in_gguf", required_quantization, "missing Modelfile needed for deterministic offline Ollama import")
    if not binaries:
        return ModelArtifact(RunnerStatus.MODEL_ARTIFACT_MISSING, str(model), str(modelfile), None, model.stat().st_size, None, "embedded_in_gguf", required_quantization, "missing offline Ollama binary in attached inputs")
    manifest, total = _stable_manifest((model, modelfile, binaries[0]), input_root)
    return ModelArtifact(RunnerStatus.SUCCESS, str(model), str(modelfile), str(binaries[0]), total, manifest, "embedded_in_gguf", required_quantization)


def materialize_offline_ollama(artifact: ModelArtifact, destination: Path) -> ModelArtifact:
    """Copy the attached runtime to writable storage and restore execute bits.

    Read-only attached-input mounts can hide execute bits, so directly
    executing a binary from an attached artifact is not reliable.
    """
    if artifact.status != RunnerStatus.SUCCESS or not artifact.ollama_binary_path:
        raise RuntimeError(f"{RunnerStatus.MODEL_ARTIFACT_MISSING.value}: {artifact.reason}")
    source_binary = Path(artifact.ollama_binary_path)
    source_app = source_binary.parent.parent if source_binary.parent.name == "bin" else None
    if source_app is not None and source_app.name == "ollama_app":
        target_app = destination / "ollama_app"
        if not target_app.exists():
            shutil.copytree(source_app, target_app)
        target_binary = target_app / "bin" / "ollama"
    else:
        destination.mkdir(parents=True, exist_ok=True)
        target_binary = destination / "ollama"
        if not target_binary.exists():
            shutil.copy2(source_binary, target_binary)
    target_binary.chmod(target_binary.stat().st_mode | 0o111)
    return ModelArtifact(artifact.status, artifact.model_path, artifact.modelfile_path, str(target_binary), artifact.total_bytes, artifact.manifest_sha256, artifact.tokenizer, artifact.quantization, artifact.reason)


def stable_round_robin(task_ids: Sequence[str], worker_count: int = 4) -> tuple[WorkerAssignment, ...]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task IDs must be unique")
    shards = [tuple(task_ids[index::worker_count]) for index in range(worker_count)]
    return tuple(WorkerAssignment(worker_id=index, gpu_id=index, task_ids=shards[index]) for index in range(worker_count))


class TaskCheckpointStore:
    def __init__(self, root: Path, identity: CheckpointIdentity) -> None:
        self.root, self.identity = root, identity

    def path_for(self, task_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
            raise ValueError(f"unsafe task ID: {task_id!r}")
        return self.root / f"task_{task_id}.json"

    def load(self, task_id: str) -> dict[str, Any] | None:
        path = self.path_for(task_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def compatible(self, task_id: str) -> tuple[bool, dict[str, Any] | None, str | None]:
        record = self.load(task_id)
        if record is None:
            return False, None, None
        if record.get("identity") != self.identity.to_dict():
            return False, record, RunnerStatus.CHECKPOINT_CONFIG_MISMATCH.value
        if record.get("status") not in {RunnerStatus.SUCCESS.value, "IMPORTED", "TRAIN_CONSISTENT", "TRAIN_INCONSISTENT", "PROVIDER_FAILED", "MODEL_ERROR", "PARSE_ERROR", "EXECUTION_ERROR"}:
            return False, record, "INCOMPLETE_CHECKPOINT"
        return True, record, None

    def write(self, task_id: str, record: Mapping[str, Any]) -> Path:
        result = dict(record)
        result["task_id"] = task_id
        result["identity"] = self.identity.to_dict()
        result.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        path = self.path_for(task_id)
        atomic_write_json(path, result)
        return path


def merge_task_checkpoints(store: TaskCheckpointStore, task_ids: Sequence[str]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    mismatches: list[str] = []
    for task_id in task_ids:
        compatible, record, reason = store.compatible(task_id)
        if compatible and record is not None:
            records[task_id] = record
        elif reason == RunnerStatus.CHECKPOINT_CONFIG_MISMATCH.value:
            mismatches.append(task_id)
    if mismatches:
        raise RuntimeError(f"{RunnerStatus.CHECKPOINT_CONFIG_MISMATCH.value}: {sorted(mismatches)}")
    if len(records) != len(task_ids):
        missing = sorted(set(task_ids) - set(records))
        raise RuntimeError(f"incomplete checkpoints: {missing}")
    worker_counts = Counter(str(record.get("worker_id")) for record in records.values())
    return {"records": records, "task_count": len(records), "worker_task_counts": dict(worker_counts), "duplicate_task_count": len(records) - len(set(records)), "identity": store.identity.to_dict()}


def import_legacy_aggregate(legacy_path: Path, store: TaskCheckpointStore) -> list[str]:
    """Import old sequential V2 records only when frozen identity is identical."""
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    if legacy.get("frozen_config_sha256") != store.identity.experiment_config_sha256:
        raise RuntimeError(RunnerStatus.CHECKPOINT_CONFIG_MISMATCH.value)
    imported: list[str] = []
    for task_id, record in legacy.get("records", {}).items():
        if store.load(task_id) is None:
            store.write(task_id, {"status": record.get("status", "IMPORTED"), "worker_id": "imported", "gpu_id": None, "legacy_record": record, "imported_from": str(legacy_path)})
            imported.append(task_id)
    return imported


def summarize_worker_timings(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(record["total_task_seconds"]) for record in records if isinstance(record.get("total_task_seconds"), (float, int))]
    if not values:
        return {"tasks_completed": 0, "median_task_seconds": None, "p90_task_seconds": None}
    ordered = sorted(values)
    # Nearest-rank p90: especially important for the 2-task-per-worker
    # four-GPU scaling condition, where flooring selected the *lower* sample.
    p90 = ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.9) - 1))]
    return {"tasks_completed": len(values), "median_task_seconds": median(values), "p90_task_seconds": p90}


@dataclass(frozen=True)
class V2WorkerSpec:
    worker_id: int
    gpu_id: int
    endpoint: str
    ollama_binary: str
    ollama_models_root: str
    model_identifier: str
    project_root: str
    task_config_path: str
    frozen_config_path: str
    checkpoint_root: str
    parameter_mode: str
    identity: CheckpointIdentity


class OfflineOllamaServer:
    """One independently GPU-pinned Ollama server, owned by one worker."""

    def __init__(self, *, binary: Path, models_root: Path, endpoint: str, gpu_id: int, log_path: Path | None = None) -> None:
        self.binary, self.models_root, self.endpoint, self.gpu_id = binary, models_root, endpoint, gpu_id
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path = log_path
        self.log_handle: Any | None = None

    @property
    def environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": str(self.gpu_id), "OLLAMA_HOST": self.endpoint.removeprefix("http://"), "OLLAMA_MODELS": str(self.models_root), "OLLAMA_KEEP_ALIVE": "24h", "OLLAMA_NUM_PARALLEL": "1"})
        return environment

    def start(self, timeout_seconds: float = 90.0) -> float:
        self.models_root.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_handle = self.log_path.open("ab")
            output = self.log_handle
        else:
            output = subprocess.DEVNULL
        self.process = subprocess.Popen([str(self.binary), "serve"], env=self.environment, stdout=output, stderr=output)
        from urllib.request import urlopen
        while time.perf_counter() - started < timeout_seconds:
            if self.process.poll() is not None:
                raise RuntimeError(f"Ollama server exited with code {self.process.returncode}")
            try:
                with urlopen(f"{self.endpoint}/api/tags", timeout=2) as response:
                    if response.status == 200:
                        return time.perf_counter() - started
            except OSError:
                time.sleep(0.25)
        raise TimeoutError(f"Ollama server did not become ready at {self.endpoint}")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _port_from_endpoint(endpoint: str) -> int:
    match = re.search(r":(\d+)$", endpoint)
    if not match:
        raise ValueError(f"endpoint must include port: {endpoint}")
    return int(match.group(1))


def gpu_observation(gpu_id: int) -> dict[str, int | None]:
    """Best-effort local telemetry; its absence never changes task semantics."""
    try:
        output = subprocess.check_output(["nvidia-smi", "-i", str(gpu_id), "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT).strip()
        utilization, used, total = (int(value.strip()) for value in output.split(","))
        return {"gpu_utilization_percent": utilization, "vram_used_mib": used, "vram_total_mib": total}
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {"gpu_utilization_percent": None, "vram_used_mib": None, "vram_total_mib": None}


def merge_peak_observation(current: Mapping[str, int | None], observed: Mapping[str, int | None]) -> dict[str, int | None]:
    """Retain the highest observed utilization and allocated VRAM per worker."""
    result = dict(current)
    for key in ("gpu_utilization_percent", "vram_used_mib", "vram_total_mib"):
        values = [value for value in (current.get(key), observed.get(key)) if isinstance(value, int)]
        result[key] = max(values) if values else None
    return result


def _retryable_model_result(record: Mapping[str, Any]) -> bool:
    """Permit exactly one retry for a transient endpoint failure.

    Compiler, verifier and program-execution failures are scientific outcomes,
    not transport failures, and therefore must never be retried.
    """
    statuses = {str(candidate.get("status")) for candidate in record.get("candidate_results", [])}
    return bool(statuses) and statuses <= {RunnerStatus.TIMEOUT.value, "PROVIDER_FAILED"}


def _v2_worker_entry(spec: V2WorkerSpec, task_ids: tuple[str, ...], result_queue: Any) -> None:
    """Spawn target: its CUDA visibility is set before the server starts."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.gpu_id)
    server: OfflineOllamaServer | None = None
    try:
        from arc.io import discover_dataset_files, load_dataset
        from llm.macro_generator_v2 import MacroHypothesisGeneratorV2
        from llm.models import GenerationConfig
        from llm.v2_task_execution import execute_v2_task

        project = Path(spec.project_root)
        frozen = _read_json(Path(spec.frozen_config_path))
        tasks = load_dataset(discover_dataset_files(project / "data" / "raw")["training_challenges"])
        generation = frozen["generation"]
        config = GenerationConfig(frozen["model"]["model"], temperature=frozen["sampling"]["temperature"], top_p=frozen["sampling"]["top_p"], seed=frozen["sampling"]["seed"], hypothesis_budget=generation["candidate_budget"], max_output_tokens=generation["max_output_tokens"], context_window=frozen["model"]["context_window"], prompt_version=frozen["macro_dsl"]["prompt_version"])
        server = OfflineOllamaServer(binary=Path(spec.ollama_binary), models_root=Path(spec.ollama_models_root), endpoint=spec.endpoint, gpu_id=spec.gpu_id, log_path=Path(spec.checkpoint_root) / f"worker_{spec.worker_id}_ollama.log")
        load_seconds = server.start()
        before = gpu_observation(spec.gpu_id)
        peak = dict(before)
        generator = MacroHypothesisGeneratorV2(config, endpoint=f"{spec.endpoint}/api/generate")
        store = TaskCheckpointStore(Path(spec.checkpoint_root), spec.identity)
        records: list[dict[str, Any]] = []
        for task_id in task_ids:
            compatible, _, mismatch = store.compatible(task_id)
            if compatible:
                continue
            if mismatch == RunnerStatus.CHECKPOINT_CONFIG_MISMATCH.value:
                result_queue.put({"worker_id": spec.worker_id, "gpu_id": spec.gpu_id, "task_id": task_id, "status": mismatch})
                continue
            try:
                attempts = 0
                record = execute_v2_task(tasks[task_id], generator, parameter_mode=spec.parameter_mode, worker_id=spec.worker_id, gpu_id=spec.gpu_id)
                if _retryable_model_result(record):
                    attempts = 1
                    record = execute_v2_task(tasks[task_id], generator, parameter_mode=spec.parameter_mode, worker_id=spec.worker_id, gpu_id=spec.gpu_id)
                record["attempt_count"] = attempts + 1
                record["model_identifier"] = spec.model_identifier
                record["model_load_seconds"] = load_seconds
                store.write(task_id, record)
                records.append(record)
                peak = merge_peak_observation(peak, gpu_observation(spec.gpu_id))
            except TimeoutError as exc:
                store.write(task_id, {"status": RunnerStatus.TIMEOUT.value, "worker_id": spec.worker_id, "gpu_id": spec.gpu_id, "reason": str(exc), "total_task_seconds": None})
            except Exception as exc:
                store.write(task_id, {"status": RunnerStatus.EXECUTION_ERROR.value, "worker_id": spec.worker_id, "gpu_id": spec.gpu_id, "reason": f"{type(exc).__name__}: {exc}", "total_task_seconds": None})
        peak = merge_peak_observation(peak, gpu_observation(spec.gpu_id))
        server.stop()
        server = None
        after = gpu_observation(spec.gpu_id)
        token_seconds = sum(float(record.get("inference_seconds", 0)) for record in records)
        tokens = sum(int(record.get("output_tokens", 0)) for record in records)
        provider_failed_tasks = sum(_retryable_model_result(record) for record in records)
        result_queue.put({"worker_id": spec.worker_id, "gpu_id": spec.gpu_id, "status": RunnerStatus.SUCCESS.value, "model_load_seconds": load_seconds, "metrics": summarize_worker_timings(records) | {"completion_tokens_per_second": tokens / token_seconds if token_seconds else None, "gpu_before": before, "gpu_peak": peak, "gpu_after": after, "failures": provider_failed_tasks, "provider_failed_tasks": provider_failed_tasks, "retries": sum(max(0, int(record.get("attempt_count", 1)) - 1) for record in records)}, "task_ids": list(task_ids), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]})
    except Exception as exc:
        result_queue.put({"worker_id": spec.worker_id, "gpu_id": spec.gpu_id, "status": RunnerStatus.MODEL_ERROR.value, "reason": f"{type(exc).__name__}: {exc}", "task_ids": list(task_ids), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    finally:
        if server is not None:
            server.stop()


class KaggleL4ParallelRunner:
    """Four independent model workers with deterministic task sharding.

    The runner deliberately starts four *processes*, not four threads, because
    each process owns a distinct GPU-pinned Ollama server and CUDA context.
    """

    def __init__(self, *, identity: CheckpointIdentity, checkpoint_root: Path, worker_count: int = 4) -> None:
        if worker_count not in {1, 2, 4}:
            raise ValueError("scaling runner supports exactly 1, 2, or 4 workers")
        self.identity, self.checkpoint_root, self.worker_count = identity, checkpoint_root, worker_count

    def assignments(self, task_ids: Sequence[str]) -> tuple[WorkerAssignment, ...]:
        return stable_round_robin(task_ids, self.worker_count)

    def run_v2(self, *, task_ids: Sequence[str], project_root: Path, task_config_path: Path, frozen_config_path: Path, parameter_mode: str, artifact: ModelArtifact, base_port: int = 11434) -> dict[str, Any]:
        if artifact.status != RunnerStatus.SUCCESS or not artifact.model_path or not artifact.modelfile_path or not artifact.ollama_binary_path:
            raise RuntimeError(f"{RunnerStatus.MODEL_ARTIFACT_MISSING.value}: {artifact.reason}")
        # The model has already been materialized into a shared local Ollama
        # store by the notebook bootstrap.  Workers only serve and infer.
        context, queue = get_context("spawn"), None
        queue = context.Queue()
        workers = []
        for assignment in self.assignments(tuple(task_ids)):
            spec = V2WorkerSpec(assignment.worker_id, assignment.gpu_id, f"http://127.0.0.1:{base_port + assignment.worker_id}", artifact.ollama_binary_path, str(self.checkpoint_root.parent / "ollama_models"), self.identity.model_identifier, str(project_root), str(task_config_path), str(frozen_config_path), str(self.checkpoint_root), parameter_mode, self.identity)
            process = context.Process(target=_v2_worker_entry, args=(spec, assignment.task_ids, queue), daemon=False)
            process.start(); workers.append(process)
        for process in workers:
            process.join()
        reports: list[dict[str, Any]] = []
        for process in workers:
            try:
                reports.append(queue.get(timeout=10))
            except Empty:
                reports.append({"worker_id": None, "gpu_id": None, "status": RunnerStatus.MODEL_ERROR.value, "reason": f"worker exited without a report (exitcode={process.exitcode})", "task_ids": []})
        failures = [report for report in reports if report["status"] != RunnerStatus.SUCCESS.value]
        if failures:
            raise RuntimeError(f"worker failures: {failures}")
        merged = merge_task_checkpoints(TaskCheckpointStore(self.checkpoint_root, self.identity), tuple(task_ids))
        return merged | {"worker_reports": sorted(reports, key=lambda report: str(report["worker_id"])), "worker_to_gpu_mapping": {str(report["worker_id"]): report["gpu_id"] for report in reports}}
