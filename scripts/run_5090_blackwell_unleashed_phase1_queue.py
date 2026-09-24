"""Bounded, no-idle Phase-1 generation queue for two Blackwell 5090 workers.

The parent process never imports CUDA.  It runs the portable bootstrap first,
then launches exactly one full-model worker process per physical GPU.  Each
worker executes only its predeclared sequence and persists each completed run
before immediately dispatching its next one.  Candidate changes are recorded,
not used as an automatic stop condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_eval3_reference_ttt import _task_hash
from scripts.run_eval3_runtime_opt import BLACKWELL_UNLEASHED_BACKEND_ID, FROZEN_STATUS


QUEUE_SCHEMA = "ARC2_BLACKWELL_UNLEASHED_QUEUE_V1"
REQUIRED_RUN_FILES = (
    "manifest.json",
    "config_resolved.json",
    "environment.json",
    "candidates_frozen.json",
    "telemetry.json",
    "events.jsonl",
    "hashes.json",
    "evaluation/report.json",
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    physical_gpu_id: int
    queue_position: int
    mode: str
    generation_micro_batch_size: int
    role: str


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_event(handle: Any, event: dict[str, Any]) -> None:
    line = json.dumps({"timestamp_unix": time.time(), **event}, sort_keys=True)
    handle.write(line + "\n")
    handle.flush()
    print(line, flush=True)


def _load_queue(path: Path) -> tuple[dict[str, Any], tuple[RunSpec, ...]]:
    queue = _read(path)
    if queue.get("schema_version") != QUEUE_SCHEMA:
        raise ValueError("unsupported Blackwell Phase-1 queue schema")
    hardware = queue.get("required_hardware", {})
    if hardware.get("compute_capability") != [12, 0]:
        raise ValueError("queue does not require sm_120 GPUs")
    if queue.get("backend", {}).get("id") != BLACKWELL_UNLEASHED_BACKEND_ID:
        raise ValueError("queue backend is not blackwell_unleashed_v1")
    if any(bool(queue["backend"].get(key)) for key in ("ttt_batching", "likelihood_batching", "torch_compile", "cuda_graphs")):
        raise ValueError("Phase-1 queue contains an out-of-scope execution change")
    try:
        runs = tuple(RunSpec(**item) for item in queue["runs"])
    except (KeyError, TypeError) as error:
        raise ValueError("invalid queue run specification") from error
    policy = queue.get("execution_policy", "persistent_two_gpu")
    expected_by_policy = {
        "persistent_two_gpu": {
            ("5090-unleashed-phase1-run-01-serial-batch1", 0, 1, 1),
            ("5090-unleashed-phase1-run-02-generation-batch2", 1, 1, 2),
            ("5090-unleashed-phase1-run-03-generation-batch4", 0, 2, 4),
            ("5090-unleashed-phase1-run-04-generation-batch4-repeat", 1, 2, 4),
        },
        "single_gpu_ephemeral_offpod_backup": {
            ("5090-unleashed-phase1-single-run-01-serial-batch1", 0, 1, 1),
            ("5090-unleashed-phase1-single-run-02-generation-batch2", 0, 2, 2),
            ("5090-unleashed-phase1-single-run-03-generation-batch4", 0, 3, 4),
            ("5090-unleashed-phase1-single-run-04-generation-batch4-repeat", 0, 4, 4),
        },
    }
    if policy not in expected_by_policy:
        raise ValueError(f"unsupported Phase-1 execution policy: {policy}")
    expected = expected_by_policy[policy]
    actual = {(item.run_id, item.physical_gpu_id, item.queue_position, item.generation_micro_batch_size) for item in runs}
    expected_gpu_count = 2 if policy == "persistent_two_gpu" else 1
    if hardware.get("gpu_count") != expected_gpu_count or actual != expected or any(item.mode != "serial" for item in runs):
        raise ValueError("queue schedule differs from the frozen 1/2/4/4 Phase-1 plan")
    return queue, runs


def _require_inputs(args: argparse.Namespace, queue: dict[str, Any]) -> dict[str, Any]:
    for path in (args.manifest, args.reference_config, args.challenge_path, args.solutions_path, args.model_path, args.native_config_dir):
        if not path.exists():
            raise FileNotFoundError(f"required Phase-1 input is missing: {path}")
    manifest = _read(args.manifest)
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "EVAL3_REFERENCE_TTT_COHORT_FROZEN" or len(task_ids) != 3 or manifest.get("task_ids_hash") != _task_hash(task_ids):
        raise ValueError("Phase-1 requires the immutable three-task Eval3 manifest")
    if hashlib.sha256(args.challenge_path.read_bytes()).hexdigest() != manifest.get("source_challenge_sha256"):
        raise ValueError("challenge file does not match the frozen Eval3 source hash")
    config = _read(args.reference_config)
    # The historical Kaggle export used CRLF, while portable Git archives may
    # normalize line endings. Verify parsed canonical JSON instead of treating
    # that non-scientific byte difference as configuration drift. The
    # companion still records the source export's raw-file digest.
    provenance_path = args.reference_config.with_name("reference_ttt_config_provenance.json")
    if not provenance_path.is_file():
        raise FileNotFoundError(f"reference configuration provenance is missing: {provenance_path}")
    provenance = _read(provenance_path)
    expected_canonical_hash = provenance.get("canonical_json_sha256")
    if not isinstance(expected_canonical_hash, str) or _json_digest(config) != expected_canonical_hash:
        raise ValueError("reference TTT configuration does not match the historical canonical config")
    frozen_required = {"rank": 256, "alpha": 32, "ttt_steps": 24, "generation_augmentation_count": 8}
    if {key: config.get(key) for key in frozen_required} != frozen_required:
        raise ValueError("reference TTT configuration violates frozen Eval3 settings")
    if queue["cohort"]["expected_task_count"] != len(task_ids):
        raise ValueError("queue and manifest task counts differ")
    if not args.ptxas_path.is_file() or not os.access(args.ptxas_path, os.X_OK):
        raise FileNotFoundError(f"verified PTXAS executable is missing: {args.ptxas_path}")
    return {"manifest": manifest, "reference_config": config, "reference_config_provenance": provenance}


def _bootstrap(args: argparse.Namespace, queue: dict[str, Any]) -> dict[str, Any]:
    bootstrap_log = args.runtime_root / "active_run" / queue["queue_id"] / "bootstrap.jsonl"
    bootstrap_log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "ARC2_GLOBAL_MOUNT": str(args.global_mount),
            "ARC2_PERSISTENT_ROOT": str(args.persistent_root),
            "ARC2_RUNTIME_ROOT": str(args.runtime_root),
            "ARC2_REPO_DIR": str(args.repo_dir),
            "ARC2_EXPECTED_GPU_COUNT": "2",
        }
    )
    if args.repo_url:
        environment["ARC2_REPO_URL"] = args.repo_url
    completed = subprocess.run(["bash", str(args.bootstrap_script)], env=environment, text=True, capture_output=True)
    _atomic_text(bootstrap_log, completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"BOOTSTRAP_FAILED exit={completed.returncode}; see {bootstrap_log}")
    ready: dict[str, Any] | None = None
    for line in completed.stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("event") == "READY_FOR_ARC2_EXPERIMENTS":
            ready = item
    if ready is None:
        raise RuntimeError("BOOTSTRAP_FAILED: ready manifest was not emitted")
    inventory = ready.get("gpu_inventory")
    if not isinstance(inventory, list) or len(inventory) != 2:
        raise RuntimeError("BOOTSTRAP_FAILED: did not validate two GPUs")
    for expected_id, item in enumerate(inventory):
        if item.get("physical_gpu_id") != expected_id or "RTX 5090" not in item.get("name", "") or item.get("capability") != [12, 0]:
            raise RuntimeError("BOOTSTRAP_FAILED: GPU inventory is not two RTX 5090 sm_120 devices")
    return ready


def _ephemeral_single_gpu_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the already-built one-GPU Blackwell environment without CUDA model loading."""
    if not args.python.is_file():
        raise FileNotFoundError(f"Blackwell environment Python is missing: {args.python}")
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(f"model config is missing: {args.model_path / 'config.json'}")
    environment = {
        **os.environ,
        "TRITON_PTXAS_PATH": str(args.ptxas_path),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    source = """
import hashlib, importlib.metadata, json, sys
from pathlib import Path
import torch, transformers, unsloth, peft, xformers
import xformers.ops as xops
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f'GPU_INVENTORY_MISMATCH: expected=1 actual={torch.cuda.device_count()}')
name = torch.cuda.get_device_name(0); capability = torch.cuda.get_device_capability(0)
if 'RTX 5090' not in name or capability != (12, 0):
    raise SystemExit(f'BLACKWELL_GPU_MISMATCH: name={name} capability={capability}')
q = torch.randn((1, 16, 4, 64), device='cuda', dtype=torch.bfloat16)
value = xops.memory_efficient_attention(q, q, q)
torch.cuda.synchronize()
if not bool(torch.isfinite(value).all()):
    raise SystemExit('XFORMERS_BF16_SMOKE_NONFINITE')
record = importlib.metadata.distribution('xformers').read_text('RECORD') or ''
model_config = Path(sys.argv[1]) / 'config.json'
print(json.dumps({
  'event':'READY_FOR_ARC2_EPHEMERAL_PHASE1',
  'gpu_inventory':[{'physical_gpu_id':0,'name':name,'capability':list(capability)}],
  'versions':{'python':'.'.join(map(str,sys.version_info[:3])),'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,'unsloth':unsloth.__version__,'peft':peft.__version__,'xformers':xformers.__version__,'torchao':importlib.metadata.version('torchao'),'triton':importlib.metadata.version('triton')},
  'model_config_sha256':hashlib.sha256(model_config.read_bytes()).hexdigest(),
  'xformers_distribution_record_sha256':hashlib.sha256(record.encode()).hexdigest(),
  'bf16_xformers_attention_smoke':True,
},sort_keys=True))
"""
    completed = subprocess.run([str(args.python), "-c", source, str(args.model_path)], env=environment, text=True, capture_output=True)
    preflight_log = args.runtime_root / "active_run" / "5090-unleashed-phase1-single-gpu-v1" / "ephemeral_preflight.log"
    preflight_log.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(preflight_log, completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"EPHEMERAL_BLACKWELL_PREFLIGHT_FAILED exit={completed.returncode}; see {preflight_log}")
    try:
        ready = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("EPHEMERAL_BLACKWELL_PREFLIGHT_FAILED: missing JSON manifest") from error
    if ready.get("event") != "READY_FOR_ARC2_EPHEMERAL_PHASE1" or ready.get("gpu_inventory", [{}])[0].get("capability") != [12, 0]:
        raise RuntimeError("EPHEMERAL_BLACKWELL_PREFLIGHT_FAILED: malformed inventory")
    return ready


def _environment_preflight(args: argparse.Namespace, queue: dict[str, Any]) -> dict[str, Any]:
    if queue.get("execution_policy") == "single_gpu_ephemeral_offpod_backup":
        return _ephemeral_single_gpu_preflight(args)
    return _bootstrap(args, queue)


def _resolved_config(*, frozen_config: dict[str, Any], ptxas_path: Path, spec: RunSpec) -> tuple[dict[str, Any], str]:
    # PTXAS is environment execution identity rather than a scientific TTT
    # parameter.  Preserve every scientific key byte-for-byte.
    scientific_config = {key: value for key, value in frozen_config.items() if key != "ptxas_path"}
    resolved = dict(frozen_config)
    # Persist a platform-neutral manifest value; the live launcher itself only
    # accepts a Linux executable on the remote Pod.
    resolved["ptxas_path"] = ptxas_path.as_posix()
    resolved["execution_backend"] = BLACKWELL_UNLEASHED_BACKEND_ID
    resolved["generation_micro_batch_size"] = spec.generation_micro_batch_size
    resolved["execution_mode"] = spec.mode
    return resolved, _json_digest(scientific_config)


def _subprocess_for_run(args: argparse.Namespace, spec: RunSpec, run_dir: Path, resolved_config_path: Path) -> list[str]:
    return [
        str(args.python), str(args.repo_dir / "scripts" / "run_eval3_runtime_opt.py"),
        "--manifest", str(args.manifest),
        "--reference-config", str(resolved_config_path),
        "--challenge-path", str(args.challenge_path),
        "--model-path", str(args.model_path),
        "--native-config-dir", str(args.native_config_dir),
        "--output", str(run_dir / "candidates_frozen.json"),
        "--checkpoint-dir", str(run_dir / "checkpoints"),
        "--mode", spec.mode,
        "--generation-micro-batch-size", str(spec.generation_micro_batch_size),
        "--backend-id", BLACKWELL_UNLEASHED_BACKEND_ID,
        "--resume",
    ]


def _gpu_identity(physical_gpu_id: int) -> dict[str, str]:
    """Read a physical-GPU identity without importing CUDA in the parent."""
    completed = subprocess.run(
        ["nvidia-smi", f"--id={physical_gpu_id}", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(values) != 4:
        raise RuntimeError(f"cannot parse nvidia-smi identity for physical GPU {physical_gpu_id}")
    return {"physical_gpu_id": values[0], "uuid": values[1], "name": values[2], "driver_version": values[3]}


def _run_model_process(args: argparse.Namespace, spec: RunSpec, run_dir: Path, event_handle: Any, resolved_config_path: Path) -> None:
    child_environment = dict(os.environ)
    child_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(spec.physical_gpu_id),
            "ARC2_PHYSICAL_GPU_ID": str(spec.physical_gpu_id),
            "ARC2_RUN_ID": spec.run_id,
            "TRITON_PTXAS_PATH": str(args.ptxas_path),
            "HF_HOME": str(args.runtime_root / "hf_cache"),
            "HUGGINGFACE_HUB_CACHE": str(args.runtime_root / "hf_cache" / "hub"),
            "TORCH_HOME": str(args.runtime_root / "torch_cache"),
            "TRITON_CACHE_DIR": str(args.runtime_root / "triton_cache"),
            "UV_CACHE_DIR": str(args.runtime_root / "uv_cache"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    command = _subprocess_for_run(args, spec, run_dir, resolved_config_path)
    _write_event(event_handle, {"event": "PHASE1_RUN_PROCESS_START", "run_id": spec.run_id, "physical_gpu_id": spec.physical_gpu_id, "command": command})
    process = subprocess.Popen(command, env=child_environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip("\n")
        try:
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError
            _write_event(event_handle, {"event": "MODEL_EVENT", "run_id": spec.run_id, "physical_gpu_id": spec.physical_gpu_id, "payload": payload})
        except (json.JSONDecodeError, ValueError):
            _write_event(event_handle, {"event": "MODEL_STDOUT", "run_id": spec.run_id, "physical_gpu_id": spec.physical_gpu_id, "line": stripped})
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"MODEL_PROCESS_FAILED run={spec.run_id} exit={return_code}")


def _candidate_hashes(records: dict[str, Any]) -> dict[str, Any]:
    per_view: dict[str, str] = {}
    pools: dict[str, str] = {}
    for task_id, record in records.items():
        raw = record.get("raw_views", [])
        per_view[task_id] = _json_digest(raw)
        pools[task_id] = _json_digest(record.get("candidates", []))
    return {"per_view_candidate_sha256": per_view, "candidate_pool_sha256": pools}


def _telemetry(records: dict[str, Any]) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for task_id, record in records.items():
        ttt = record.get("telemetry", {}).get("ttt", {})
        generation = record.get("telemetry", {}).get("generation", {})
        whole = record.get("telemetry", {}).get("whole_task", {})
        tasks.append(
            {
                "task_id": task_id,
                "total_wall_seconds": whole.get("seconds"),
                "ttt_seconds": ttt.get("seconds"),
                "generation_seconds": generation.get("seconds"),
                "generated_token_count": generation.get("generated_token_count"),
                "generation_tokens_per_second": generation.get("generation_tokens_per_second"),
                "gpu_utilization_avg_pct": generation.get("gpu_utilization_avg_pct"),
                "gpu_utilization_max_pct": generation.get("gpu_utilization_max_pct"),
                "peak_allocated_bytes": generation.get("peak_allocated_bytes"),
                "peak_reserved_bytes": generation.get("peak_reserved_bytes"),
                "max_ttt_sequence_tokens": record.get("max_ttt_sequence_tokens"),
                "max_generation_prompt_tokens": generation.get("max_generation_prompt_tokens"),
                "max_generated_tokens": generation.get("max_generated_tokens"),
                "batch_size": record.get("generation_micro_batch_size"),
                "padding_tokens": generation.get("padding_tokens"),
                "padding_ratio": generation.get("padding_ratio"),
                "attention_backend": record.get("attention_backend"),
                "KV_CACHE_ACTIVE": record.get("KV_CACHE", {}).get("KV_CACHE_ACTIVE"),
                "invalid_candidate_count": record.get("invalid_candidate_count"),
                "fail_soft_events": record.get("fail_soft_events", []),
            }
        )
    generation_total = sum(float(item["generation_seconds"] or 0.0) for item in tasks)
    tokens_total = sum(int(item["generated_token_count"] or 0) for item in tasks)
    return {
        "task_count": len(tasks),
        "tasks": tasks,
        "aggregate": {
            "total_wall_seconds": sum(float(item["total_wall_seconds"] or 0.0) for item in tasks),
            "ttt_seconds": sum(float(item["ttt_seconds"] or 0.0) for item in tasks),
            "generation_seconds": generation_total,
            "generated_token_count": tokens_total,
            "generation_tokens_per_second": tokens_total / max(generation_total, 1e-9),
            "peak_allocated_bytes": max((int(item["peak_allocated_bytes"] or 0) for item in tasks), default=0),
            "peak_reserved_bytes": max((int(item["peak_reserved_bytes"] or 0) for item in tasks), default=0),
            "gpu_utilization_avg_pct": sum(float(item["gpu_utilization_avg_pct"] or 0.0) for item in tasks) / max(len(tasks), 1),
            "gpu_utilization_max_pct": max((float(item["gpu_utilization_max_pct"] or 0.0) for item in tasks), default=0.0),
            "padding_tokens": sum(int(item["padding_tokens"] or 0) for item in tasks),
        },
    }


def _write_hashes(run_dir: Path) -> None:
    files = [path for path in run_dir.rglob("*") if path.is_file() and path.name not in {"hashes.json", "SHA256SUMS"}]
    payload = {str(path.relative_to(run_dir)): _sha256_file(path) for path in sorted(files)}
    atomic_write_json(run_dir / "hashes.json", {"algorithm": "sha256", "files": payload})


def _validate_frozen_run(run_dir: Path) -> None:
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"run is incomplete before persistent sync: {missing}")
    candidates = _read(run_dir / "candidates_frozen.json")
    manifest = _read(run_dir / "manifest.json")
    if candidates.get("status") != FROZEN_STATUS or set(candidates.get("records", ())) != set(manifest.get("task_ids", ())):
        raise RuntimeError("run candidate artifact failed final schema validation")


def _freeze_and_score_run(
    args: argparse.Namespace,
    queue: dict[str, Any],
    inputs: dict[str, Any],
    bootstrap_ready: dict[str, Any],
    spec: RunSpec,
    run_dir: Path,
    event_handle: Any,
) -> None:
    resolved_config, scientific_hash = _resolved_config(frozen_config=inputs["reference_config"], ptxas_path=args.ptxas_path, spec=spec)
    resolved_config_path = run_dir / "config_resolved.json"
    atomic_write_json(resolved_config_path, resolved_config)
    _run_model_process(args, spec, run_dir, event_handle, resolved_config_path)
    candidates = _read(run_dir / "candidates_frozen.json")
    if candidates.get("status") != FROZEN_STATUS:
        raise RuntimeError(f"{spec.run_id}: runtime did not produce a frozen candidate artifact")
    records = candidates.get("records", {})
    if set(records) != set(inputs["manifest"]["task_ids"]):
        raise RuntimeError(f"{spec.run_id}: incomplete candidate records")
    if any(record.get("status") not in {"SUCCESS", "NO_VALID_NATIVE_CANDIDATE"} for record in records.values()):
        raise RuntimeError(f"{spec.run_id}: malformed task status")
    for task_id, record in records.items():
        if not record.get("ttt", {}).get("loss_finite") or not record.get("ttt", {}).get("adapter_updated") or not record.get("ttt", {}).get("base_model_unchanged"):
            raise RuntimeError(f"{spec.run_id}/{task_id}: TTT integrity failure")

    environment = {
        "bootstrap_ready": bootstrap_ready,
        "gpu": _gpu_identity(spec.physical_gpu_id),
        "physical_gpu_id": spec.physical_gpu_id,
        "cuda_visible_devices": str(spec.physical_gpu_id),
        "ptxas_path": str(args.ptxas_path),
        "environment_lock_sha256": _sha256_file(args.environment_lock) if args.environment_lock else None,
        "environment_manifest_sha256": _sha256_file(args.environment_manifest) if args.environment_manifest else None,
        "environment_versions": _read(args.environment_manifest).get("versions") if args.environment_manifest else bootstrap_ready.get("versions"),
        "model_manifest_sha256": _sha256_file(args.model_manifest) if args.model_manifest else None,
        "queue_config_sha256": _sha256_file(args.queue_config),
    }
    manifest = {
        "schema_version": "ARC2_BLACKWELL_UNLEASHED_RUN_V1",
        "run_id": spec.run_id,
        "queue_id": queue["queue_id"],
        "backend_id": BLACKWELL_UNLEASHED_BACKEND_ID,
        "role": spec.role,
        "physical_gpu_id": spec.physical_gpu_id,
        "task_ids": inputs["manifest"]["task_ids"],
        "task_ids_hash": inputs["manifest"]["task_ids_hash"],
        "source_challenge_sha256": inputs["manifest"]["source_challenge_sha256"],
        "scientific_config_sha256": scientific_hash,
        "backend_config_sha256": _json_digest({"mode": spec.mode, "generation_micro_batch_size": spec.generation_micro_batch_size, "backend": BLACKWELL_UNLEASHED_BACKEND_ID}),
        "candidate_status": candidates["status"],
        "solutions_opened_during_generation": False,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "environment.json", environment)
    atomic_write_json(run_dir / "telemetry.json", _telemetry(records))
    _write_event(event_handle, {"event": "PHASE1_CANDIDATES_FROZEN", "run_id": spec.run_id, "candidate_hashes": _candidate_hashes(records), "solutions_opened": False})

    report_path = run_dir / "evaluation" / "report.json"
    if args.defer_local_scoring:
        atomic_write_json(report_path, {
            "status": "CANDIDATES_FROZEN_PENDING_LOCAL_POST_FREEZE_SCORING",
            "run_id": spec.run_id,
            "candidate_artifact_sha256": _sha256_file(run_dir / "candidates_frozen.json"),
            "solutions_opened_on_gpu_pod": False,
        })
        _write_event(event_handle, {"event": "PHASE1_LOCAL_SCORE_DEFERRED", "run_id": spec.run_id, "solutions_opened": False})
    else:
        score_command = [
            str(args.python), str(args.repo_dir / "scripts" / "score_eval3_blackwell_unleashed_phase1.py"),
            "--manifest", str(args.manifest),
            "--candidates", str(run_dir / "candidates_frozen.json"),
            "--challenge-path", str(args.challenge_path),
            "--solutions-path", str(args.solutions_path),
            "--output", str(report_path),
        ]
        score_environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        completed = subprocess.run(score_command, env=score_environment, text=True, capture_output=True)
        _write_event(event_handle, {"event": "PHASE1_POST_FREEZE_SCORE", "run_id": spec.run_id, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
        if completed.returncode:
            raise RuntimeError(f"{spec.run_id}: post-freeze Any-of-K scoring failed")
    _write_event(event_handle, {"event": "PHASE1_RUN_FREEZE_COMPLETE", "run_id": spec.run_id})
    _write_hashes(run_dir)
    _validate_frozen_run(run_dir)


def _sync_run(args: argparse.Namespace, run_dir: Path) -> None:
    completed = subprocess.run(["bash", str(args.sync_script), str(run_dir), str(args.persistent_root)], text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"PERSISTENT_SYNC_FAILED run={run_dir.name}: {completed.stdout}{completed.stderr}")


def _candidate_grid_set(candidates: dict[str, Any]) -> set[str]:
    return {
        _json_digest(candidate.get("prediction"))
        for record in candidates.get("records", {}).values()
        for candidate in record.get("candidates", [])
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _final_queue_summary(args: argparse.Namespace, queue: dict[str, Any], runs: Iterable[RunSpec], run_root: Path, bootstrap_ready: dict[str, Any]) -> Path:
    by_id = {spec.run_id: _read(run_root / spec.run_id / "candidates_frozen.json") for spec in runs}
    reference = _candidate_grid_set(by_id["5090-unleashed-phase1-run-01-serial-batch1"])
    metrics: dict[str, Any] = {}
    structurally_valid: list[dict[str, Any]] = []
    for spec in runs:
        run_dir = run_root / spec.run_id
        telemetry = _read(run_dir / "telemetry.json")["aggregate"]
        report = _read(run_dir / "evaluation" / "report.json")
        candidate_set = _candidate_grid_set(by_id[spec.run_id])
        details = {
            "generation_seconds": telemetry["generation_seconds"],
            "total_wall_seconds": telemetry["total_wall_seconds"],
            "generation_tokens_per_second": telemetry["generation_tokens_per_second"],
            "gpu_utilization_avg_pct": telemetry["gpu_utilization_avg_pct"],
            "gpu_utilization_max_pct": telemetry["gpu_utilization_max_pct"],
            "peak_allocated_vram_mb": telemetry["peak_allocated_bytes"] / (1024 * 1024),
            "peak_reserved_vram_mb": telemetry["peak_reserved_bytes"] / (1024 * 1024),
            "ANY_OF_K": report["ANY_OF_K"],
            "candidate_jaccard_vs_serial": _jaccard(reference, candidate_set),
            "invalid_candidate_count": sum(row.get("invalid_candidate_count", 0) for row in by_id[spec.run_id]["records"].values()),
            "generation_micro_batch_size": spec.generation_micro_batch_size,
        }
        metrics[spec.run_id] = details
        # Completion validation already rejected mapping, integrity, malformed
        # records and failed task states.  Candidate divergence is intentionally
        # not part of the promotion predicate.
        structurally_valid.append({"run_id": spec.run_id, **details})
    promoted = min(structurally_valid, key=lambda item: (item["generation_seconds"], item["total_wall_seconds"], item["run_id"]))
    serial = metrics["5090-unleashed-phase1-run-01-serial-batch1"]
    summary = {
        "schema_version": "ARC2_BLACKWELL_UNLEASHED_QUEUE_SUMMARY_V1",
        "queue_id": queue["queue_id"],
        "bootstrap_ready": bootstrap_ready,
        "runs": metrics,
        "promotion": {
            "backend": "blackwell_unleashed_generation_v1",
            "promoted_run_id": promoted["run_id"],
            "generation_micro_batch_size": promoted["generation_micro_batch_size"],
            "generation_speedup_vs_serial": serial["generation_seconds"] / max(promoted["generation_seconds"], 1e-9),
            "total_speedup_vs_serial": serial["total_wall_seconds"] / max(promoted["total_wall_seconds"], 1e-9),
            "candidate_parity_required": False,
        },
        "integrity": {"all_runs_structurally_valid": True, "scientific_config_changed": False, "eval60_started": False, "ttt12_started": False},
    }
    summary_dir = run_root / f"{queue['queue_id']}-summary"
    summary_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(summary_dir / "phase1_queue_summary.json", summary)
    _write_hashes(summary_dir)
    return summary_dir


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-config", type=Path, required=True)
    parser.add_argument("--bootstrap-script", type=Path, required=True)
    parser.add_argument("--sync-script", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--solutions-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--environment-manifest", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--ptxas-path", type=Path, required=True)
    parser.add_argument("--persistent-root", type=Path, default=Path("/workspace-global/arc2"))
    parser.add_argument("--global-mount", type=Path, default=Path("/workspace-global"))
    parser.add_argument("--runtime-root", type=Path, default=Path("/root/arc-runtime"))
    parser.add_argument("--repo-dir", type=Path, default=Path("/root/arc-runtime/arc2"))
    parser.add_argument("--repo-url", default="")
    parser.add_argument("--python", type=Path, default=Path("/root/arc-runtime/env/5090-blackwell-env-v2/bin/python"))
    parser.add_argument("--run-id", help="execute exactly one predeclared run; required by the ephemeral off-Pod controller")
    parser.add_argument("--skip-persistent-sync", action="store_true", help="only valid for the single-GPU off-Pod-backup policy")
    parser.add_argument("--defer-local-scoring", action="store_true", help="only valid for the single-GPU off-Pod controller; candidate freeze is scored after verified local backup")
    return parser.parse_args()


def main() -> None:
    args = _args()
    queue, runs = _load_queue(args.queue_config)
    ephemeral = queue.get("execution_policy") == "single_gpu_ephemeral_offpod_backup"
    if not args.bootstrap_script.is_file() or not args.sync_script.is_file():
        raise FileNotFoundError("queue launcher is missing bootstrap/sync scripts")
    if not ephemeral and (not args.environment_lock or not args.environment_lock.is_file() or not args.environment_manifest or not args.environment_manifest.is_file() or not args.model_manifest or not args.model_manifest.is_file()):
        raise FileNotFoundError("persistent queue requires environment lock/manifest and model manifest")
    for optional in (args.environment_lock, args.environment_manifest, args.model_manifest):
        if optional is not None and not optional.is_file():
            raise FileNotFoundError(f"environment identity input is missing: {optional}")
    if ephemeral and not args.run_id:
        raise ValueError("ephemeral Phase-1 requires --run-id so the local controller can back up every completed run before dispatching the next")
    if not ephemeral and args.skip_persistent_sync:
        raise ValueError("persistent queue may not skip atomic persistent sync")
    if args.defer_local_scoring and not (ephemeral and args.skip_persistent_sync):
        raise ValueError("deferred scoring is only valid for the single-GPU off-Pod backup controller")
    inputs = _require_inputs(args, queue)
    bootstrap_ready = _environment_preflight(args, queue)
    if not args.python.is_file():
        raise FileNotFoundError(f"ENVIRONMENT_PREFLIGHT_FAILED: validated environment Python missing: {args.python}")
    run_root = args.runtime_root / "active_run" / queue["queue_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    if args.run_id:
        runs = tuple(spec for spec in runs if spec.run_id == args.run_id)
        if len(runs) != 1:
            raise ValueError(f"run ID is not in the frozen queue: {args.run_id}")
    fatal = threading.Event()
    errors: list[str] = []
    errors_lock = threading.Lock()

    def worker(physical_gpu_id: int) -> None:
        selected = sorted((spec for spec in runs if spec.physical_gpu_id == physical_gpu_id), key=lambda spec: spec.queue_position)
        for spec in selected:
            if fatal.is_set():
                return
            run_dir = run_root / spec.run_id
            if run_dir.exists():
                with errors_lock:
                    errors.append(f"{spec.run_id}: local active-run directory already exists; refusing ambiguous reuse")
                fatal.set()
                return
            run_dir.mkdir(parents=True)
            try:
                with (run_dir / "events.jsonl").open("a", encoding="utf-8") as events:
                    _write_event(events, {"event": "PHASE1_RUN_START", "queue_id": queue["queue_id"], "run_id": spec.run_id, "physical_gpu_id": spec.physical_gpu_id, "generation_micro_batch_size": spec.generation_micro_batch_size})
                    _freeze_and_score_run(args, queue, inputs, bootstrap_ready, spec, run_dir, events)
                if not args.skip_persistent_sync:
                    _sync_run(args, run_dir)
            except Exception as error:  # deliberate terminal queue guard
                with errors_lock:
                    errors.append(f"{spec.run_id}: {type(error).__name__}: {error}")
                fatal.set()
                return

    workers = [threading.Thread(target=worker, args=(gpu_id,), name=f"phase1-gpu-{gpu_id}", daemon=False) for gpu_id in sorted({spec.physical_gpu_id for spec in runs})]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    if errors:
        raise RuntimeError("PHASE1_QUEUE_STOPPED: " + " | ".join(errors))
    if args.run_id:
        print(json.dumps({"event": "PHASE1_RUN_COMPLETE", "queue_id": queue["queue_id"], "run_id": args.run_id, "local_run_dir": str(run_root / args.run_id), "persistent_sync_skipped": args.skip_persistent_sync}, sort_keys=True), flush=True)
        return
    summary_dir = _final_queue_summary(args, queue, runs, run_root, bootstrap_ready)
    _sync_run(args, summary_dir)
    print(json.dumps({"event": "PHASE1_QUEUE_COMPLETE", "queue_id": queue["queue_id"], "summary": str(summary_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
