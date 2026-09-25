"""Serial, single-RTX-5090 execution queue for Blackwell Phase 2.

The queue owns no model math. It invokes the frozen Eval3 runtime entry point
with a single development task and freezes a self-contained record after each
stage before atomically syncing it to the mounted Global Volume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.phase2_backend_contract import BACKENDS, execution_for_backend, valid_frozen_artifact


QUEUE_ID = "5090-blackwell-unleashed-phase2-serial-v1"
BACKEND_ID = "blackwell_unleashed_v2_candidate"
TASK_ID = "5dbc8537"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _event(handle: Any, name: str, **fields: Any) -> None:
    payload = {"timestamp_unix": time.time(), "event": name, **fields}
    line = json.dumps(payload, sort_keys=True)
    handle.write(line + "\n")
    handle.flush()
    print(line, flush=True)


def _hash_tree(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name not in {"hashes.json", "SHA256SUMS"}}


def _validate_global_mount(path: Path) -> None:
    completed = subprocess.run(["findmnt", "-T", str(path)], text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"GLOBAL_VOLUME_UNAVAILABLE: {path}")


def _git_identity(repo_dir: Path) -> str:
    status = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain"], text=True, capture_output=True, check=True)
    if status.stdout.strip():
        raise RuntimeError("EXACT_SOURCE_REQUIRED: remote RunPod checkout is dirty")
    return subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()


def _is_ancestor(repo_dir: Path, older: str, newer: str) -> bool:
    return subprocess.run(["git", "-C", str(repo_dir), "merge-base", "--is-ancestor", older, newer], text=True, capture_output=True).returncode == 0


def _gpu_identity() -> dict[str, str]:
    completed = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    pieces = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(pieces) != 4 or "RTX 5090" not in pieces[2]:
        raise RuntimeError(f"RTX_5090_REQUIRED: {completed.stdout.strip()}")
    return {"physical_gpu_id": pieces[0], "uuid": pieces[1], "name": pieces[2], "driver_version": pieces[3]}


def _model_command(args: argparse.Namespace, run_dir: Path, execution: str) -> list[str]:
    return [
        str(args.python), str(args.repo_dir / "scripts" / "run_eval3_runtime_opt.py"),
        "--manifest", str(args.manifest), "--reference-config", str(args.reference_config),
        "--challenge-path", str(args.challenge_path), "--model-path", str(args.model_path),
        "--native-config-dir", str(args.native_config_dir), "--output", str(run_dir / "candidates_frozen.json"),
        "--checkpoint-dir", str(run_dir / "checkpoints"), "--mode", "serial",
        "--generation-micro-batch-size", "4", "--backend-id", BACKEND_ID,
        "--generation-execution", execution, "--task-id", TASK_ID,
    ]


def _run_child(args: argparse.Namespace, run_dir: Path, execution: str, events: Any) -> None:
    environment = dict(os.environ)
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0", "ARC2_PHYSICAL_GPU_ID": "0", "TRITON_PTXAS_PATH": str(args.ptxas_path),
        "HF_HOME": str(args.runtime_root / "hf_cache"), "HUGGINGFACE_HUB_CACHE": str(args.runtime_root / "hf_cache" / "hub"),
        "TORCH_HOME": str(args.runtime_root / "torch_cache"), "TRITON_CACHE_DIR": str(args.runtime_root / "triton_cache"),
        "UV_CACHE_DIR": str(args.runtime_root / "uv_cache"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    })
    command = _model_command(args, run_dir, execution)
    _event(events, "PHASE2_MODEL_START", execution=execution, command=command)
    process = subprocess.Popen(command, cwd=args.repo_dir, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        _event(events, "PHASE2_MODEL_STDOUT", line=line.rstrip("\n"))
    if process.wait():
        raise RuntimeError(f"MODEL_PROCESS_FAILED execution={execution} exit={process.returncode}")


def _score(args: argparse.Namespace, run_dir: Path, events: Any) -> dict[str, Any]:
    command = [
        str(args.python), str(args.repo_dir / "scripts" / "score_blackwell_phase2.py"),
        "--candidates", str(run_dir / "candidates_frozen.json"), "--solutions-path", str(args.solutions_path),
        "--task-id", TASK_ID, "--output", str(run_dir / "evaluation" / "report.json"),
    ]
    completed = subprocess.run(command, cwd=args.repo_dir, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, text=True, capture_output=True)
    _event(events, "PHASE2_POST_FREEZE_SCORE", returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    if completed.returncode:
        raise RuntimeError("POST_FREEZE_SCORE_FAILED")
    return _read(run_dir / "evaluation" / "report.json")


def _sync_verified(run_dir: Path, frozen_root: Path) -> Path:
    _validate_global_mount(frozen_root)
    destination = frozen_root / run_dir.name
    staging = frozen_root / f".{run_dir.name}.staging.{os.getpid()}"
    if destination.exists() or staging.exists():
        raise RuntimeError(f"IMMUTABLE_BACKUP_DESTINATION_EXISTS: {destination}")
    frozen_root.mkdir(parents=True, exist_ok=True)
    try:
        # Global Volume FUSE accepts ordinary file bytes and directory
        # creation, but rejects metadata operations such as chmod/copystat.
        # Do not use copytree: its metadata propagation turns an otherwise
        # complete immutable backup into a permission failure.
        for source in sorted(run_dir.rglob("*")):
            relative = source.relative_to(run_dir)
            target = staging / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as reader, target.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
        source_hashes = _hash_tree(run_dir)
        destination_hashes = _hash_tree(staging)
        if source_hashes != destination_hashes:
            raise RuntimeError("OFF_POD_BACKUP_HASH_MISMATCH")
        _atomic_json(staging / "hashes.json", {"algorithm": "sha256", "files": source_hashes})
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _metrics(candidates: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    record = candidates["records"][TASK_ID]
    generation = record["telemetry"]["generation"]
    whole = record["telemetry"]["whole_task"]
    return {
        "generation_seconds": generation["seconds"], "total_wall_seconds": whole["seconds"],
        "real_suffix_tokens": generation["generated_token_count"], "end_to_end_generation_tokens_per_second": generation["generation_tokens_per_second"],
        "gpu_utilization_avg_pct": generation.get("gpu_utilization_avg_pct"), "gpu_utilization_p90_pct": generation.get("gpu_utilization_p90_pct"), "gpu_utilization_max_pct": generation.get("gpu_utilization_max_pct"),
        "gpu_power_avg_w": generation.get("gpu_power_avg_w"), "gpu_power_p90_w": generation.get("gpu_power_p90_w"), "gpu_power_max_w": generation.get("gpu_power_max_w"),
        "peak_allocated_vram_mb": generation["peak_allocated_bytes"] / (1024 * 1024), "peak_reserved_vram_mb": generation["peak_reserved_bytes"] / (1024 * 1024),
        "active_sequence_trace": generation.get("active_sequence_trace", []), "finished_slot_waste_fraction": generation.get("finished_slot_waste_fraction"),
        "padding_ratio": generation.get("padding_ratio"), "kv_cache": record.get("KV_CACHE", {}), "compile_status": "ENABLED" if "torch_compile" in record.get("KV_CACHE", {}).get("generation_execution", "") else "DISABLED",
        "cuda_graph_status": "NOT_APPLICABLE: dynamic greedy completion has variable active-batch shapes", "invalid_candidate_count": record["invalid_candidate_count"],
        "candidate_count": record["unique_candidate_count"], "ANY_OF_K": report["ANY_OF_K"],
    }


def _freeze_not_applicable(run_dir: Path, backend: str, reason: str, source_commit: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(run_dir / "run.json", {"queue_id": QUEUE_ID, "backend": backend, "status": "NOT_APPLICABLE", "reason": reason, "source_commit": source_commit})
    _atomic_json(run_dir / "hashes.json", {"algorithm": "sha256", "files": _hash_tree(run_dir)})


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "reference_config", "challenge_path", "solutions_path", "model_path", "native_config_dir", "ptxas_path", "python", "repo_dir", "runtime_root", "global_root"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="resume only an already valid, unmodified Phase-2 run after a backup interruption")
    return parser.parse_args()


def main() -> None:
    args = _args()
    source_commit = _git_identity(args.repo_dir)
    if source_commit != os.environ.get("ARC2_EXPECTED_SOURCE_COMMIT", source_commit):
        raise RuntimeError("EXACT_SOURCE_REQUIRED: checkout does not match ARC2_EXPECTED_SOURCE_COMMIT")
    if not args.ptxas_path.is_file() or not os.access(args.ptxas_path, os.X_OK):
        raise RuntimeError("PTXAS_REQUIRED")
    if not args.python.is_file():
        raise RuntimeError("VALIDATED_ENVIRONMENT_PYTHON_REQUIRED")
    _gpu_identity()
    manifest = _read(args.manifest)
    if TASK_ID not in manifest.get("task_ids", []):
        raise RuntimeError("PHASE2_TASK_NOT_IN_FROZEN_EVAL3_MANIFEST")
    for path in (args.reference_config, args.challenge_path, args.solutions_path, args.model_path, args.native_config_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    run_root = args.runtime_root / "active_run" / QUEUE_ID
    if run_root.exists() and not args.resume:
        raise RuntimeError(f"REFUSING_AMBIGUOUS_REUSE: {run_root}")
    run_root.mkdir(parents=True, exist_ok=args.resume)
    frozen_root = args.global_root / "frozen-runs"
    successful: set[str] = set()
    summaries: dict[str, Any] = {}
    for index, backend in enumerate(BACKENDS, start=1):
        execution, reason = execution_for_backend(backend, successful=successful)
        run_name = f"{QUEUE_ID}-{index:02d}-{backend}"
        run_dir = run_root / run_name
        if run_dir.exists():
            if not args.resume:
                raise RuntimeError(f"REFUSING_AMBIGUOUS_REUSE: {run_dir}")
            prior = _read(run_dir / "run.json") if (run_dir / "run.json").is_file() else {}
            prior_source = prior.get("source_commit")
            if prior.get("status") != "SUCCESS" or not isinstance(prior_source, str):
                raise RuntimeError(f"RESUME_REJECTED: {run_dir} is not a successful frozen run")
            # A resume may run under a descendant commit that fixes only the
            # FUSE backup transport.  Keep the original execution commit in
            # the evidence and refuse unrelated source histories.
            if prior_source != source_commit and not _is_ancestor(args.repo_dir, prior_source, source_commit):
                raise RuntimeError(f"RESUME_REJECTED: {run_dir} source is not an ancestor of the backup-resume source")
            candidates = _read(run_dir / "candidates_frozen.json")
            if not valid_frozen_artifact(candidates, TASK_ID):
                raise RuntimeError(f"RESUME_REJECTED: {run_dir} candidate freeze is invalid")
            report = _read(run_dir / "evaluation" / "report.json")
            metrics = _read(run_dir / "telemetry.json")
            if report.get("status") != "COMPLETE_SCORED_AFTER_CANDIDATE_FREEZE" or metrics.get("invalid_candidate_count") != 0:
                raise RuntimeError(f"RESUME_REJECTED: {run_dir} score or structural metrics are invalid")
            successful.add(backend)
            summaries[backend] = {"status": "SUCCESS", "execution": prior.get("execution"), "execution_source_commit": prior_source, "resume_controller_commit": source_commit, **metrics, "resumed_without_model_execution": True}
            destination = frozen_root / run_dir.name
            if destination.exists():
                summaries[backend]["persistent_run"] = str(destination)
            else:
                summaries[backend]["persistent_run"] = str(_sync_verified(run_dir, frozen_root))
            continue
        if execution is None:
            _freeze_not_applicable(run_dir, backend, reason or "NOT_APPLICABLE", source_commit)
            persistent = _sync_verified(run_dir, frozen_root)
            summaries[backend] = {"status": "NOT_APPLICABLE", "reason": reason, "persistent_run": str(persistent)}
            continue
        run_dir.mkdir(parents=True)
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as events:
            _event(events, "PHASE2_RUN_START", backend=backend, execution=execution, source_commit=source_commit, task_id=TASK_ID)
            _atomic_json(run_dir / "config_resolved.json", {"backend": backend, "generation_execution": execution, "generation_micro_batch_size": 4, "mode": "serial", "task_id": TASK_ID, "source_commit": source_commit, "scientific_config_changed": False})
            try:
                _run_child(args, run_dir, execution, events)
                candidates = _read(run_dir / "candidates_frozen.json")
                if not valid_frozen_artifact(candidates, TASK_ID):
                    raise RuntimeError("INVALID_CANDIDATE_FREEZE")
                report = _score(args, run_dir, events)
                metrics = _metrics(candidates, report)
                _atomic_json(run_dir / "telemetry.json", metrics)
                _atomic_json(run_dir / "run.json", {"queue_id": QUEUE_ID, "backend": backend, "execution": execution, "status": "SUCCESS", "source_commit": source_commit, "task_id": TASK_ID, "candidate_artifact_sha256": _sha256_file(run_dir / "candidates_frozen.json"), "metrics": metrics})
                successful.add(backend)
                summaries[backend] = {"status": "SUCCESS", "execution": execution, **metrics}
                _event(events, "PHASE2_RUN_FROZEN", backend=backend, generation_seconds=metrics["generation_seconds"], solutions_opened_after_freeze=True)
            except Exception as error:
                _atomic_json(run_dir / "run.json", {"queue_id": QUEUE_ID, "backend": backend, "execution": execution, "status": "FAILED", "source_commit": source_commit, "error_type": type(error).__name__, "error": str(error)})
                _event(events, "PHASE2_RUN_FAILURE", backend=backend, error_type=type(error).__name__, error=str(error))
                # A backend-specific compile/compaction incompatibility is
                # evidence for this queue, not a scientific or environment
                # failure.  Preserve it, verify the backup, and continue.
                # Reference/static-cache failures remain terminal because they
                # are the baseline execution path or core cache mechanism.
                if backend in {"compaction", "torch_compile", "best_combined"}:
                    summaries[backend] = {"status": "NOT_APPLICABLE", "execution": execution, "reason": f"{type(error).__name__}: {error}"}
                else:
                    raise
        _atomic_json(run_dir / "hashes.json", {"algorithm": "sha256", "files": _hash_tree(run_dir)})
        persistent = _sync_verified(run_dir, frozen_root)
        summaries[backend]["persistent_run"] = str(persistent)
    valid = [(name, data) for name, data in summaries.items() if data.get("status") == "SUCCESS" and data["invalid_candidate_count"] == 0]
    if not valid:
        raise RuntimeError("NO_STRUCTURALLY_VALID_PHASE2_BACKEND")
    promoted_name, promoted = min(valid, key=lambda item: (float(item[1]["generation_seconds"]), item[0]))
    summary = {"queue_id": QUEUE_ID, "source_commit": source_commit, "task_id": TASK_ID, "runs": summaries, "promotion": {"backend": BACKEND_ID, "promoted_phase2_run": promoted_name, "generation_seconds": promoted["generation_seconds"], "scientific_config_changed": False}, "eval60_started": False, "ttt12_started": False}
    summary_dir = run_root / f"{QUEUE_ID}-summary"
    summary_dir.mkdir()
    _atomic_json(summary_dir / "phase2_queue_summary.json", summary)
    _atomic_json(summary_dir / "hashes.json", {"algorithm": "sha256", "files": _hash_tree(summary_dir)})
    persistent = _sync_verified(summary_dir, frozen_root)
    print(json.dumps({"event": "PHASE2_QUEUE_COMPLETE", "source_commit": source_commit, "summary": str(persistent), "promoted": promoted_name}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
