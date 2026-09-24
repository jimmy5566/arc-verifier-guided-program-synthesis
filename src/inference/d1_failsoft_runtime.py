"""Shared fail-soft runtime state and finalization for D1 V2.

This module is deliberately CPU/model independent.  CUDA workers emit source
evidence; this layer owns task state, atomic checkpoints, failure provenance,
deadline completion, and the deterministic per-output D1 finalization used by
both the release notebook and the governed experiment workbench.

Scientific generation/scoring lives in the unchanged D1 implementation.  A
failure record never masquerades as model evidence, and release identity or
schema corruption always remains a hard failure.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from .d1_release_contract import (
    PORTFOLIO,
    ReleaseContractError,
    atomic_json,
    digest,
    runtime_manifest,
    select_record,
    validate_grid,
)


FAILSOFT_SCHEMA_VERSION = "ARC2_D1_FAILSOFT_RUNTIME_V1"
PENDING_SOURCE = "PENDING_INPUT_COPY"
FINAL_SOURCES = {
    "DUAL_SOURCE_D1",
    "SINGLE_SOURCE_TTT24",
    "SINGLE_SOURCE_TTT48",
    "COMPLETED_EMPTY_INPUT_COPY",
    "TASK_EXECUTION_FALLBACK",
    "WORKER_CRASH_FALLBACK",
    "DEADLINE_FALLBACK",
}
FALLBACK_SOURCES = {
    "TASK_EXECUTION_FALLBACK",
    "WORKER_CRASH_FALLBACK",
    "DEADLINE_FALLBACK",
}
SOURCE_STATES = {"SUCCESS", "COMPLETED_EMPTY", "FAILED"}


def _now() -> float:
    return time.time()


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Atomically replace a JSONL event stream.

    The parent is the sole writer.  Replacing the whole small control log makes
    a crash unable to leave a half JSON record that looks valid.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _empty_evidence(test_count: int) -> list[dict[str, Any]]:
    return [{"test_index": index, "candidates": []} for index in range(test_count)]


def _input_copy_record(
    task_id: str,
    manifest: Mapping[str, Any],
    source: str,
    reason: str,
    *,
    worker_id: int | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    if source not in FALLBACK_SOURCES:
        raise ReleaseContractError(f"invalid terminal fallback source: {source}")
    return {
        "task_id": task_id,
        "status": "FALLBACK",
        "fallback_source": source,
        "fallback_reason": str(reason),
        "worker_id": worker_id,
        "stage": stage,
        "release_identity": manifest["release_identity"],
    }


def _source_has_candidates(source: Mapping[str, Any]) -> bool:
    return source.get("status") == "SUCCESS" and isinstance(source.get("candidates"), list) and bool(source["candidates"])


def _validate_source(source_name: str, source: Any, test_count: int) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise ReleaseContractError(f"{source_name} source is missing or malformed")
    status = source.get("status")
    if status not in SOURCE_STATES:
        raise ReleaseContractError(f"{source_name} has invalid source status: {status}")
    candidates = source.get("candidates")
    evidence = source.get("per_output_evidence")
    if not isinstance(candidates, list) or not isinstance(evidence, list):
        raise ReleaseContractError(f"{source_name} lacks candidate/evidence lists")
    indices = [row.get("test_index") for row in evidence if isinstance(row, Mapping)]
    if indices != list(range(test_count)):
        raise ReleaseContractError(f"{source_name} per-output evidence mapping is malformed")
    if status == "SUCCESS" and not candidates:
        raise ReleaseContractError(f"{source_name} SUCCESS source is empty")
    if status in {"COMPLETED_EMPTY", "FAILED"} and candidates:
        raise ReleaseContractError(f"{source_name} {status} source contains candidates")
    if status == "FAILED" and not source.get("error"):
        raise ReleaseContractError(f"{source_name} failed without explicit error provenance")
    return copy.deepcopy(dict(source))


def validate_task_record(task_id: str, manifest: Mapping[str, Any], record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ReleaseContractError(f"task {task_id} record is not a mapping")
    if record.get("task_id") != task_id or record.get("release_identity") != manifest.get("release_identity"):
        raise ReleaseContractError(f"task {task_id} record identity mismatch")
    status = record.get("status")
    if status == "FALLBACK":
        if record.get("fallback_source") not in FALLBACK_SOURCES:
            raise ReleaseContractError(f"task {task_id} fallback source is invalid")
        return copy.deepcopy(dict(record))
    if status != "SUCCESS":
        raise ReleaseContractError(f"task {task_id} has invalid terminal status: {status}")
    sources = record.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(PORTFOLIO):
        raise ReleaseContractError(f"task {task_id} must retain both source records")
    test_count = len(manifest["tasks"][task_id]["test_outputs"])
    result = copy.deepcopy(dict(record))
    result["sources"] = {
        name: _validate_source(name, sources[name], test_count)
        for name in PORTFOLIO
    }
    if all(source["status"] == "FAILED" for source in result["sources"].values()):
        raise ReleaseContractError(f"task {task_id} has no usable completed source")
    return result


def _checkpoint_payload(task_id: str, manifest: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": FAILSOFT_SCHEMA_VERSION,
        "release_identity": manifest["release_identity"],
        "task_id": task_id,
        "task_contract": manifest["tasks"][task_id],
        "record": dict(record),
    }


def load_checkpoint(path: Path, task_id: str, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = (
        FAILSOFT_SCHEMA_VERSION,
        manifest.get("release_identity"),
        task_id,
        manifest.get("tasks", {}).get(task_id),
    )
    actual = (
        payload.get("schema_version"),
        payload.get("release_identity"),
        payload.get("task_id"),
        payload.get("task_contract"),
    )
    if actual != expected:
        return None
    try:
        return validate_task_record(task_id, manifest, payload.get("record"))
    except ReleaseContractError:
        return None


def record_source(record: Mapping[str, Any]) -> str:
    if record.get("status") == "FALLBACK":
        source = str(record.get("fallback_source"))
        if source not in FALLBACK_SOURCES:
            raise ReleaseContractError("invalid fallback source")
        return source
    sources = record.get("sources")
    if not isinstance(sources, Mapping):
        raise ReleaseContractError("successful record lacks source evidence")
    available = [name for name in PORTFOLIO if _source_has_candidates(sources[name])]
    if len(available) == 2:
        return "DUAL_SOURCE_D1"
    if available == ["TTT24"]:
        return "SINGLE_SOURCE_TTT24"
    if available == ["TTT48"]:
        return "SINGLE_SOURCE_TTT48"
    if all(sources[name].get("status") == "COMPLETED_EMPTY" for name in PORTFOLIO):
        return "COMPLETED_EMPTY_INPUT_COPY"
    return "TASK_EXECUTION_FALLBACK"


def _neutral_empty_source(test_count: int) -> dict[str, Any]:
    return {"status": "COMPLETED_EMPTY", "candidates": [], "per_output_evidence": _empty_evidence(test_count)}


def _input_copy_selection(task_id: str, task: Mapping[str, Any], source: str, reason: str) -> dict[str, Any]:
    outputs = []
    for test_index, example in enumerate(task["test"]):
        grid = copy.deepcopy(validate_grid(example["input"]))
        outputs.append({
            "test_index": test_index,
            "attempt_1": grid,
            "attempt_2": copy.deepcopy(grid),
            "ordered_grid_keys": [],
            "likelihood_ranks": {},
            "likelihood_rrf": {},
            "candidate_pool": [],
            "selection_source": source,
            "fallback_reason": reason,
        })
    return {"status": "SUCCESS", "method": "INPUT_COPY_FAILSOFT_V1", "task_id": task_id, "outputs": outputs}


def finalize_failsoft(
    challenges: Mapping[str, Any],
    release_config: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build complete predictions or hard-fail on global identity/schema faults."""
    manifest = runtime_manifest(challenges, release_config)
    if artifact.get("schema_version") != FAILSOFT_SCHEMA_VERSION:
        raise ReleaseContractError("fail-soft artifact schema mismatch")
    if artifact.get("release_identity") != manifest["release_identity"]:
        raise ReleaseContractError("fail-soft artifact identity mismatch")
    records = artifact.get("records")
    if not isinstance(records, Mapping) or set(records) != set(manifest["task_ids"]):
        raise ReleaseContractError("fail-soft artifact lacks complete task coverage")

    selections: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    task_sources: dict[str, str] = {}
    output_source_counts = {name: 0 for name in FINAL_SOURCES}
    task_source_counts = {name: 0 for name in FINAL_SOURCES}
    for task_id in manifest["task_ids"]:
        record = validate_task_record(task_id, manifest, records[task_id])
        source = record_source(record)
        task = challenges[task_id]
        if source in FALLBACK_SOURCES or source == "TASK_EXECUTION_FALLBACK":
            reason = str(record.get("fallback_reason", "no candidate-bearing source completed"))
            selection = _input_copy_selection(task_id, task, source, reason)
        elif source == "COMPLETED_EMPTY_INPUT_COPY":
            selection = _input_copy_selection(task_id, task, source, "both sources completed normally with empty pools")
        else:
            test_count = len(task["test"])
            selection_record = copy.deepcopy(record)
            for source_name in PORTFOLIO:
                if selection_record["sources"][source_name]["status"] == "FAILED":
                    selection_record["sources"][source_name] = _neutral_empty_source(test_count)
            selection = select_record(
                selection_record,
                manifest["tasks"][task_id],
                [example["input"] for example in task["test"]],
            )
            for output in selection["outputs"]:
                output["selection_source"] = source
        outputs = []
        for expected_index, output in enumerate(selection["outputs"]):
            if int(output.get("test_index", -1)) != expected_index:
                raise ReleaseContractError(f"test index mapping failure for {task_id}")
            outputs.append({
                "attempt_1": validate_grid(output["attempt_1"]),
                "attempt_2": validate_grid(output["attempt_2"]),
            })
            output_source_counts[source] += 1
        if len(outputs) != len(task["test"]):
            raise ReleaseContractError(f"test output coverage failure for {task_id}")
        selections[task_id] = selection
        predictions[task_id] = outputs
        task_sources[task_id] = source
        task_source_counts[source] += 1

    selection_artifact = {
        "schema_version": FAILSOFT_SCHEMA_VERSION,
        "release_identity": manifest["release_identity"],
        "task_ids": manifest["task_ids"],
        "records": selections,
        "task_sources": task_sources,
        "solutions_opened": False,
    }
    provenance = {
        "status": "FAILSOFT_COMPLETE_COVERAGE_PASS",
        "schema_version": FAILSOFT_SCHEMA_VERSION,
        "release_identity": manifest["release_identity"],
        "challenge_sha256": manifest["challenge_sha256"],
        "task_count": len(predictions),
        "test_output_count": sum(len(rows) for rows in predictions.values()),
        "task_source_counts": task_source_counts,
        "output_source_counts": output_source_counts,
        "solutions_opened": False,
    }
    return selection_artifact, predictions, provenance


class FailsoftRuntime:
    """Parent-owned state machine for task, worker, checkpoint and deadline state."""

    def __init__(
        self,
        challenges: Mapping[str, Any],
        release_config: Mapping[str, Any],
        artifact_dir: Path,
        *,
        checkpoint_dir: Path | None = None,
        resume: bool = True,
        clock: Any = _now,
    ) -> None:
        self.challenges = copy.deepcopy(dict(challenges))
        self.release_config = copy.deepcopy(dict(release_config))
        self.manifest = runtime_manifest(self.challenges, self.release_config)
        self.artifact_dir = Path(artifact_dir)
        self.checkpoint_dir = Path(checkpoint_dir) / "tasks" if checkpoint_dir is not None else self.artifact_dir / "tasks"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.started_at = float(clock())
        events_path = self.artifact_dir / "events.jsonl"
        self.events: list[dict[str, Any]] = []
        if events_path.is_file():
            try:
                self.events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except (OSError, json.JSONDecodeError) as exc:
                raise ReleaseContractError("existing runtime event stream is corrupt") from exc
        self.failures: list[dict[str, Any]] = []
        self.workers: dict[int, dict[str, Any]] = {}
        self.records: dict[str, dict[str, Any]] = {}
        self.pending: set[str] = set(self.manifest["task_ids"])
        self.fallback_map = {
            task_id: {
                "source": PENDING_SOURCE,
                "outputs": [
                    {"attempt_1": copy.deepcopy(row["input"]), "attempt_2": copy.deepcopy(row["input"])}
                    for row in self.challenges[task_id]["test"]
                ],
            }
            for task_id in self.manifest["task_ids"]
        }
        if resume:
            for task_id in self.manifest["task_ids"]:
                recovered = load_checkpoint(self.checkpoint_dir / f"{task_id}.json", task_id, self.manifest)
                if recovered is not None:
                    self.records[task_id] = recovered
                    self.pending.discard(task_id)
                    self.fallback_map[task_id]["source"] = record_source(recovered)
        self._event("RUNTIME_INITIALIZED", resumed_task_count=len(self.records), fallback_coverage=len(self.fallback_map))
        self._persist()

    def _event(self, event: str, **fields: Any) -> None:
        self.events.append({"event": event, "timestamp": float(self.clock()), **fields})

    def _counts(self) -> dict[str, int]:
        values = [record_source(record) for record in self.records.values()]
        return {
            "dual_source": values.count("DUAL_SOURCE_D1"),
            "single_source": values.count("SINGLE_SOURCE_TTT24") + values.count("SINGLE_SOURCE_TTT48"),
            "completed_empty": values.count("COMPLETED_EMPTY_INPUT_COPY"),
            "task_fallback": values.count("TASK_EXECUTION_FALLBACK"),
            "worker_crash_fallback": values.count("WORKER_CRASH_FALLBACK"),
            "deadline_fallback": values.count("DEADLINE_FALLBACK"),
        }

    def _persist(self) -> None:
        _atomic_jsonl(self.artifact_dir / "events.jsonl", self.events)
        atomic_json(self.artifact_dir / "failures.json", {
            "schema_version": FAILSOFT_SCHEMA_VERSION,
            "release_identity": self.manifest["release_identity"],
            "failures": self.failures,
        })
        atomic_json(self.artifact_dir / "runtime_status.json", {
            "schema_version": FAILSOFT_SCHEMA_VERSION,
            "release_identity": self.manifest["release_identity"],
            "state": "COMPLETE" if not self.pending else "RUNNING",
            "started_at": self.started_at,
            "updated_at": float(self.clock()),
            "completed_task_count": len(self.records),
            "pending_task_count": len(self.pending),
            "workers": self.workers,
            "fallback_map": self.fallback_map,
            "counts": self._counts(),
        })
        atomic_json(self.artifact_dir / "task_summary.json", {
            "schema_version": FAILSOFT_SCHEMA_VERSION,
            "release_identity": self.manifest["release_identity"],
            "tasks": {
                task_id: {
                    "terminal": task_id in self.records,
                    "source": record_source(self.records[task_id]) if task_id in self.records else PENDING_SOURCE,
                    "worker_id": self.records.get(task_id, {}).get("worker_id"),
                    "reason": self.records.get(task_id, {}).get("fallback_reason"),
                }
                for task_id in self.manifest["task_ids"]
            },
            "counts": self._counts(),
        })

    def worker_ready(self, worker_id: int, **metadata: Any) -> None:
        self.workers[worker_id] = {
            "worker_id": worker_id,
            "state": "READY",
            "active_task_id": None,
            "stage": "READY",
            "last_heartbeat": float(self.clock()),
            **metadata,
        }
        self._event("WORKER_READY", worker_id=worker_id, **metadata)
        self._persist()

    def task_started(self, worker_id: int, task_id: str) -> None:
        if task_id not in self.pending or task_id in self.records:
            raise ReleaseContractError(f"task {task_id} cannot be dispatched twice")
        worker = self.workers.get(worker_id)
        if not worker or worker.get("active_task_id") is not None or worker.get("state") in {"DEAD", "UNSAFE"}:
            raise ReleaseContractError(f"worker {worker_id} is not available")
        worker.update({"state": "BUSY", "active_task_id": task_id, "stage": "TASK_START", "last_heartbeat": float(self.clock())})
        self._event("TASK_START", worker_id=worker_id, task_id=task_id)
        self._persist()

    def heartbeat(self, worker_id: int, task_id: str | None, stage: str, **metadata: Any) -> None:
        worker = self.workers.get(worker_id)
        if not worker:
            raise ReleaseContractError(f"heartbeat from unknown worker {worker_id}")
        if task_id is not None and worker.get("active_task_id") != task_id:
            raise ReleaseContractError(f"worker {worker_id} heartbeat task mismatch")
        worker.update({"stage": stage, "last_heartbeat": float(self.clock())})
        self._event("WORKER_HEARTBEAT", worker_id=worker_id, task_id=task_id, stage=stage, **metadata)
        self._persist()

    def _complete(self, task_id: str, record: Mapping[str, Any], event: str) -> dict[str, Any]:
        if task_id not in self.pending or task_id in self.records:
            raise ReleaseContractError(f"duplicate terminal record for {task_id}")
        validated = validate_task_record(task_id, self.manifest, record)
        checkpoint = self.checkpoint_dir / f"{task_id}.json"
        atomic_json(checkpoint, _checkpoint_payload(task_id, self.manifest, validated))
        recovered = load_checkpoint(checkpoint, task_id, self.manifest)
        if recovered is None:
            raise ReleaseContractError(f"atomic checkpoint validation failed for {task_id}")
        self.records[task_id] = recovered
        self.pending.discard(task_id)
        self.fallback_map[task_id]["source"] = record_source(recovered)
        worker_id = recovered.get("worker_id")
        if isinstance(worker_id, int) and worker_id in self.workers:
            worker = self.workers[worker_id]
            if worker.get("state") not in {"DEAD", "UNSAFE"}:
                worker.update({"state": "READY", "active_task_id": None, "stage": "READY", "last_heartbeat": float(self.clock())})
        self._event(event, task_id=task_id, worker_id=worker_id, source=record_source(recovered))
        self._persist()
        return recovered

    def accept_result(self, task_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
        return self._complete(task_id, record, "TASK_RESULT_FROZEN")

    def fallback(
        self,
        task_id: str,
        source: str,
        reason: str,
        *,
        worker_id: int | None = None,
        stage: str | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = _input_copy_record(task_id, self.manifest, source, reason, worker_id=worker_id, stage=stage)
        failure = {"task_id": task_id, "source": source, "reason": reason, "worker_id": worker_id, "stage": stage}
        if error:
            failure["error"] = copy.deepcopy(dict(error))
        self.failures.append(failure)
        return self._complete(task_id, record, source)

    def worker_died(self, worker_id: int, reason: str, *, unsafe: bool = False) -> None:
        worker = self.workers.setdefault(worker_id, {"worker_id": worker_id, "active_task_id": None})
        active = worker.get("active_task_id")
        stage = worker.get("stage")
        worker.update({"state": "UNSAFE" if unsafe else "DEAD", "last_heartbeat": float(self.clock())})
        self._event("WORKER_UNSAFE" if unsafe else "WORKER_DIED", worker_id=worker_id, active_task_id=active, stage=stage, reason=reason)
        if isinstance(active, str) and active in self.pending:
            self.fallback(active, "WORKER_CRASH_FALLBACK", reason, worker_id=worker_id, stage=str(stage))
            worker["active_task_id"] = None
        else:
            self._persist()

    def finalize_deadline(self, reason: str = "global dispatch/finalization deadline reached") -> None:
        for task_id in list(self.manifest["task_ids"]):
            if task_id in self.pending:
                worker_id = next((wid for wid, state in self.workers.items() if state.get("active_task_id") == task_id), None)
                stage = self.workers.get(worker_id, {}).get("stage") if worker_id is not None else "UNSTARTED"
                self.fallback(task_id, "DEADLINE_FALLBACK", reason, worker_id=worker_id, stage=str(stage))
        self._event("DEADLINE_FINALIZATION_COMPLETE", task_count=len(self.records))
        self._persist()

    def finalize_worker_exhaustion(self, reason: str = "no safe workers remain") -> None:
        for task_id in list(self.manifest["task_ids"]):
            if task_id in self.pending:
                self.fallback(task_id, "WORKER_CRASH_FALLBACK", reason, stage="UNSTARTED")
        self._event("WORKER_EXHAUSTION_FINALIZATION_COMPLETE", task_count=len(self.records))
        self._persist()

    def artifact(self) -> dict[str, Any]:
        if self.pending or set(self.records) != set(self.manifest["task_ids"]):
            raise ReleaseContractError(f"fail-soft runtime is not terminal: {sorted(self.pending)}")
        artifact = {
            "schema_version": FAILSOFT_SCHEMA_VERSION,
            "release_identity": self.manifest["release_identity"],
            "manifest": self.manifest,
            "records": {task_id: self.records[task_id] for task_id in self.manifest["task_ids"]},
            "runtime_summary": {
                "counts": self._counts(),
                "worker_count": len(self.workers),
                "elapsed_seconds": float(self.clock()) - self.started_at,
                "solutions_opened": False,
            },
            "solutions_opened": False,
        }
        # Invoke the pure finalizer now so corrupt selector/finalizer/schema is
        # a global hard failure rather than a valid-looking fallback artifact.
        finalize_failsoft(self.challenges, self.release_config, artifact)
        self._event("RUNTIME_COMPLETE", counts=self._counts())
        self._persist()
        return artifact
