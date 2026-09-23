"""Fail-closed runtime contract for the fixed TTT24/TTT48 4+4 D1 release.

This is deliberately model-free.  Workers provide frozen evidence; this
module validates it against the challenge mounted for *this* run and performs
the deterministic per-test-output D1 selection.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from .selector_d1 import attempts_from_order, d1_order, verify_d1_scope


SCHEMA_VERSION = "ARC2_D1_RELEASE_V2"
EMPTY_POOL_POLICY = "ARC2_D1_COMPLETED_EMPTY_INPUT_COPY_V1"
PORTFOLIO = {
    "TTT24": ("flip_lr", "flip_ud", "transpose", "anti_transpose"),
    "TTT48": ("identity", "rot90", "flip_ud", "anti_transpose"),
}


class ReleaseContractError(ValueError):
    """Evidence or runtime data does not satisfy the fail-closed contract."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def grid_key(grid: list[list[int]]) -> str:
    validate_grid(grid)
    return canonical(grid)


def validate_grid(grid: Any) -> list[list[int]]:
    if not isinstance(grid, list) or not grid or not all(isinstance(row, list) and row for row in grid):
        raise ReleaseContractError("grid must be a non-empty two-dimensional list")
    width = len(grid[0])
    if any(len(row) != width or any(type(cell) is not int or not 0 <= cell <= 9 for cell in row) for row in grid):
        raise ReleaseContractError("grid must be rectangular ARC colors")
    return grid


def runtime_manifest(challenges: Mapping[str, Any], release_config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the entire task and test-index contract from the challenge file."""
    if not isinstance(challenges, Mapping) or not challenges:
        raise ReleaseContractError("mounted challenge must be a non-empty task mapping")
    tasks: dict[str, Any] = {}
    for task_id in sorted(challenges):
        task = challenges[task_id]
        tests = task.get("test") if isinstance(task, Mapping) else None
        if not isinstance(tests, list) or not tests:
            raise ReleaseContractError(f"task {task_id} has no test inputs")
        inputs = []
        for index, example in enumerate(tests):
            if not isinstance(example, Mapping) or "input" not in example:
                raise ReleaseContractError(f"task {task_id} test index {index} lacks input")
            validate_grid(example["input"])
            inputs.append({"test_index": index, "input_sha256": digest(example["input"])})
        tasks[str(task_id)] = {"task_sha256": digest(task), "test_outputs": inputs}
    config = dict(release_config)
    for key in ("model_identity", "ttt24_recipe", "ttt48_recipe", "generation", "scoring"):
        if key not in config:
            raise ReleaseContractError(f"release configuration missing {key}")
    return {
        "schema_version": SCHEMA_VERSION,
        "challenge_sha256": digest(challenges),
        "task_ids": sorted(tasks),
        "tasks": tasks,
        "release_config_sha256": digest(config),
        "release_identity": digest({"schema_version": SCHEMA_VERSION, "challenge": tasks, "challenge_sha256": digest(challenges), "config": config}),
    }


def checkpoint_payload(task_id: str, manifest: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "release_identity": manifest["release_identity"], "task_id": task_id, "task_contract": manifest["tasks"][task_id], "record": dict(record)}


def valid_checkpoint(path: Path, task_id: str, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (payload.get("schema_version"), payload.get("release_identity"), payload.get("task_id"), payload.get("task_contract")) != (SCHEMA_VERSION, manifest.get("release_identity"), task_id, manifest.get("tasks", {}).get(task_id)):
        return None
    record = payload.get("record")
    if not isinstance(record, Mapping) or record.get("status") != "SUCCESS" or record.get("task_id") != task_id or record.get("release_identity") != manifest.get("release_identity"):
        return None
    sources = record.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(PORTFOLIO):
        return None
    for source in sources.values():
        if not isinstance(source, Mapping) or not isinstance(source.get("candidates"), list):
            return None
        if source.get("status", "SUCCESS") != ("SUCCESS" if source["candidates"] else "COMPLETED_EMPTY"):
            return None
        evidence = source.get("per_output_evidence")
        if not isinstance(evidence, list) or {row.get("test_index") for row in evidence if isinstance(row, Mapping)} != {row["test_index"] for row in manifest["tasks"][task_id]["test_outputs"]}:
            return None
    return dict(record)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _tags(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """All raw view tags supporting a deduplicated candidate, in stable order."""
    values = candidate.get("support_augmentations", [])
    if isinstance(values, list) and values and all(isinstance(item, Mapping) for item in values):
        return tuple(str(item.get("geometry", "")) for item in values)
    if isinstance(candidate.get("augmentation"), Mapping):
        return (str(candidate["augmentation"].get("geometry", "")),)
    return (str(candidate.get("geometry", "")),)


def _evidence_by_index(source: Mapping[str, Any], test_index: int) -> dict[int, Mapping[str, Any]]:
    evidence = source.get("per_output_evidence")
    if not isinstance(evidence, list):
        raise ReleaseContractError("missing per-output likelihood evidence")
    row = next((item for item in evidence if int(item.get("test_index", -1)) == test_index), None)
    if not isinstance(row, Mapping) or not isinstance(row.get("candidates"), list):
        raise ReleaseContractError(f"missing likelihood evidence for test index {test_index}")
    result: dict[int, Mapping[str, Any]] = {}
    for item in row["candidates"]:
        index = int(item.get("candidate_index", -1))
        if index in result or "original_log_likelihood" not in item:
            raise ReleaseContractError("ambiguous or incomplete likelihood evidence")
        result[index] = item
    return result


def pool_for_output(sources: Mapping[str, Mapping[str, Any]], test_index: int) -> list[dict[str, Any]]:
    """Reconstruct source-local B-RRF evidence for one output, then D1 can rank it."""
    grouped: dict[str, dict[str, Any]] = {}
    for source_name, wanted_tags in PORTFOLIO.items():
        source = sources.get(source_name)
        if not isinstance(source, Mapping) or not isinstance(source.get("candidates"), list):
            raise ReleaseContractError(f"missing {source_name} candidate source")
        if not source["candidates"] and source.get("status") != "COMPLETED_EMPTY":
            raise ReleaseContractError(f"{source_name} empty pool lacks completed-empty provenance")
        if source["candidates"] and source.get("status", "SUCCESS") != "SUCCESS":
            raise ReleaseContractError(f"{source_name} candidate source is not successful")
        evidence = _evidence_by_index(source, test_index)
        if not source["candidates"] and evidence:
            raise ReleaseContractError(f"{source_name} empty pool has candidate evidence")
        per_grid: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for index, candidate in enumerate(source["candidates"]):
            selected_tags = tuple(tag for tag in _tags(candidate) if tag in wanted_tags)
            if not selected_tags:
                continue
            predictions = candidate.get("prediction")
            if not isinstance(predictions, list) or test_index >= len(predictions):
                raise ReleaseContractError(f"candidate lacks test prediction for {source_name}")
            if index not in evidence:
                raise ReleaseContractError(f"candidate lacks likelihood evidence for {source_name}")
            grid = validate_grid(predictions[test_index]); token = grid_key(grid); item = evidence[index]
            nlls = item.get("view_negative_log_likelihoods")
            if not isinstance(nlls, list) or not nlls:
                raise ReleaseContractError("missing B-support view likelihoods")
            per_grid[token].append({"source": source_name, "candidate_index": index, "original_log_likelihood": float(item["original_log_likelihood"]), "mean_view_nll": fmean(float(value) for value in nlls), "grid": grid, "slot_tags": selected_tags, "selected_support_count": len(selected_tags)})
        # A source can legitimately parse no grid on a task.  It remains in
        # the evidence record, but D1 must not invent an attempt; fail only if
        # *both* sources leave the output with no candidate at all.
        if not per_grid:
            continue
        # Existing source-local B-support: support multiplicity minus mean view NLL.
        source_ranked = sorted(per_grid, key=lambda token: (-max(row["selected_support_count"] - row["mean_view_nll"] for row in per_grid[token]), min(row["candidate_index"] for row in per_grid[token]), token))
        for rank, token in enumerate(source_ranked, 1):
            entry = grouped.setdefault(token, {"grid_key": token, "grid": per_grid[token][0]["grid"], "source_rows": []})
            entry["source_rows"].extend(per_grid[token])
            entry.setdefault("_source_ranks", {})[source_name] = rank
    for entry in grouped.values():
        entry["rrf_score"] = sum(1.0 / rank for rank in entry.pop("_source_ranks").values())
    return list(grouped.values())


def select_record(record: Mapping[str, Any], task_contract: Mapping[str, Any], test_inputs: list[list[list[int]]] | None = None) -> dict[str, Any]:
    sources = record.get("sources")
    if record.get("status") != "SUCCESS" or not isinstance(sources, Mapping):
        raise ReleaseContractError("worker did not produce successful dual-source evidence")
    outputs = []
    for output in task_contract["test_outputs"]:
        index = int(output["test_index"])
        pool = pool_for_output(sources, index)
        if not pool:
            if test_inputs is None or index >= len(test_inputs) or digest(validate_grid(test_inputs[index])) != output["input_sha256"]:
                raise ReleaseContractError("completed-empty fallback requires the bound runtime test input")
            fallback = validate_grid(test_inputs[index])
            outputs.append({"test_index": index, "attempt_1": fallback, "attempt_2": fallback, "ordered_grid_keys": [], "likelihood_ranks": {}, "likelihood_rrf": {}, "candidate_pool": [], "selection_source": "COMPLETED_EMPTY_INPUT_COPY", "fallback_policy": EMPTY_POOL_POLICY})
            continue
        ordered, likelihood_ranks, l_rrf = d1_order(pool)
        verify_d1_scope(pool, ordered)
        first, second = attempts_from_order(pool, ordered)
        if first is None or second is None:
            raise ReleaseContractError("empty candidate pool")
        outputs.append({"test_index": index, "attempt_1": first, "attempt_2": second, "ordered_grid_keys": ordered, "likelihood_ranks": likelihood_ranks, "likelihood_rrf": l_rrf, "candidate_pool": pool, "selection_source": "D1_MODEL"})
    return {"status": "SUCCESS", "method": "fixed_TTT24_TTT48_4plus4_per_output_D1", "outputs": outputs}


def quota_status(*, available: float | None = None, estimate: float | None = None, reserve: float | None = None) -> dict[str, Any]:
    if any(value is None for value in (available, estimate, reserve)):
        return {"status": "QUOTA_NOT_VERIFIED", "available_gpu_hours": available, "estimated_gpu_hours": estimate, "recovery_reserve_gpu_hours": reserve}
    required = float(estimate) + float(reserve)
    return {"status": "QUOTA_SUFFICIENT" if float(available) >= required else "QUOTA_INSUFFICIENT", "available_gpu_hours": float(available), "estimated_gpu_hours": float(estimate), "recovery_reserve_gpu_hours": float(reserve), "required_gpu_hours": required}
