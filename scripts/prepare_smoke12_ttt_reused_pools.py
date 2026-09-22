"""Freeze the CPU-only S0 and S3 candidate pools for Smoke12.

This program deliberately receives no solution path.  It subsets the already
frozen Base Aug8 and strong-TTT candidate artifacts to the frozen Smoke12
cohort, then writes:

* S0: the existing strong-TTT 24-step greedy pool; and
* S3: the deterministic Base-then-TTT prediction union.

S3 is an Any-of-K diagnostic only.  It does not rank, select, or otherwise
create a new prediction policy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_eval3_reference_ttt import _read, _task_hash
from scripts.run_smoke12_ttt_cost_ablation import _frozen_task_ids


S0_STATUS = "SMOKE12_S0_REUSED_STRONG_TTT24_GREEDY_CANDIDATES_FROZEN"
S3_STATUS = "SMOKE12_S3_BASE_TTT_UNION_CANDIDATES_FROZEN"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_key(candidate: dict[str, Any]) -> str:
    prediction = candidate.get("prediction")
    if not isinstance(prediction, list):
        raise ValueError("candidate has no list prediction")
    return json.dumps(prediction, sort_keys=False, separators=(",", ":"))


def _records_for(value: dict[str, Any], task_ids: list[str], label: str) -> dict[str, dict[str, Any]]:
    records = value.get("records")
    if not isinstance(records, dict) or not set(task_ids).issubset(records):
        raise ValueError(f"{label}: artifact does not cover the frozen Smoke12 cohort")
    selected: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        record = records[task_id]
        if not isinstance(record, dict) or record.get("task_id") != task_id or not isinstance(record.get("candidates"), list):
            raise ValueError(f"{label}: invalid candidate record for {task_id}")
        selected[task_id] = record
    return selected


def _s0_record(record: dict[str, Any]) -> dict[str, Any]:
    candidates = copy.deepcopy(record["candidates"])
    return {
        "task_id": record["task_id"],
        "status": "SUCCESS" if candidates else "NO_VALID_NATIVE_CANDIDATE",
        "candidates": candidates,
        "source": "strong_ttt_24_greedy_reused",
        "source_status": record.get("status"),
    }


def _union_candidates(base: list[dict[str, Any]], ttt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate exact output grids in frozen Base-then-TTT source order."""
    unique: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for source, candidates in (("base_aug8", base), ("strong_ttt_24_greedy", ttt)):
        for candidate in candidates:
            key = _prediction_key(candidate)
            if key in by_key:
                by_key[key]["union_sources"].append(source)
                continue
            item = copy.deepcopy(candidate)
            item["union_sources"] = [source]
            by_key[key] = item
            unique.append(item)
    return unique


def _s3_record(base: dict[str, Any], ttt: dict[str, Any]) -> dict[str, Any]:
    candidates = _union_candidates(base["candidates"], ttt["candidates"])
    return {
        "task_id": base["task_id"],
        "status": "SUCCESS" if candidates else "NO_VALID_NATIVE_CANDIDATE",
        "candidates": candidates,
        "source": "base_aug8_then_strong_ttt_24_greedy_exact_prediction_union",
        "base_source_status": base.get("status"),
        "ttt_source_status": ttt.get("status"),
    }


def _artifact(*, status: str, task_ids: list[str], task_ids_hash: str, records: dict[str, dict[str, Any]], base_path: Path, ttt_path: Path) -> dict[str, Any]:
    return {
        "experiment_id": "ARC2_SMOKE12_TTT_COST_ACCURACY_ABLATION",
        "status": status,
        "protocol": "CPU-only subset/union of pre-existing frozen candidate pools; no model, GPU, ranking, selector, or solution access.",
        "solutions_opened": False,
        "task_ids": task_ids,
        "task_ids_hash": task_ids_hash,
        "base_candidates_sha256": _sha256(base_path),
        "strong_ttt_candidates_sha256": _sha256(ttt_path),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "base", "ttt", "s0_output", "s3_output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.s0_output.exists() or args.s3_output.exists():
        raise FileExistsError("refusing to overwrite frozen Smoke12 S0/S3 candidate pools")

    manifest, base, ttt = _read(args.manifest), _read(args.base), _read(args.ttt)
    try:
        task_ids, task_ids_hash = _frozen_task_ids(manifest)
    except ValueError as exc:
        raise ValueError("invalid frozen Smoke12 cohort manifest") from exc
    if manifest.get("status") != "SMOKE12_TTT_COST_ABLATION_COHORT_FROZEN":
        raise ValueError("invalid frozen Smoke12 cohort manifest")
    base_records = _records_for(base, task_ids, "base")
    ttt_records = _records_for(ttt, task_ids, "strong TTT")
    s0 = {task_id: _s0_record(ttt_records[task_id]) for task_id in task_ids}
    s3 = {task_id: _s3_record(base_records[task_id], ttt_records[task_id]) for task_id in task_ids}
    atomic_write_json(args.s0_output, _artifact(status=S0_STATUS, task_ids=task_ids, task_ids_hash=task_ids_hash, records=s0, base_path=args.base, ttt_path=args.ttt))
    atomic_write_json(args.s3_output, _artifact(status=S3_STATUS, task_ids=task_ids, task_ids_hash=task_ids_hash, records=s3, base_path=args.base, ttt_path=args.ttt))
    print(json.dumps({
        "event": "SMOKE12_REUSED_CANDIDATE_POOLS_FROZEN",
        "task_count": len(task_ids),
        "s0_unique_candidate_count": sum(len(item["candidates"]) for item in s0.values()),
        "s3_unique_candidate_count": sum(len(item["candidates"]) for item in s3.values()),
        "solutions_opened": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
