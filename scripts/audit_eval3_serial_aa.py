"""Target-blind comparison for two frozen Eval3 serial A/A candidate runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from inference.kaggle_l4_parallel_runner import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _grid_key(candidate: dict[str, Any]) -> str:
    return _canonical(candidate.get("prediction"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _trace_by_step(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["step"]): item for item in record["ttt_trace"]["ttt_trace"]}


def _task_report(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    trace_a, trace_b = _trace_by_step(left), _trace_by_step(right)
    common_steps = sorted(set(trace_a) & set(trace_b))
    first_divergence: dict[str, Any] | None = None
    trace_comparison: list[dict[str, Any]] = []
    for step in common_steps:
        a, b = trace_a[step], trace_b[step]
        schedule_same = all(
            a.get(key) == b.get(key)
            for key in ("selected_training_sequence_index", "selected_training_sequence_hash", "selected_labels_hash", "lr")
        )
        adapter_same = a.get("adapter_fingerprint") == b.get("adapter_fingerprint")
        loss_same = a.get("loss") == b.get("loss")
        row = {
            "step": step,
            "schedule_same": schedule_same,
            "adapter_same": adapter_same,
            "loss_a": a.get("loss"),
            "loss_b": b.get("loss"),
            "loss_abs_delta": None if a.get("loss") is None or b.get("loss") is None else abs(float(a["loss"]) - float(b["loss"])),
        }
        trace_comparison.append(row)
        if first_divergence is None and (not schedule_same or not adapter_same or not loss_same):
            first_divergence = row

    candidates_a = [_grid_key(item) for item in left["candidates"]]
    candidates_b = [_grid_key(item) for item in right["candidates"]]
    pools_a, pools_b = set(candidates_a), set(candidates_b)
    views_a = [item.get("grid_sha256") for item in left.get("raw_views", ())]
    views_b = [item.get("grid_sha256") for item in right.get("raw_views", ())]
    return {
        "task_id": left["task_id"],
        "input_identity_same": left["input_identity"] == right["input_identity"],
        "initial_adapter_same": trace_a[0].get("adapter_fingerprint") == trace_b[0].get("adapter_fingerprint"),
        "initial_adapter_a": trace_a[0].get("adapter_fingerprint"),
        "initial_adapter_b": trace_b[0].get("adapter_fingerprint"),
        "variant_order_same": left["ttt_trace"].get("variant_order_hash") == right["ttt_trace"].get("variant_order_hash"),
        "token_ids_same": left["ttt_trace"].get("token_ids_hash") == right["ttt_trace"].get("token_ids_hash"),
        "labels_same": left["ttt_trace"].get("labels_hash") == right["ttt_trace"].get("labels_hash"),
        "first_divergence": first_divergence,
        "trace": trace_comparison,
        "candidate_order_same": candidates_a == candidates_b,
        "candidate_pool_jaccard": _jaccard(pools_a, pools_b),
        "candidate_pool_intersection": len(pools_a & pools_b),
        "candidate_pool_union": len(pools_a | pools_b),
        "raw_view_grid_hashes_same": views_a == views_b,
        "raw_view_hashes_a": views_a,
        "raw_view_hashes_b": views_b,
        "invalid_a": left.get("invalid_candidate_count"),
        "invalid_b": right.get("invalid_candidate_count"),
        "telemetry_a": left.get("telemetry"),
        "telemetry_b": right.get("telemetry"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-a", type=Path, required=True)
    parser.add_argument("--serial-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen A/A comparison")
    a, b = _read(args.serial_a), _read(args.serial_b)
    if a.get("solutions_opened") or b.get("solutions_opened"):
        raise ValueError("A/A input must remain target-blind")
    ids_a, ids_b = set(a["records"]), set(b["records"])
    if ids_a != ids_b:
        raise ValueError("serial A/B task IDs differ")
    tasks = [_task_report(a["records"][task_id], b["records"][task_id]) for task_id in sorted(ids_a)]
    initial_mismatch = any(not item["initial_adapter_same"] for item in tasks)
    later_drift = any(item["first_divergence"] is not None for item in tasks)
    pool_drift = any(item["candidate_pool_jaccard"] < 1.0 for item in tasks)
    if initial_mismatch:
        status = "SERIAL_NONDETERMINISTIC_INITIAL_ADAPTER"
    elif later_drift or pool_drift:
        status = "SERIAL_NONDETERMINISTIC"
    else:
        status = "SERIAL_STABLE"
    report = {
        "event": "EVAL3_SERIAL_AA_TARGET_BLIND_COMPARISON_FROZEN",
        "solutions_opened": False,
        "serial_a_sha256": _sha256(a),
        "serial_b_sha256": _sha256(b),
        "status": status,
        "task_count": len(tasks),
        "tasks": tasks,
    }
    atomic_write_json(args.output, report)
    print(json.dumps({"event": report["event"], "status": status, "task_count": len(tasks), "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
