#!/usr/bin/env python3
"""Audit the exact historical d24 adapters required by frozen E1 Micro24."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from collections.abc import Callable
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audit_adapter_state(cohort: dict[str, Any], *, identity_reader: Callable[[Path], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Hash every unique task adapter and compare to its frozen manifest."""
    if identity_reader is None:
        from scripts.run_non_s_rolling_resident_v1 import _adapter_identity
        identity_reader = _adapter_identity

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cohort["outputs"]:
        by_task[str(row["task_id"])].append(row)
    records: list[dict[str, Any]] = []
    for task_id, rows in sorted(by_task.items()):
        exemplar = rows[0]
        expected = exemplar["adapter_identity"]
        actual = identity_reader(Path(exemplar["adapter_path"]))
        mismatches = {
            key: {"expected": expected.get(key), "actual": actual.get(key)}
            for key in ("adapter_sha256", "adapter_config_sha256", "adapter_config_semantics", "status")
            if expected.get(key) != actual.get(key)
        }
        records.append({
            "task_id": task_id, "output_ids": sorted(str(row["output_id"]) for row in rows),
            "adapter_path": exemplar["adapter_path"], "status": "PASS" if not mismatches else "FAIL",
            "mismatches": mismatches,
        })
    adapter_state = "EXACT_HISTORICAL" if records and all(row["status"] == "PASS" for row in records) else "MISSING_OR_MISMATCH"
    return {
        "experiment": "SEARCH_ORDER_MICRO24_V1", "adapter_state": adapter_state,
        "gold_loaded": False, "cohort_sha256": cohort["cohort_sha256"],
        "unique_tasks": len(records), "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    payload = audit_adapter_state(cohort)
    _atomic_json(args.output, payload)
    print(_canonical({"adapter_state": payload["adapter_state"], "output": str(args.output), "unique_tasks": payload["unique_tasks"]}))
    if payload["adapter_state"] != "EXACT_HISTORICAL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
