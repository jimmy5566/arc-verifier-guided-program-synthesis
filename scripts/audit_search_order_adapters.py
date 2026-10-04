#!/usr/bin/env python3
"""Audit the exact historical d24 adapters required by frozen E1 Micro24."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_identity(adapter: Path) -> dict[str, Any]:
    model_file = adapter / "adapter_model.safetensors"
    config_file = adapter / "adapter_config.json"
    if not model_file.is_file() or not config_file.is_file():
        raise FileNotFoundError(f"missing depth_024 adapter files: {adapter}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    targets = config.get("target_modules")
    if isinstance(targets, str):
        try:
            targets = ast.literal_eval(targets)
        except (SyntaxError, ValueError):
            targets = ()
    semantics = {
        "peft_type": config.get("peft_type"), "task_type": config.get("task_type"), "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"), "target_modules": sorted(map(str, targets)) if isinstance(targets, (list, tuple, set)) else [],
    }
    expected_targets = ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]
    valid = semantics == {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 256, "lora_alpha": 32, "target_modules": expected_targets}
    return {"adapter_path": str(adapter), "adapter_sha256": _sha_file(model_file),
            "adapter_config_sha256": _sha_file(config_file), "adapter_config_semantics": semantics,
            "status": "PASS" if valid else "FAIL"}


def audit_adapter_state(cohort: dict[str, Any]) -> dict[str, Any]:
    """Hash every unique task adapter and compare to its frozen manifest."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cohort["outputs"]:
        by_task[str(row["task_id"])].append(row)
    records: list[dict[str, Any]] = []
    for task_id, rows in sorted(by_task.items()):
        exemplar = rows[0]
        expected = exemplar["adapter_identity"]
        actual = _adapter_identity(Path(exemplar["adapter_path"]))
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
