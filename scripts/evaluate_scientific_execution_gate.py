#!/usr/bin/env python3
"""Evaluate the deterministic, fail-closed ARC2 scientific execution gate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

REQUIRED = (
    "frozen_baseline_identity", "no_eval60_gold_in_training_path",
    "no_diagnostic_gold_in_training_labels", "surface_isolation",
    "baseline_collector_target_blind", "model_runtime_identity",
    "training_budget_available", "final_audit_unopened", "provenance_clean",
)
def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"); temp.replace(path)
def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--input", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); value = json.loads(a.input.read_text(encoding="utf-8"))
    conditions = value.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(REQUIRED):
        raise RuntimeError("GATE_CONDITIONS_INVALID")
    passed = all(conditions[name].get("status") == "PASS" for name in REQUIRED)
    result = {**value, "schema_version": 1, "required_conditions": list(REQUIRED),
              "AUTO_SCIENTIFIC_EXECUTION_AUTHORIZED": passed,
              "baseline_reference_collection_authorized": passed,
              "scientific_training_authorized": passed,
              "status": "PASS" if passed else "FAIL_CLOSED"}
    atomic(a.output, result); print(json.dumps({"status": result["status"], "authorized": passed}, sort_keys=True))
    return 0 if passed else 1
if __name__ == "__main__": raise SystemExit(main())
