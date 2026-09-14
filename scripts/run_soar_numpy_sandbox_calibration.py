"""Calibrate the isolated SOAR program sandbox without loading any model."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from inference.dual_reasoning_smoke import execute_program, verify_program


SYNTHETICS = {
    "A_python_identity": "def transform(grid):\n    return grid\n",
    "B_numpy_array": "import numpy as np\n\ndef transform(grid):\n    return np.array(grid)\n",
    "C_numpy_flip": "import numpy as np\n\ndef transform(grid):\n    x = np.array(grid)\n    return np.flip(x, axis=1)\n",
}
PROBE_TIMEOUT_SECONDS = 30.0  # Measurement ceiling only; never used as the selected runtime timeout.


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def completed(row: dict[str, Any]) -> bool:
    return bool(row.get("process_started") and row.get("executable") and row.get("status") != "TIMEOUT")


def candidate_summary(program: str | None, pairs: list[tuple[Any, Any]], timeout_seconds: float) -> dict[str, Any]:
    if not program:
        return {"process_started": False, "runtime_completed": False, "output_valid": False, "all_train_exact": False, "failure_reason": "no_transform_code_extracted"}
    verification = verify_program(program, pairs, timeout_seconds=timeout_seconds)
    rows = verification["train_execution"]
    return {
        "process_started": bool(rows) and all(bool(row.get("process_started")) for row in rows),
        "runtime_completed": bool(rows) and all(completed(row) for row in rows),
        "output_valid": bool(rows) and all(bool(row.get("output_valid")) for row in rows),
        "all_train_exact": verification["all_train_exact"],
        "per_train_pair_exact": [bool(row.get("ok") and row.get("grid") == target) for row, (_source, target) in zip(rows, pairs, strict=True)],
        "failure_reason": next((row.get("error") or row.get("reason") for row in rows if not row.get("ok")), verification.get("reason")),
        "verification": verification,
        "timeout_seconds": timeout_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--frozen-induction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grid = [[1, 2], [3, 4]]
    synthetic = {name: execute_program(program, grid, timeout_seconds=PROBE_TIMEOUT_SECONDS) for name, program in SYNTHETICS.items()}
    if not all(row.get("ok") for row in synthetic.values()):
        write_json(args.output, {"status": "SYNTHETIC_FAILED", "synthetic": synthetic})
        raise RuntimeError("synthetic sandbox transport did not complete")

    # The selected value is a deterministic twofold margin over the slowest
    # observed NumPy cold-start wall time, not an unrelated fixed timeout.
    numpy_cold_wall = max(float(synthetic[name]["total_wall_seconds"]) for name in ("B_numpy_array", "C_numpy_flip"))
    selected_timeout = math.ceil(numpy_cold_wall * 2.0 * 1000.0) / 1000.0
    frozen = json.loads(args.frozen_induction.read_text(encoding="utf-8"))
    tasks = load_dataset(args.challenge_path)
    candidates: dict[str, list[dict[str, Any]]] = {}
    for task_id, record in frozen["records"].items():
        pairs = [(example.input.to_list(), example.output.to_list()) for example in tasks[task_id].train]
        candidates[task_id] = [
            {"candidate_index": item["candidate_index"], **candidate_summary(item.get("extracted_code"), pairs, selected_timeout)}
            for item in record["candidate_programs"]
        ]
        print(json.dumps({"event": "CALIBRATED_CANDIDATES_EXECUTED", "task_id": task_id, "count": len(candidates[task_id])}), flush=True)
    flat = [item for group in candidates.values() for item in group]
    output = {
        "status": "CALIBRATION_AND_FROZEN_CANDIDATE_EXECUTION_COMPLETE",
        "synthetic": synthetic,
        "timeout_policy": {"probe_ceiling_seconds": PROBE_TIMEOUT_SECONDS, "numpy_cold_wall_seconds": numpy_cold_wall, "selected_timeout_seconds": selected_timeout, "rule": "ceil(2 * max(B_numpy_array, C_numpy_flip) wall time to milliseconds)"},
        "candidate_funnel": {
            "generated": len(flat), "process_started": sum(item["process_started"] for item in flat),
            "runtime_completed": sum(item["runtime_completed"] for item in flat), "output_valid": sum(item["output_valid"] for item in flat),
            "all_train_exact": sum(item["all_train_exact"] for item in flat),
        },
        "candidates": candidates,
    }
    write_json(args.output, output)
    print(json.dumps({"event": "CALIBRATION_COMPLETE", "funnel": output["candidate_funnel"], "selected_timeout_seconds": selected_timeout}), flush=True)


if __name__ == "__main__":
    main()
