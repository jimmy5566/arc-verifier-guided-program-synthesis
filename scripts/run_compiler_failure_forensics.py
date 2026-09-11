"""Create safe Phase-A compiler reachability and R2-forensics artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.compiler_failure_forensics import forensics_result, safe_report
from llm.compiler_reachability import reachability_inventory


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-checkpoints", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, default=ROOT / "experiments" / "results" / "COMPILER_REACHABILITY_INVENTORY_V1.json")
    parser.add_argument("--forensics-output", type=Path, default=ROOT / "experiments" / "results" / "COMPILER_FAILURE_FORENSICS_V1.json")
    parser.add_argument("--report-output", type=Path, default=ROOT / "reports" / "compiler_failure_forensics_v1.md")
    args = parser.parse_args()
    if args.inventory_output.exists() or args.forensics_output.exists() or args.report_output.exists():
        raise FileExistsError("forensics artifacts are one-shot and will not be overwritten")
    inventory = reachability_inventory(max_depth=3)
    result = forensics_result(args.r2_checkpoints)
    write_json(args.inventory_output, inventory)
    write_json(args.forensics_output, result)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(safe_report(result), encoding="utf-8")


if __name__ == "__main__":
    main()
