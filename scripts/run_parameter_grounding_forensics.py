"""Produce the deterministic no-inference S2 parameter forensics artifact."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.macro_api_benchmark import benchmark_definition
from llm.parameter_grounding_forensics import analyze_s2_parameter_failures, safe_report


def update_ledger(path: Path, result: dict, git_commit: str) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle); fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row["experiment_id"] != result["experiment_id"]]
    rows.append({
        "experiment_id": result["experiment_id"], "date": str(date.today()), "git_commit": git_commit,
        "solver": "Deterministic S2 parameter failure forensics", "representation": "frozen Family-First checkpoint + Macro contract",
        "search_method": "no inference deterministic audit", "llm_model": "qwen-lm/qwen-3/Transformers/8b/1",
        "candidate_budget": 0, "validation_split": "frozen_macro_api_60_no_arc", "tasks_solved": 0, "accuracy": 0.0,
        "runtime_seconds": 0.0, "gpu_hours": 0.0,
        "notes": json.dumps({"wrong_parameter": result["s2_wrong_parameter_cases"], "pure_parameter": result["pure_parameter_failure_count"], "structural_confounds": result["non_parameter_structural_failure_count"], "arc_data_used": False}, sort_keys=True),
    })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "results" / "PARAMETER_GROUNDING_FORENSICS_V1.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "parameter_grounding_forensics_v1.md")
    parser.add_argument("--ledger", type=Path, default=ROOT / "experiments" / "experiments.csv")
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("forensics result/report already exists")
    result = analyze_s2_parameter_failures(benchmark=benchmark_definition(), checkpoint_root=args.checkpoint_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(safe_report(result), encoding="utf-8")
    update_ledger(args.ledger, result, args.git_commit)
    print(json.dumps({"status": result["status"], "s2_wrong_parameter_cases": result["s2_wrong_parameter_cases"], "pure_parameter_failure_count": result["pure_parameter_failure_count"]}))


if __name__ == "__main__":
    main()
