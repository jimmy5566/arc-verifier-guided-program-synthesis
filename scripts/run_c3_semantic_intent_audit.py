"""Run the no-inference semantic audit of frozen C3 checkpoint output."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from llm.c3_semantic_intent import audit_c3, safe_report


ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_ledger(path: Path, result: dict, code_commit: str) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames or ()), list(reader)
    if any(row.get("experiment_id") == result["experiment_id"] for row in rows):
        raise ValueError("duplicate semantic audit ledger entry")
    metrics = result["metrics"]
    row = {"experiment_id": result["experiment_id"], "date": datetime.now().date().isoformat(), "git_commit": code_commit, "solver": "C3 deterministic semantic intent audit", "representation": "frozen compiler-constrained C3 output", "search_method": "no new inference; deterministic benchmark-derived constraints", "llm_model": "qwen-lm/qwen-3/Transformers/8b/1", "candidate_budget": "0", "validation_split": "frozen API-only 60-case development benchmark", "tasks_solved": str(metrics["semantic_success"]), "accuracy": str(metrics["semantic_success_rate"]), "runtime_seconds": "0", "gpu_hours": "0", "notes": json.dumps({"status": result["decision"]["status"], "arc_oracle_ladder_allowed": result["decision"]["arc_oracle_ladder_allowed"], "arc_data_used": False}, sort_keys=True)}
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    parser.add_argument("--result", type=Path, default=ROOT / "experiments" / "results" / "C3_SEMANTIC_INTENT_AUDIT_V1.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "c3_semantic_intent_audit_v1.md")
    parser.add_argument("--ledger", type=Path, default=ROOT / "experiments" / "experiments.csv")
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if args.result.exists() or args.report.exists():
        raise FileExistsError("semantic audit public artifacts are one-shot")
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    result = audit_c3(benchmark=benchmark, checkpoint_root=args.checkpoints)
    write(args.result, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(safe_report(result), encoding="utf-8")
    append_ledger(args.ledger, result, args.code_commit)


if __name__ == "__main__":
    main()
