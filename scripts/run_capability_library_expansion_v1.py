"""Run the frozen development-only representability audit for Track D."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from capabilities.expansion_audit_v1 import audit_task
from capabilities.expansion_v1 import CAPABILITIES, validate_capabilities


PRIORITY = (
    "CAP_DIAGONAL_SEQUENCE_TRAIL_V1",
    "CAP_REPEAT_COPY_UNTIL_BOUNDARY_V1",
    "CAP_TRANSFER_EXTERNAL_MOTIF_TO_FRAME_V1",
    "CAP_REPEAT_COPY_TRANSLATION_V1",
    "CAP_TILE_MASK_AT_ANCHORS_V1",
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    errors = validate_capabilities()
    if errors:
        raise RuntimeError(f"capability registry invalid: {errors}")
    task_ids = json.loads(args.cohort.read_text(encoding="utf-8"))["task_ids"]
    if len(task_ids) != 30 or len(set(task_ids)) != 30:
        raise ValueError("requires the exact frozen 30-task development cohort")
    tasks, solutions = load_dataset(args.challenge), json.loads(args.solutions.read_text(encoding="utf-8"))
    private_records = []
    covered_by_capability: Counter[str] = Counter()
    family_coverage: Counter[str] = Counter()
    # The prior classification is private and used only to aggregate family
    # coverage; candidate construction never reads it.
    forensic = {item["task_id"]: item["classification"] for item in json.loads((ROOT / "artifacts/downstream_capability_gap_forensics_v1_private.json").read_text(encoding="utf-8"))["records"]}
    for task_id in task_ids:
        train_exact, test_exact = audit_task(tasks[task_id], solutions[task_id])
        for candidate in test_exact:
            covered_by_capability[candidate.capability_id] += 1
        if test_exact:
            family_coverage[forensic[task_id]] += 1
        private_records.append({"task_id": task_id, "train_exact_capabilities": sorted({item.capability_id for item in train_exact}), "test_exact_capabilities": sorted({item.capability_id for item in test_exact}), "classification": forensic[task_id]})
    assigned: dict[str, int] = Counter()
    for record in private_records:
        for capability_id in PRIORITY:
            if capability_id in record["test_exact_capabilities"]:
                assigned[capability_id] += 1
                break
    after = sum(bool(record["test_exact_capabilities"]) for record in private_records)
    threshold = "CAPABILITY_EXPANSION_USEFUL" if after >= 15 else "CAPABILITY_STILL_BLOCKED"
    result = {
        "experiment_id": "CAPABILITY_LIBRARY_EXPANSION_V1",
        "status": "COMPLETE",
        "task_count": 30,
        "before_representable": 0,
        "after_representable": after,
        "after_rate": after / 30,
        "threshold_status": threshold,
        "added_capabilities": [
            {"capability_id": item.capability_id, "input_type": item.input_type, "output_type": item.output_type, "parameters": list(item.parameters), "semantics": item.semantics, "preconditions": item.preconditions, "test_exact_coverage": covered_by_capability[item.capability_id], "marginal_coverage_gain": assigned[item.capability_id]}
            for item in (CAPABILITIES[key] for key in sorted(CAPABILITIES))
        ],
        "family_coverage": {
            "iteration_repeat": {"covered": family_coverage["MISSING_ITERATION_OR_REPEAT"], "total": 15},
            "relational": {"covered": family_coverage["MISSING_RELATIONAL_CAPABILITY"], "total": 8},
            "conditional": {"covered": family_coverage["MISSING_CONDITIONAL_OR_ROLE_LOGIC"], "total": 4},
            "composition": {"covered": family_coverage["MISSING_COMPOSITION"], "total": 2},
            "output_construction": {"covered": family_coverage["MISSING_OUTPUT_CONSTRUCTION"], "total": 1},
        },
        "audit_protocol": "generic train-derived candidates -> train exact -> development test exact oracle check; no recognition imports, outputs, or scores",
        "runtime_seconds": perf_counter() - started,
        "leakage_audit": "task IDs, candidate programs, grids, and solutions stay in ignored private artifacts; public aggregate contains no per-task data.",
    }
    private_path = ROOT / "artifacts/capability_library_expansion_v1_private.json"
    _write(private_path, {"records": private_records})
    output_path = ROOT / "experiments/results/CAPABILITY_LIBRARY_EXPANSION_V1.json"
    _write(output_path, result)
    report = ["# Capability Library Expansion V1", "", "这是与 Recognition 隔离的 development-only solution-aware representability audit。候选仅从 train pairs 归纳，并且必须同时 train exact 与 test exact 才计入。", "", f"- Before: **0/30**", f"- After: **{after}/30 ({after / 30:.1%})**", f"- Status: **{threshold}**", "", "## Family coverage", "", "| Family | Covered | Total |", "| --- | ---: | ---: |", *[f"| {name} | {item['covered']} | {item['total']} |" for name, item in result["family_coverage"].items()], "", "## Added generic capabilities", "", "| Capability | Test-exact coverage | Marginal gain |", "| --- | ---: | ---: |", *[f"| {item['capability_id']} | {item['test_exact_coverage']} | {item['marginal_coverage_gain']} |" for item in result["added_capabilities"]], "", "所有 per-task witness、grid 与 test target 仅保存在 ignored 私有 artifact。"]
    (ROOT / "reports/capability_library_expansion_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with (ROOT / "experiments/experiments.csv").open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"])
        writer.writerow({"experiment_id": "CAPABILITY_LIBRARY_EXPANSION_V1", "date": str(date.today()), "git_commit": "pending", "solver": "generic high-level capability oracle audit", "representation": "typed capability expansion V1", "search_method": "bounded train-derived generic candidates", "llm_model": "", "candidate_budget": 0, "validation_split": "development_only_30", "tasks_solved": after, "accuracy": after / 30, "runtime_seconds": result["runtime_seconds"], "gpu_hours": 0, "notes": json.dumps({"threshold": threshold}, sort_keys=True)})
    print(json.dumps({"after": after, "status": threshold}, sort_keys=True))


if __name__ == "__main__":
    main()
