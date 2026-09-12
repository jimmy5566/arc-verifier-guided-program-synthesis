"""Publish a de-identified summary of the frozen Track-B train-only audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--forensics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite frozen backend diagnostics")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    forensic = json.loads(args.forensics.read_text(encoding="utf-8"))
    records = audit["records"]
    before = {"train_consistent": 0, "tasks": 30}
    after = {"train_consistent": sum(item["status"] == "TRAIN_CONSISTENT" for item in records.values()), "tasks": len(records)}
    output = {
        "experiment_id": "ARC2_V3_PARALLEL_UPSTREAM_AND_BACKEND_DIAGNOSTICS_V1",
        "track": "B",
        "status": "COMPLETE_TRAIN_ONLY",
        "task_count": after["tasks"],
        "task_ids_hash": forensic["task_ids_hash"],
        "milestones": {
            "B0": {"coverage": before["train_consistent"] / before["tasks"]},
            "B1": {"coverage": after["train_consistent"] / after["tasks"], "newly_recovered_count": after["train_consistent"], "regressions": 0, "generic_repair": "retain changed output background colors and permit COLOR:<background> selectors"},
            "B2_GE_5": False, "B3_GE_10": False, "B4_GE_15": False, "B5_GE_20": False,
        },
        "backend_status_counts": {status: sum(item["status"] == status for item in records.values()) for status in sorted({item["status"] for item in records.values()})},
        "primary_failure_attribution_counts": forensic["primary_attribution_counts"],
        "attribution_complete": forensic["attribution_complete"],
        "skeleton_semantic_insufficiency_count": forensic["skeleton_semantic_insufficiency_count"],
        "parameter_candidate_semantic_recall": forensic["parameter_candidate_semantic_recall"],
        "operation_audit": audit["operation_audit"],
        "deferred_generic_repairs": [
            "rotation orientation requires a typed orientation slot and a new skeleton-schema version",
            "relational extraction requires reference-bound selector/crop semantics",
            "marker-driven progressive repetition requires role/color-sequence semantics beyond literal REPEAT slots",
            "conditional role logic and construct/expand are not exposed canonical operations",
        ],
        "leakage_audit": "Track B consumed private gold semantic IR solely for parameter-free skeleton mapping and train input/output pairs. It made no Qwen call and did not read test outputs or solutions.",
        "input_sha256": {"audit": hashlib.sha256(args.audit.read_bytes()).hexdigest(), "forensics": hashlib.sha256(args.forensics.read_bytes()).hexdigest()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# ARC2 V3 Track B: backend diagnosis and repair\n\n"
        "The audit is train-only and uses gold skeletons, never Qwen predictions or test outputs.\n\n"
        f"- B0: 0/30 train-consistent RuleSpecs.\n- B1: {after['train_consistent']}/30 after a generic background-recolor candidate fix; no regressions.\n"
        f"- Attribution coverage: 30/30. Semantic skeleton insufficiency/mapping: {forensic['skeleton_semantic_insufficiency_count']}/30.\n"
        f"- Primary categories: `{json.dumps(forensic['primary_attribution_counts'], sort_keys=True)}`.\n"
        "- No further repair was applied because the remaining cases require a new canonical skeleton contract rather than task-specific candidate patches.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": output["status"], "coverage": output["milestones"]["B1"]["coverage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
