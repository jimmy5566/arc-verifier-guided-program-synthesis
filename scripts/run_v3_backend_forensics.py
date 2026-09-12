"""Classify every frozen-30 V3 backend outcome without Qwen or test outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3.diagnostics.backend_forensics import primary_failure_attribution, semantic_sufficiency
from v3.diagnostics.backend_forensics import recolor_candidate_semantic_recall
from arc.io import load_dataset
from v3.evidence.extractor import extract_task_evidence
from v3.schema.rule_skeleton import OperationId, RuleSkeleton


def _skeleton(value: dict[str, object] | None) -> RuleSkeleton | None:
    if value is None:
        return None
    return RuleSkeleton.from_operations(str(value["family"]), tuple(OperationId(step["operation"]) for step in value["steps"]))  # type: ignore[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--oracle-semantic", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen forensic artifact")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle_semantic.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    ids = tuple(cohort["task_ids"])
    task_hash = hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
    if len(ids) != 30 or task_hash != oracle["task_ids_hash"]:
        raise ValueError("requires frozen development-30")
    tasks = load_dataset(args.challenge_path)
    records: dict[str, dict[str, object]] = {}
    for task_id in ids:
        audited = audit["records"][task_id]
        skeleton = _skeleton(audited.get("gold_skeleton"))
        attribution = primary_failure_attribution(
            oracle["gold"][task_id], skeleton, audited["status"], audited.get("unsupported_reason")
        )
        sufficient, sufficiency_reason = semantic_sufficiency(oracle["gold"][task_id], skeleton)
        candidate_recall = (
            {"assessable": False, "reason": "NO_GOLD_SKELETON"}
            if skeleton is None else recolor_candidate_semantic_recall(skeleton, extract_task_evidence(tasks[task_id]))
        )
        records[task_id] = {
            "backend_status": audited["status"],
            "primary_attribution": attribution,
            "gold_skeleton_semantically_sufficient": sufficient,
            "sufficiency_reason": sufficiency_reason,
            "candidate_semantic_recall": candidate_recall,
        }
    counts = Counter(item["primary_attribution"] for item in records.values())
    assessable = [item["candidate_semantic_recall"] for item in records.values() if item["candidate_semantic_recall"]["assessable"]]
    output = {
        "experiment_id": "ARC2_V3_PARALLEL_UPSTREAM_AND_BACKEND_DIAGNOSTICS_V1",
        "track": "B",
        "status": "TRAIN_ONLY_FORENSICS_COMPLETE",
        "task_ids_hash": task_hash,
        "protocol": "private gold semantic IR + train-only backend audit; no Qwen predictions, test outputs, or solution file read",
        "records": records,
        "primary_attribution_counts": dict(sorted(counts.items())),
        "attribution_complete": len(records) == 30 and all(item["primary_attribution"] for item in records.values()),
        "skeleton_semantic_insufficiency_count": sum(not item["gold_skeleton_semantically_sufficient"] for item in records.values()),
        "parameter_candidate_semantic_recall": {
            "assessable_tasks": len(assessable),
            "mean_joint_recall": sum(float(item["semantic_recall"]) for item in assessable) / len(assessable) if assessable else None,
            "selector_recall": sum(float(item["selector_recall"]) for item in assessable) / len(assessable) if assessable else None,
            "target_color_recall": sum(float(item["target_color_recall"]) for item in assessable) / len(assessable) if assessable else None,
            "scope": "uniform SELECT->RECOLOR train-derived witnesses only; relational/repeat slots are explicitly not inferred from semantic oracle labels",
        },
        "input_audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(records), "attributions": output["primary_attribution_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
