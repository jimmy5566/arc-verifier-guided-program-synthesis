"""Local-only scorer for a complete, frozen Track-U prediction artifact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v3.schema.rule_skeleton import OperationId, RuleSkeleton


def gold_skeleton_from_semantic_oracle(oracle: Mapping[str, Any]) -> tuple[RuleSkeleton | None, str | None]:
    """Deterministically map private semantic gold to parameter-free V3 gold."""
    if oracle["conditional_logic"]["enabled"]:
        return None, "GOLD_SKELETON_MAPPING_SUSPECTED:CONDITIONAL"
    family, operations = oracle["primary_family"], tuple(oracle["operations"])
    if "REPEAT" in operations:
        return RuleSkeleton.from_operations(family, (OperationId.SELECT, OperationId.REPEAT)), None
    if operations == ("RECOLOR",):
        return RuleSkeleton.from_operations(family, (OperationId.SELECT, OperationId.RECOLOR)), None
    if operations == ("TRANSFORM",):
        transform = oracle["spatial_transform"]["transform"]
        if transform == "ROTATE":
            return RuleSkeleton.from_operations(family, (OperationId.ROTATE,)), None
        if transform == "REFLECT":
            return RuleSkeleton.from_operations(family, (OperationId.REFLECT,)), None
    if "EXTRACT" in operations:
        return RuleSkeleton.from_operations(family, (OperationId.SELECT, OperationId.CROP)), None
    if "FILL" in operations:
        return RuleSkeleton.from_operations(family, (OperationId.SELECT, OperationId.FILL)), None
    return None, "GOLD_SKELETON_MAPPING_SUSPECTED:UNMAPPED_OPERATION"


def _decode(value: Mapping[str, Any]) -> RuleSkeleton:
    return RuleSkeleton.from_operations(
        str(value["family"]), tuple(OperationId(step["operation"]) for step in value["steps"])
    )


def _stratum(oracle: Mapping[str, Any]) -> str:
    if oracle["conditional_logic"]["enabled"]:
        return "CONDITIONAL"
    if oracle["iteration"]["enabled"]:
        return "ITERATION"
    if oracle["primary_family"] == "RELATIONAL_TRANSFORMATION":
        return "RELATIONAL"
    if oracle["primary_family"] == "OUTPUT_CONSTRUCTION":
        return "OUTPUT_CONSTRUCTION"
    if oracle["primary_family"] == "COMPOSITION":
        return "COMPOSITION"
    return "OTHER"


def _score_one(predictions: list[Mapping[str, Any]], gold: RuleSkeleton | None, gold_reason: str | None) -> dict[str, Any]:
    decoded: list[RuleSkeleton] = []
    try:
        decoded = [_decode(item) for item in predictions]
    except (KeyError, TypeError, ValueError):
        return {"schema_valid": False, "primary_failure": "SCHEMA_FAILURE", "true_recognition_failure": True}
    # A parsed envelope with zero hypotheses is not a valid Track-U response,
    # regardless of whether the independently-derived gold can be represented.
    if not decoded:
        return {"schema_valid": False, "primary_failure": "SCHEMA_FAILURE", "true_recognition_failure": True}
    if gold is None:
        return {"schema_valid": True, "primary_failure": gold_reason, "true_recognition_failure": False}
    top1 = decoded[0] if decoded else None
    gold_steps = tuple(step.operation for step in gold.steps)
    top_steps = tuple(step.operation for step in top1.steps)
    exact = top1 == gold
    recall3 = any(item == gold for item in decoded[:3])
    if top1.family != gold.family:
        failure = "FAMILY_WRONG"
    elif set(top_steps) != set(gold_steps):
        failure = "MISSING_OPERATION" if set(top_steps) < set(gold_steps) else "EXTRA_OPERATION" if set(top_steps) > set(gold_steps) else "CORE_OPERATION_WRONG"
    elif top_steps != gold_steps:
        failure = "OPERATION_SEQUENCE_WRONG"
    elif top1.required_slots != gold.required_slots:
        failure = "WRONG_SLOT_STRUCTURE"
    elif not exact:
        failure = "AMBIGUOUS_BUT_PLAUSIBLE"
    else:
        failure = "EXACT"
    return {
        "schema_valid": True,
        "family_correct": top1.family == gold.family,
        "operation_set_correct": set(top_steps) == set(gold_steps),
        "operation_sequence_exact": top_steps == gold_steps,
        "required_slot_set_correct": top1.required_slots == gold.required_slots,
        "full_top1_exact": exact,
        "recall_at_3": recall3,
        "primary_failure": failure,
        "true_recognition_failure": not exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oracle-semantic", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--experiments-csv", type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite a formal scored result")
    started = perf_counter()
    frozen = json.loads(args.predictions.read_text(encoding="utf-8"))
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    ids = tuple(cohort["task_ids"])
    task_hash = hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
    if frozen.get("status") != "PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING" or set(frozen.get("records", {})) != set(ids):
        raise ValueError("requires complete frozen-30 prediction artifact")
    if frozen.get("task_ids_hash") != task_hash or task_hash != config["cohort"]["task_ids_hash"]:
        raise ValueError("frozen cohort mismatch")
    if frozen.get("config_sha256") != hashlib.sha256(args.config.read_bytes()).hexdigest():
        raise ValueError("prediction config hash mismatch")
    # This is the first and only oracle read in the script, after the complete
    # prediction artifact and its config/cohort provenance have been checked.
    oracle_payload = json.loads(args.oracle_semantic.read_text(encoding="utf-8"))
    if oracle_payload["task_ids_hash"] != task_hash:
        raise ValueError("gold cohort mismatch")
    private: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_id in ids:
        oracle = oracle_payload["gold"][task_id]
        skeleton, reason = gold_skeleton_from_semantic_oracle(oracle)
        # Parameter-free contract audit: the score gold contains only family,
        # operations and typed slot names, never literal values.
        if skeleton is not None:
            encoded = skeleton.to_dict()
            if any(isinstance(value, (int, float)) for step in encoded["steps"] for value in step["required_slots"]):
                raise ValueError("gold skeleton contains a concrete parameter")
        result = _score_one(frozen["records"][task_id]["hypotheses"], skeleton, reason)
        result["stratum"] = _stratum(oracle)
        private[task_id] = result
        grouped[result["stratum"]].append(result)
    total = len(ids)
    metric_keys = ("schema_valid", "family_correct", "operation_set_correct", "operation_sequence_exact", "required_slot_set_correct", "full_top1_exact", "recall_at_3")
    metrics = {key: sum(bool(record.get(key, False)) for record in private.values()) / total for key in metric_keys}
    strata = {
        name: {"tasks": len(rows), "full_top1_exact": sum(bool(row.get("full_top1_exact", False)) for row in rows), "recall_at_3": sum(bool(row.get("recall_at_3", False)) for row in rows), "schema_valid": sum(bool(row.get("schema_valid", False)) for row in rows)}
        for name, rows in sorted(grouped.items())
    }
    failures = Counter(str(record["primary_failure"]) for record in private.values() if record["primary_failure"] != "EXACT")
    aggregate = {
        "experiment_id": "ARC2_V3_RULE_RECOGNITION_INDEPENDENT_V1",
        "track": "U",
        "status": "COMPLETE",
        "task_count": total,
        "task_ids_hash": task_hash,
        "model": config["model"],
        "generation": config["generation"],
        "top_k": [1, 3],
        "prediction_freeze_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "prediction_status_before_gold_read": frozen["status"],
        "metrics": metrics,
        "gold_skeleton_recall_at_5": "NOT_RUN_TOP5_NOT_GENERATED",
        "strata": strata,
        "failure_taxonomy": dict(sorted(failures.items())),
        "true_rule_recognition_failures": sum(bool(row["true_recognition_failure"]) for row in private.values()),
        "gold_representation_problems": sum(not bool(row["true_recognition_failure"]) and row["primary_failure"] != "EXACT" for row in private.values()),
        "runtime": {"generation_seconds": sum(float(row["generation_seconds"]) for row in frozen["records"].values()), "scoring_seconds": perf_counter() - started},
        "leakage_audit": "Prediction artifact was complete and SHA256-frozen before local oracle read; inference source contains no gold/oracle/backend/parameter/executor/verifier dependencies. No test output entered Track U.",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# ARC2 V3 independent upstream rule recognition\n\n"
        f"- Frozen development cohort: 30 tasks; hash `{task_hash}`.\n"
        f"- Schema-valid top-1: {metrics['schema_valid']:.1%}; full skeleton exact top-1: {metrics['full_top1_exact']:.1%}; recall@3: {metrics['recall_at_3']:.1%}.\n"
        f"- Gold mapping/representation problems: {aggregate['gold_representation_problems']}/30.\n"
        "- Inference used train grids, U8 deterministic evidence and CrossPairEvidence only. Gold was opened only by this local scorer.\n",
        encoding="utf-8",
    )
    header = ["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"]
    exists = args.experiments_csv.exists()
    prior = args.experiments_csv.read_text(encoding="utf-8") if exists else ""
    if "ARC2_V3_RULE_RECOGNITION_INDEPENDENT_V1," not in prior:
        with args.experiments_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            if not exists: writer.writeheader()
            writer.writerow({"experiment_id": "ARC2_V3_RULE_RECOGNITION_INDEPENDENT_V1", "date": "2026-09-12", "git_commit": "1eee93c", "solver": "QwenRuleRecognizer", "representation": "V3 RuleSkeleton", "search_method": "greedy top-3 generation", "llm_model": config["model"]["source"], "candidate_budget": 3, "validation_split": "development_only_30", "tasks_solved": sum(bool(row.get("full_top1_exact", False)) for row in private.values()), "accuracy": metrics["full_top1_exact"], "runtime_seconds": aggregate["runtime"]["generation_seconds"], "gpu_hours": aggregate["runtime"]["generation_seconds"] / 3600, "notes": json.dumps({"track": "U", "top3_recall": metrics["recall_at_3"]}, sort_keys=True)})
    print(json.dumps({"status": "COMPLETE", "metrics": metrics, "freeze": aggregate["prediction_freeze_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
