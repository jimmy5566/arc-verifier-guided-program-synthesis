"""Phase-A local oracle-skeleton backend audit; deliberately makes no LLM call."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from v3.evidence.extractor import extract_task_evidence
from v3.execution.rule_executor import RuleExecutor
from v3.pipeline import train_consistent_rule_specs
from v3.schema.rule_skeleton import OperationId, RuleSkeleton


def gold_skeleton_from_semantic_oracle(oracle: Mapping[str, Any]) -> tuple[RuleSkeleton | None, str | None]:
    """Map frozen legacy semantic operations to V3 semantics, without values."""
    if oracle["conditional_logic"]["enabled"]:
        return None, "DEFERRED_CAPABILITY_CONDITIONAL"
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
        return None, "MISSING_OPERATION_SEMANTICS_TRANSFORM"
    if "EXTRACT" in operations:
        return RuleSkeleton.from_operations(family, (OperationId.SELECT, OperationId.CROP)), None
    if "FILL" in operations:
        return RuleSkeleton.from_operations(family, (OperationId.SELECT, OperationId.FILL)), None
    return None, f"MISSING_OPERATION_SEMANTICS_{'_'.join(operations)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--oracle-semantic", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen train audit")
    cohort, oracle_payload, config = (json.loads(path.read_text(encoding="utf-8")) for path in (args.cohort, args.oracle_semantic, args.config))
    task_ids = tuple(cohort["task_ids"])
    task_hash = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    if len(task_ids) != 30 or task_hash != config["cohort"]["task_ids_hash"] or task_hash != oracle_payload["task_ids_hash"]:
        raise ValueError("requires exact frozen development-30 cohort")
    tasks = load_dataset(args.challenge_path)
    records: dict[str, dict[str, Any]] = {}
    executor = RuleExecutor()
    for task_id in task_ids:
        skeleton, unsupported = gold_skeleton_from_semantic_oracle(oracle_payload["gold"][task_id])
        if skeleton is None:
            records[task_id] = {"status": "EXECUTION_CAPABILITY_FAILURE", "unsupported_reason": unsupported}
            continue
        task = tasks[task_id]
        train = tuple((example.input.values, example.output.values) for example in task.train)
        evidence = extract_task_evidence(task)
        from v3.parameters.joint_solver import infer_parameters
        hypotheses = infer_parameters(skeleton, evidence)
        assignments = hypotheses.assignments()
        specs = train_consistent_rule_specs((skeleton,), evidence, train)
        records[task_id] = {
            "status": "TRAIN_CONSISTENT" if specs else "VERIFICATION_FAILURE" if assignments else "PARAMETER_INFERENCE_FAILURE",
            "gold_skeleton": skeleton.to_dict(),
            "parameter_assignment_count": len(assignments),
            "train_consistent_rulespecs": [spec.to_dict() for spec in specs],
        }
    artifact = {
        "experiment_id": "ARC2_V3_BACKEND_AUDIT_V1",
        "status": "TRAIN_RULESPEC_AUDIT_FROZEN_BEFORE_OPTIONAL_TEST_DIAGNOSTIC",
        "protocol": "Frozen oracle semantic IR supplied only parameter-free V3 skeletons. Evidence, parameter inference, executor, and verifier used train pairs only. No LLM, test output, Macro DSL, compiler, repair, or tool stitching was invoked.",
        "task_ids_hash": task_hash,
        "operation_audit": executor.operation_audit(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(records), "train_consistent": sum(item["status"] == "TRAIN_CONSISTENT" for item in records.values()), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
