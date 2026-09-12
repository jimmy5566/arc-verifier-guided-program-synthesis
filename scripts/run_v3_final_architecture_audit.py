"""Frozen30 train-only audit of complete V3 RuleSpecs.

The private semantic oracle may select an operation family only.  Every
concrete RuleSpec parameter remains derived from that task's train pairs, and
the audit never opens a solution file or a test output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from v3.binding import BindingError, InstanceBinder
from v3.evidence.extractor import extract_task_evidence
from v3.execution.rule_executor import RuleExecutor
from v3.schema.rule_skeleton import OperationId, RuleSkeleton
from v3.upstream import complete_rule_specs
from v3.validation import RuleSpecPreflightValidator
from v3.verification.verifier import HardVerifier


def skeleton_from_oracle(semantic: Mapping[str, Any]) -> tuple[RuleSkeleton | None, str | None]:
    """Generic semantic-family mapping; values are never supplied by gold."""
    if semantic["conditional_logic"]["enabled"]:
        return None, "MISSING_GENERIC_CONDITIONAL_ROLE_OPERATION"
    family, operations = str(semantic["primary_family"]), tuple(semantic["operations"])
    if "REPEAT" in operations:
        return RuleSkeleton.from_operations(family, (OperationId.SELECT, OperationId.REPEAT)), None
    if operations == ("RECOLOR",):
        return RuleSkeleton.from_operations(family, (OperationId.SELECT, OperationId.RECOLOR)), None
    if operations == ("TRANSFORM",):
        transform = semantic["spatial_transform"]["transform"]
        if transform == "ROTATE": return RuleSkeleton.from_operations(family, (OperationId.ROTATE,)), None
        if transform == "REFLECT": return RuleSkeleton.from_operations(family, (OperationId.REFLECT,)), None
        return None, "MISSING_GENERIC_TRANSFORM_OPERATION"
    if "EXTRACT" in operations:
        return RuleSkeleton.from_operations(family, (OperationId.SELECT, OperationId.CROP)), None
    if "FILL" in operations:
        return RuleSkeleton.from_operations(family, (OperationId.SELECT, OperationId.FILL)), None
    return None, "MISSING_GENERIC_OPERATION_SEMANTICS"


def _classify(spec: Any, train: tuple[tuple[np.ndarray, np.ndarray], ...], validator: RuleSpecPreflightValidator, binder: InstanceBinder, executor: RuleExecutor, verifier: HardVerifier) -> str:
    preflight = validator.validate(spec)
    if not preflight.passed: return "PREFLIGHT_FAILURE"
    for source, expected in train:
        try:
            bound = binder.bind(spec, source)
        except BindingError:
            return "BINDING_FAILURE"
        try:
            actual = executor.execute_bound(bound, source)
        except (TypeError, ValueError):
            return "EXECUTION_FAILURE"
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            return "VERIFIER_FAILURE"
    return "COVERED" if verifier.verify(spec, train).passed else "VERIFIER_FAILURE"


def _semantic_taxonomy(semantic: Mapping[str, Any], status: str, reason: str | None) -> str:
    """Generic category attribution; it never branches on a task identifier."""
    if reason: return reason
    operations = set(semantic["operations"])
    if "REPEAT" in operations and status != "COVERED":
        return "REPEAT_MOTIF_OR_TERMINATION_SEMANTICS_INSUFFICIENT"
    if "EXTRACT" in operations and status != "COVERED":
        return "REFERENCE_ROLE_CROP_RELATION_INSUFFICIENT"
    if semantic["conditional_logic"]["enabled"]:
        return "CONDITIONAL_ROLE_SEMANTICS_INSUFFICIENT"
    if status == "PREFLIGHT_FAILURE": return "NO_COMPLETE_RULESPEC_FROM_GENERIC_TRAIN_FACTS"
    if status == "VERIFIER_FAILURE": return "GENERIC_OPERATION_COMPOSITION_INSUFFICIENT"
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--oracle-semantic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite frozen audit")
    cohort, oracle = (json.loads(path.read_text(encoding="utf-8")) for path in (args.cohort, args.oracle_semantic))
    task_ids = tuple(cohort["task_ids"])
    task_hash = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    if len(task_ids) != 30 or task_hash != oracle["task_ids_hash"]:
        raise ValueError("requires exact frozen development30 cohort")
    tasks = load_dataset(args.challenge_path)
    validator, binder, executor, verifier = RuleSpecPreflightValidator(), InstanceBinder(), RuleExecutor(), HardVerifier()
    records: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        skeleton, reason = skeleton_from_oracle(oracle["gold"][task_id])
        semantic = oracle["gold"][task_id]
        if skeleton is None:
            records[task_id] = {"status": "PREFLIGHT_FAILURE", "reason": reason, "taxonomy": _semantic_taxonomy(semantic, "PREFLIGHT_FAILURE", reason), "complete_rulespec_candidates": 0, "stage": {"preflight": False, "binding": False, "execution": False, "exact": False}}
            continue
        task = tasks[task_id]
        train = tuple((item.input.values, item.output.values) for item in task.train)
        candidates = complete_rule_specs((skeleton,), extract_task_evidence(task))
        statuses = [_classify(item, train, validator, binder, executor, verifier) for item in candidates]
        if "COVERED" in statuses: status = "COVERED"
        elif "VERIFIER_FAILURE" in statuses: status = "VERIFIER_FAILURE"
        elif "EXECUTION_FAILURE" in statuses: status = "EXECUTION_FAILURE"
        elif "BINDING_FAILURE" in statuses: status = "BINDING_FAILURE"
        else: status = "PREFLIGHT_FAILURE"
        records[task_id] = {
            "status": status, "reason": reason, "taxonomy": _semantic_taxonomy(semantic, status, reason),
            "complete_rulespec_candidates": len(candidates), "candidate_status_counts": dict(Counter(statuses)),
            "stage": {
                "preflight": bool(candidates) and any(item != "PREFLIGHT_FAILURE" for item in statuses),
                "binding": any(item in {"COVERED", "VERIFIER_FAILURE", "EXECUTION_FAILURE"} for item in statuses),
                "execution": any(item in {"COVERED", "VERIFIER_FAILURE"} for item in statuses),
                "exact": status == "COVERED",
            },
        }
    counts = Counter(record["status"] for record in records.values())
    taxonomy = Counter(record["taxonomy"] for record in records.values())
    artifact = {
        "experiment_id": "ARC2_V3_FINAL_ARCHITECTURE_HARDENING",
        "status": "TRAIN_ONLY_COMPLETE_RULESPEC_AUDIT_FROZEN",
        "task_ids_hash": task_hash,
        "protocol": "Private semantic oracle selected only generic operation family. Train pairs derived every concrete RuleSpec value. Validator, binder, executor and HardVerifier used train pairs only; no Qwen, test grid/output or solution file was opened.",
        "coverage": {"covered": counts["COVERED"], "total": len(records)},
        "stage_coverage": {name: sum(bool(record["stage"][name]) for record in records.values()) for name in ("preflight", "binding", "execution", "exact")},
        "failures": {key.lower(): counts[key] for key in ("PREFLIGHT_FAILURE", "BINDING_FAILURE", "EXECUTION_FAILURE", "VERIFIER_FAILURE")},
        "semantic_taxonomy": dict(sorted(taxonomy.items())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"coverage": artifact["coverage"], "failures": artifact["failures"], "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
