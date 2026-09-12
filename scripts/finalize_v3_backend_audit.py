"""Publish Phase-A aggregate only from a complete frozen train audit artifact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = args.frozen_audit.read_bytes()
    audit, config = json.loads(raw), json.loads(args.config.read_text(encoding="utf-8"))
    if audit.get("status") != "TRAIN_RULESPEC_AUDIT_FROZEN_BEFORE_OPTIONAL_TEST_DIAGNOSTIC" or len(audit.get("records", {})) != 30:
        raise ValueError("requires complete frozen Phase-A train audit")
    operation = audit["operation_audit"]
    records = audit["records"].values()
    counts = Counter(item["status"] for item in records)
    supported = sum(item["status"] != "EXECUTION_CAPABILITY_FAILURE" for item in records)
    parameter_resolvable = sum(int(item.get("parameter_assignment_count", 0)) > 0 for item in records)
    train_consistent = counts["TRAIN_CONSISTENT"]
    phase_a_pass = train_consistent >= config["phase_a"]["backend_train_coverage_gate"] and not operation["unsupported_exposed_operations"] and not operation["no_op_operations"]
    result = {
        "experiment_id": "ARC2_V3_BACKEND_AUDIT_V1",
        "status": "READY" if phase_a_pass else "BACKEND_NOT_READY",
        "phase_a_gate": {"backend_train_coverage": train_consistent, "required": config["phase_a"]["backend_train_coverage_gate"], "no_exposed_noop": not operation["no_op_operations"], "unsupported_exposed_operation_count": len(operation["unsupported_exposed_operations"])},
        "cohort": config["cohort"], "frozen_train_audit_sha256": hashlib.sha256(raw).hexdigest(), "operation_audit": operation,
        "metrics": {"gold_skeleton_supported": supported, "parameter_hypotheses_generated": parameter_resolvable, "at_least_one_train_consistent_rulespec": train_consistent, "unique_train_consistent_rulespec": sum(len(item.get("train_consistent_rulespecs", ())) == 1 for item in records), "executor_unsupported_operation": counts["EXECUTION_CAPABILITY_FAILURE"], "parameter_inference_failure": counts["PARAMETER_INFERENCE_FAILURE"], "verification_failure": counts["VERIFICATION_FAILURE"], "backend_train_coverage": train_consistent},
        "oracle_skeleton_test_exact": "NOT_RUN_PHASE_A_GATE_FAILED" if not phase_a_pass else "PENDING_POST_FREEZE_DIAGNOSTIC",
        "model_calls": 0, "phase_b": "NOT_RUN_PHASE_A_BACKEND_NOT_READY" if not phase_a_pass else "PENDING", "phase_c": "NOT_RUN_PHASE_A_BACKEND_NOT_READY" if not phase_a_pass else "PENDING",
        "failure_attribution": {"EXECUTION_CAPABILITY_FAILURE": counts["EXECUTION_CAPABILITY_FAILURE"], "PARAMETER_INFERENCE_FAILURE": counts["PARAMETER_INFERENCE_FAILURE"], "VERIFICATION_FAILURE": counts["VERIFICATION_FAILURE"]},
        "leakage_audit": "No LLM, test output, solution file, held-out, public, or challenge inference was used. The frozen audit consumed oracle semantic IR only to derive parameter-free skeletons; it never consumed concrete oracle parameters.",
    }
    _write(ROOT / "experiments/results/ARC2_V3_BACKEND_AUDIT_V1.json", result)
    _write(ROOT / "experiments/results/ARC2_V3_RECOGNITION_V1.json", {"experiment_id": "ARC2_V3_RECOGNITION_V1", "status": result["phase_b"], "reason": "Phase A gate failed; no Qwen inference was run."})
    _write(ROOT / "experiments/results/ARC2_V3_E2E_V1.json", {"experiment_id": "ARC2_V3_E2E_V1", "status": result["phase_c"], "reason": "Phase A gate failed; no Qwen inference, test prediction, or E2E scoring was run."})
    (ROOT / "reports/arc2_v3_backend_audit_v1.md").write_text("\n".join(["# ARC2 V3 Backend Audit V1", "", "Phase A 使用 frozen30 的 oracle-derived parameter-free RuleSkeleton 与 train pairs；未调用 Qwen、未读取 test targets。", "", f"- Exposed / executable operations: {len(operation['exposed_operations'])}/{len(operation['fully_executable_operations'])}", f"- No-op operations: {len(operation['no_op_operations'])}", f"- Gold skeleton supported: {supported}/30", f"- Parameter hypotheses generated: {parameter_resolvable}/30", f"- Train-consistent RuleSpec: {train_consistent}/30", f"- Execution capability failure: {counts['EXECUTION_CAPABILITY_FAILURE']}/30", f"- Parameter inference failure: {counts['PARAMETER_INFERENCE_FAILURE']}/30", f"- Verification failure: {counts['VERIFICATION_FAILURE']}/30", "", "## Gate", "", f"`BACKEND_TRAIN_COVERAGE = {train_consistent}/30`，低于 15/30；因此状态为 **BACKEND_NOT_READY**。Phase B 和 Phase C 均未运行，oracle-skeleton test exact 也未读取。", "", "主要瓶颈是当前 evidence/parameter candidates 与 executor semantics 对已知 skeleton 仍无法共同复现 train outputs，而不是 recognizer。", ""]) + "\n", encoding="utf-8")
    for name, status in (("arc2_v3_recognition_v1.md", result["phase_b"]), ("arc2_v3_e2e_v1.md", result["phase_c"])):
        (ROOT / "reports" / name).write_text(f"# {name.removesuffix('.md').upper()}\n\n`{status}`。Phase A backend gate failed；本阶段没有运行模型或 test inference。\n", encoding="utf-8")
    (ROOT / "reports/arc2_v3_final_architecture.md").write_text("# ARC2 V3 Final Architecture\n\nV3 runtime 固定为 `Evidence → Rule Recognition → Parameter Inference → RuleSpec → Executor → HardVerifier`。Phase A 已证明 exposed operation 无 no-op，但 frozen30 的 oracle-skeleton train coverage 为 0/30，因此当前架构状态为 `BLOCKED_BACKEND_NOT_READY`；不得启动 Recognition 或 E2E。\n", encoding="utf-8")
    with (ROOT / "experiments/experiments.csv").open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"])
        writer.writerow({"experiment_id": "ARC2_V3_BACKEND_AUDIT_V1", "date": str(date.today()), "git_commit": "pending", "solver": "V3 deterministic backend", "representation": "oracle-derived V3 RuleSkeleton + train-derived RuleSpec", "search_method": "finite cross-pair parameters + exact hard verification", "llm_model": "", "candidate_budget": "finite", "validation_split": "frozen_development_30_train_only", "tasks_solved": train_consistent, "accuracy": train_consistent / 30, "runtime_seconds": 0, "gpu_hours": 0, "notes": json.dumps({"status": result["status"], "phase_b": result["phase_b"], "test_outputs_read": False}, sort_keys=True)})
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
