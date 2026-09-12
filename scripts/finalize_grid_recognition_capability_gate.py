"""Publish the negative hard-gate diagnostic without starting A/B/C inference."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"; REPORTS = ROOT / "reports"; LEDGER = ROOT / "experiments/experiments.csv"


def digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    task_set = json.loads((ROOT / "configs/GRID_RECOGNITION_DIAGNOSTIC_V1_TASKS.json").read_text(encoding="utf-8"))
    private = json.loads((ROOT / "artifacts/grid_recognition_diagnostic_v1_capability_gate_private.json").read_text(encoding="utf-8"))
    if task_set["status"] != "BLOCKED_CAPABILITY_COVERAGE" or len(private["test_exact"]) >= 24:
        raise ValueError("this finalizer is only for a verified pre-formal capability gate block")
    frozen = {
        "experiment_id": "GRID_RECOGNITION_AND_END2END_DIAGNOSTIC_V1", "status": "BLOCKED_CAPABILITY_COVERAGE_BEFORE_FORMAL_A_B_C",
        "task_set_hash": digest(ROOT / "configs/GRID_RECOGNITION_DIAGNOSTIC_V1_TASKS.json"),
        "structured_hypothesis_ir_schema_hash": digest(ROOT / "src/recognition/structured_hypothesis_ir.py"),
        "feature_extractor_hash": digest(ROOT / "src/recognition/grid_feature_extractor.py"),
        "downstream_v1": {"macro_registry_hash": digest(ROOT / "configs/macro_capability_registry_v1.json"), "compiler_hash": digest(ROOT / "src/llm/macro_compiler_v1.py"), "family_first_hash": digest(ROOT / "src/llm/semantic_program_selector.py"), "q1_ontology_hash": digest(ROOT / "src/llm/parameter_semantic_ontology.py"), "r1_relation_hash": digest(ROOT / "src/llm/semantic_relation_normalizer.py")},
        "recognizer": {"model": "qwen-lm/qwen-3/Transformers/8b/1", "backend": "transformers_local", "dtype": "bfloat16", "temperature": 0, "thinking": False, "top_k": 1, "prompt_hash": hashlib.sha256(b"GRID_FEATURES_TO_STRUCTURED_HYPOTHESIS_IR_V1_NO_FREE_PROSE").hexdigest(), "executed": False},
        "prediction_freeze_protocol": "GRID_DIAGNOSTIC_PREDICTION_FREEZE_V1", "formal_passes": {"A_RECOGNITION_ONLY": 0, "B_ORACLE_IR_DOWNSTREAM": 0, "C_END_TO_END": 0},
        "gate": {key: task_set[key] for key in ("development_pool_size_after_exposure_exclusion", "excluded_historical_prompt_study_count", "train_consistent_oracle_pool_size", "test_exact_oracle_pool_size", "candidate_audit_totals", "prediction_freeze_hash", "gate_reason")},
        "protocol": {"development_only": True, "public_eval_used": False, "held_out_used": False, "challenge_like_used": False, "solutions_opened_only_after_oracle_prediction_freeze": private["prediction_freeze_hash"] == task_set["prediction_freeze_hash"], "oracle_details_public": False, "raw_grids_public": False, "raw_test_outputs_public": False},
    }
    frozen_path = ROOT / "configs/GRID_RECOGNITION_AND_END2END_DIAGNOSTIC_V1_FROZEN_CONFIG.json"; write_json(frozen_path, frozen)
    common = {"experiment_id": "GRID_RECOGNITION_AND_END2END_DIAGNOSTIC_V1", "status": "NOT_RUN_CAPABILITY_GATE_FAILED", "reason": task_set["gate_reason"], "task_count": 0, "metric_status": "N/A", "new_model_inference": False, "prediction_freeze_hash": task_set["prediction_freeze_hash"]}
    write_json(RESULTS / "GRID_RECOGNITION_ONLY_V1.json", common | {"condition": "A_RECOGNITION_ONLY", "oracle_access": False})
    write_json(RESULTS / "ORACLE_IR_DOWNSTREAM_V1.json", common | {"condition": "B_ORACLE_IR_DOWNSTREAM", "oracle_access": True, "gate_train_consistent": task_set["train_consistent_oracle_pool_size"], "gate_test_exact": task_set["test_exact_oracle_pool_size"]})
    write_json(RESULTS / "GRID_TO_END2END_V1.json", common | {"condition": "C_END_TO_END", "oracle_access": False})
    aggregate = common | {"condition": "AGGREGATE", "frozen_config": str(frozen_path.relative_to(ROOT)), "gate": frozen["gate"], "main_bottleneck": "DOWNSTREAM_CAPABILITY_COVERAGE_GATE_BLOCKED", "conclusion": "No valid 24-task controlled cohort exists under frozen DOWNSTREAM_V1; formal recognition/downstream/end-to-end metrics are intentionally not fabricated."}
    write_json(RESULTS / "GRID_RECOGNITION_AND_END2END_DIAGNOSTIC_V1.json", aggregate)
    report = """# Grid Recognition + End-to-End Diagnostic V1

## Hard-gate result

- Status: `BLOCKED_CAPABILITY_COVERAGE_BEFORE_FORMAL_A_B_C`.
- Scope: development split only; public evaluation, held-out, challenge-like and full-1000 inference were not used.
- Development pool after historical prompt-study exclusion: {pool}.
- Frozen DOWNSTREAM_V1 audit: {generated:,} candidates generated, {compiled:,} compiler-valid, {train_consistent} train-consistent, {test_exact} test-exact.
- Oracle prediction hash was frozen before the development solution file was opened: `{freeze}`.

## Conditions

| Condition | Status | Reason |
| --- | --- | --- |
| A Recognition Only | Not run | No 24/24 capability-covered cohort |
| B Oracle IR → Downstream | Not run | No test-exact oracle task exists in audited pool |
| C End-to-End | Not run | Formal prediction would confound recognition with known downstream coverage gap |

## Conclusion

The requested controlled diagnostic cannot validly start with frozen DOWNSTREAM_V1: the hard gate found zero test-exact oracle tasks, far below the required 24. This is evidence of a `DOWNSTREAM_CAPABILITY` bottleneck at the cohort-construction stage, not evidence about Qwen grid-recognition quality. No recognizer, GPU inference, prompt tuning, retry, or manual correction was run.

## Safety

- No raw ARC grids, solutions, oracle programs, model outputs, or test predictions are published.
- Oracle program details remain in an ignored local artifact; public artifacts contain aggregate counts and a prediction-freeze hash only.
""".format(pool=task_set["development_pool_size_after_exposure_exclusion"], generated=task_set["candidate_audit_totals"]["generated"], compiled=task_set["candidate_audit_totals"]["compiled"], train_consistent=task_set["train_consistent_oracle_pool_size"], test_exact=task_set["test_exact_oracle_pool_size"], freeze=task_set["prediction_freeze_hash"])
    report_path = REPORTS / "grid_recognition_and_end2end_diagnostic_v1.md"
    if report_path.exists(): raise FileExistsError(report_path)
    report_path.write_text(report, encoding="utf-8")
    with LEDGER.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle); fields = list(reader.fieldnames or []); rows = [row for row in reader if row["experiment_id"] != frozen["experiment_id"]]
    rows.append({"experiment_id": frozen["experiment_id"], "date": str(date.today()), "git_commit": "pending", "solver": "Structured grid recognition/end-to-end diagnostic", "representation": "StructuredHypothesisIRV1 + frozen DOWNSTREAM_V1", "search_method": "development-only oracle capability gate", "llm_model": frozen["recognizer"]["model"], "candidate_budget": 1, "validation_split": "development_only_capability_gate", "tasks_solved": 0, "accuracy": 0.0, "runtime_seconds": 0.0, "gpu_hours": 0.0, "notes": json.dumps({"status": frozen["status"], "eligible_exact": task_set["test_exact_oracle_pool_size"], "formal_aborted": True, "arc_data_used": True}, sort_keys=True)})
    with LEDGER.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": frozen["status"], "frozen": str(frozen_path), "report": str(report_path)}))


if __name__ == "__main__": main()
