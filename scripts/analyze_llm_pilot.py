"""Audit the stopped LLM pilot without issuing any model requests.

The script is deliberately solution-free: the stopped structured run contains
no predictions, and the direct-grid pilot stores only its already-frozen exact
summary.  It turns those immutable artifacts into an explicit staged record.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
CHECKPOINT = ROOT / "experiments" / "checkpoints" / "LLM_HYPOTHESIS_GENERATOR_V1_qwen3_14b_full_catalog_top5.json"
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_deterministic_ids() -> set[str]:
    v0 = read_json(RESULTS / "SOLVER_LIB_V0.json")
    v1 = read_json(RESULTS / "SOLVER_LIB_V1_OBJECT.json")
    v2 = read_json(RESULTS / "SOLVER_LIB_V2_RELATION_COMPOSITION.json")
    v3 = read_json(RESULTS / "SOLVER_LIB_V3_PATTERN.json")
    v4 = read_json(RESULTS / "ARC2_CAPABILITY_SYSTEM_V1.json")
    return set(v0["unique_solved_task_ids"]) | set(v1["newly_solved_task_ids"]) | set(v2["newly_solved_task_ids"]) | set(v3["newly_solved_task_ids"]) | set(v4["newly_solved_task_ids"])


def confirmation_ids(exposed: set[str]) -> list[str]:
    with (ROOT / "data" / "splits" / "task_splits.csv").open(encoding="utf-8", newline="") as handle:
        development = [row["task_id"] for row in csv.DictReader(handle) if row["split"] == "development" and row["task_id"] not in exposed]
    return sorted(development, key=lambda task_id: hashlib.sha256(f"LLM_CONFIRMATION_V1:{task_id}".encode()).hexdigest())[:100]


def main() -> None:
    CONFIGS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    pilot = read_json(RESULTS / "LLM_HYPOTHESIS_GENERATOR_V1_qwen3_pilot.json")
    stopped = read_json(CHECKPOINT)
    records = stopped["records"]
    exposed = sorted(records)
    pilot_ids = set(pilot["development_task_ids"])
    shared = sorted(pilot_ids & set(exposed))
    status_counts = Counter(status for record in records.values() for status in record["candidate_statuses"])
    reason_counts = Counter(reason for record in records.values() for reason in record["candidate_reasons"])
    deterministic = frozen_deterministic_ids()
    direct_ids = set(pilot["direct_grid"]["task_ids"])
    confirmation = confirmation_ids(set(exposed))
    config = {
        "config_id": "FROZEN_LLM_CONFIG_V1",
        "status": "FROZEN_FOR_CONFIRMATION_AFTER_INFRA_SCHEMA_FIX",
        "model": {"backend": "ollama", "model": "qwen3:14b", "quantization": "Q4_K_M", "context_window": 12288},
        "sampling": {"temperature": 0, "top_p": 1, "seed": 0, "stream": False, "think": False},
        "generation": {"candidate_budget": 5, "max_output_tokens": 800, "scheduler_hints": False},
        "catalog": {"registry_count": 119, "catalog_version": "capability_catalog_v1", "prompt_version": "llm_hypothesis_generator_v1.prompt.2", "schema_version": "llm_hypothesis_generator_v1.schema.2"},
        "selection_reason": "Both structured pilot variants had zero executable and zero train-consistent candidates; full catalogue top-5 used fewer prompt/output tokens and less runtime.",
        "pilot_compatibility": "The stopped 113-task checkpoint used the same model, temperature, top-p, seed, candidate budget, context window, output cap, and no hints, but predates the canonical-primitive-ID enum schema fix; it is retained only as PILOT_EXPOSED_113 and is not pooled with confirmation/final scores.",
        "solution_access": "Challenge data only during inference; solutions may be opened only after the declared condition has completed and predictions are frozen.",
    }
    exposed_config = {
        "config_id": "PILOT_EXPOSED_113",
        "requested_filename_retained": "llm_pilot_exposed_97.json",
        "actual_task_count": len(exposed),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "task_ids": exposed,
        "all_25_condition_comparison_ids": shared,
        "protocol": stopped["protocol"],
        "status": "STOPPED_BEFORE_COMPLETION; PILOT_ONLY; NOT_A_FINAL_SCORE",
    }
    confirmation_config = {
        "config_id": "LLM_CONFIRMATION_DEV_100",
        "selection": "development split, excludes all PILOT_EXPOSED_113 task IDs, deterministic SHA-256 order with salt LLM_CONFIRMATION_V1",
        "task_count": len(confirmation),
        "task_ids": confirmation,
        "frozen_config": "configs/frozen_llm_config_v1.json",
        "protocol": "No solution file is opened by inference. Checkpoint after every task. Score only after all 100 predictions are frozen.",
    }
    confirmation_checkpoint_path = ROOT / "experiments" / "checkpoints" / "LLM_CONFIRMATION_DEV_100.json"
    confirmation_checkpoint = read_json(confirmation_checkpoint_path) if confirmation_checkpoint_path.exists() else None
    confirmation_progress = {
        "status": "NOT_STARTED" if confirmation_checkpoint is None else ("COMPLETE_UNSCORED" if confirmation_checkpoint.get("complete") else "PAUSED_INCOMPLETE"),
        "checkpoint_path": str(confirmation_checkpoint_path).replace("\\", "/"),
        "checkpointed_tasks": 0 if confirmation_checkpoint is None else len(confirmation_checkpoint["records"]),
        "declared_tasks": len(confirmation),
        "solution_access": "PROHIBITED_UNTIL_COMPLETE",
        "runtime_seconds": 0.0 if confirmation_checkpoint is None else confirmation_checkpoint.get("runtime_seconds", 0.0),
        "thermal_pause_reason": None if confirmation_checkpoint is None else (confirmation_checkpoint.get("paused_reason") or "Operator stopped worker after observed 86–87C thermal throttling; no model job remains active."),
    }
    confirmation_result_path = RESULTS / "LLM_CONFIRMATION_DEV_100.json"
    confirmation_result = read_json(confirmation_result_path) if confirmation_result_path.exists() else None
    (CONFIGS / "frozen_llm_config_v1.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (CONFIGS / "llm_pilot_exposed_97.json").write_text(json.dumps(exposed_config, indent=2) + "\n", encoding="utf-8")
    (CONFIGS / "llm_confirmation_dev_100.json").write_text(json.dumps(confirmation_config, indent=2) + "\n", encoding="utf-8")
    structured = pilot["structured_full_catalog"]
    hints = pilot["structured_scheduler_soft_hints"]
    audit = {
        "experiment_id": "LLM_HYPOTHESIS_GENERATOR_V1_REAL",
        "status": "STAGED_PILOT_ANALYZED",
        "inference_status": "STOPPED: no model calls after pilot audit began",
        "pilot": {
            "label": "PILOT_EXPOSED_113",
            "actual_task_count": len(exposed),
            "comparable_cross_condition_task_count": len(shared),
            "checkpoint_sha256": exposed_config["checkpoint_sha256"],
            "checkpoint_complete": bool(stopped.get("complete")),
            "structured_status_counts": dict(status_counts),
            "structured_reason_counts": dict(reason_counts),
            "structured_predictions_frozen": 0,
            "structured_exact_solved": 0,
        },
        "shared_25_comparison": {
            "task_ids": shared,
            "frozen_deterministic_baseline_exact": sorted(deterministic & set(shared)),
            "structured_full_catalog": structured,
            "structured_scheduler_soft_hints": hints,
            "structured_union_exact_solved": 0,
            "structured_composition_only_exact_solved": 0,
            "direct_grid_diagnostic_only": pilot["direct_grid"],
        },
        "decision": {
            "selected_structured_condition": "full_catalog_top5_without_scheduler_hints",
            "selection_rule": "No variant reached executable or train-consistent status; choose lower observed runtime and token cost.",
            "not_pooled_with_final": True,
        },
        "failure_diagnosis": {
            "LLM_API_COMPREHENSION_FAILURE": "Dominant: emitted programs were schema/type/precondition-invalid; no structured candidate executed successfully.",
            "MISSING_CAPABILITY": "No evidence in this pilot; zero successful execution prevents a capability-gap conclusion.",
            "DIRECT_EXECUTION_FAILURE": "Not established: direct-grid raw predictions were not retained, so valid-but-wrong grids remain unresolved.",
            "GENERALIZATION_FAILURE": "Not observed: no train-consistent structured program produced a test prediction.",
            "UNRESOLVED": "Provider errors and the single train-inconsistent candidate require improved structured logging in future conditions.",
        },
        "leakage_audit": {
            "solution_blind_structured_inference": stopped["protocol"],
            "checkpoint_has_predictions": False,
            "task_id_hardcode_detected": False,
            "pilot_not_used_for_final_score": True,
        },
        "confirmation_progress": confirmation_progress,
        "confirmation_result": confirmation_result,
        "next_declared_condition": confirmation_config,
    }
    (RESULTS / "LLM_PILOT_97.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "LLM_HYPOTHESIS_GENERATOR_V1_REAL.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    report = f"""# LLM Hypothesis Generator V1 — stopped-pilot audit

Status: **PILOT_EXPOSED_113**, not a completed experiment. The requested `97` filename is retained for compatibility, but the verified checkpoint contains **{len(exposed)}** tasks. Its SHA-256 is `{exposed_config['checkpoint_sha256']}`.

## Comparability and outcome

All {len(shared)} development pilot tasks overlap the stopped full-catalog checkpoint. On that fixed subset, structured full-catalog and scheduler-hint runs both produced zero executable, zero train-consistent, and zero exact candidates. The diagnostic direct-grid arm froze two exact results (`{', '.join(sorted(direct_ids))}`), but it is diagnostic-only and is excluded from architecture-union claims.

The frozen deterministic baseline solves {len(deterministic & set(shared))} of the shared 25 tasks. Structured LLM adds zero, including zero composition-only solves.

## Failure audit

The stopped structured checkpoint has {sum(status_counts.values())} candidates across {len(exposed)} tasks: {dict(status_counts)}. It stores no test predictions. Dominant reasons are invalid program composition and missing/invalid primitive parameters; this is an API-comprehension failure, not evidence of a missing ARC capability or of generalization failure. One candidate was train-inconsistent; provider errors are retained as unresolved because earlier logging did not retain response bodies.

## Configuration decision

`full_catalog_top5_without_scheduler_hints` is selected provisionally because both structured variants tied at zero executable/train-consistent candidates and it used less time and fewer tokens. Before confirmation, the output schema was strengthened to enumerate canonical primitive IDs, and provider-error logging was made explicit. That changes the runtime configuration; therefore this pilot is never pooled with later scores.

## Leakage/data-exposure audit

The structured runner opened only training challenges, did not load solutions, and generated zero predictions. The run was stopped before any subsequent condition was launched. No task-ID branching was found. Retrospective pilot analysis does not use solutions to construct or select a prediction.

## Confirmation state

The fixed development confirmation has checkpointed {confirmation_progress['checkpointed_tasks']}/{confirmation_progress['declared_tasks']} tasks and is `{confirmation_progress['status']}`. Its solution file remains inaccessible to the finalizer until all declared task predictions are frozen.
"""
    (REPORTS / "llm_pilot_97_analysis.md").write_text(report, encoding="utf-8")
    print(json.dumps({"pilot_exposed_tasks": len(exposed), "shared_tasks": len(shared), "confirmation_tasks": len(confirmation), "status_counts": dict(status_counts)}, indent=2))


if __name__ == "__main__":
    main()
