"""Build the final auditable LLM experiment report after every frozen run ends."""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
REPORTS = ROOT / "reports"


def read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE_SCORED_AFTER_PREDICTION_FREEZE":
        raise RuntimeError(f"not a completed scored frozen condition: {path}")
    return payload


def baseline_ids() -> set[str]:
    v0 = json.loads((RESULTS / "SOLVER_LIB_V0.json").read_text(encoding="utf-8"))
    ids = set(v0["unique_solved_task_ids"])
    for filename in ("SOLVER_LIB_V1_OBJECT.json", "SOLVER_LIB_V2_RELATION_COMPOSITION.json", "SOLVER_LIB_V3_PATTERN.json", "ARC2_CAPABILITY_SYSTEM_V1.json"):
        ids.update(json.loads((RESULTS / filename).read_text(encoding="utf-8"))["newly_solved_task_ids"])
    if len(ids) != 22:
        raise RuntimeError(f"expected frozen deterministic baseline of 22, got {len(ids)}")
    return ids


def update_csv(result: dict) -> None:
    path = ROOT / "experiments" / "experiments.csv"
    fields = ["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["experiment_id"] != result["experiment_id"]]
    rows.append({
        "experiment_id": result["experiment_id"], "date": str(date.today()), "git_commit": "uncommitted",
        "solver": "LLM Hypothesis Generator V1", "representation": "frozen canonical registry structured programs",
        "search_method": "Qwen3 generation + hard train verification", "llm_model": result["frozen_config"]["model"]["model"],
        "candidate_budget": str(result["frozen_config"]["generation"]["candidate_budget"]), "validation_split": "development + held_out + challenge_like + official_training_all",
        "tasks_solved": str(result["full_1000"]["exact_solved"]), "accuracy": str(result["full_1000"]["exact_solved"] / 1000),
        "runtime_seconds": str(result["total_runtime_seconds"]), "gpu_hours": str(result["total_runtime_seconds"] / 3600),
        "notes": json.dumps({"status": result["status"], "newly_solved": result["newly_solved_vs_22"], "bottleneck": result["bottleneck_conclusion"]["primary"]}),
    })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    confirmation = read(RESULTS / "LLM_CONFIRMATION_DEV_100.json")
    held = read(RESULTS / "LLM_HELD_OUT_FROZEN_V1.json")
    challenge = read(RESULTS / "LLM_CHALLENGE_LIKE_FROZEN_V1.json")
    full = read(RESULTS / "LLM_FULL_1000_FROZEN_V1.json")
    if full["task_count"] != 1000:
        raise RuntimeError("full result does not cover 1000 tasks")
    config = json.loads((ROOT / "configs" / "frozen_llm_config_v1.json").read_text(encoding="utf-8"))
    pilot = json.loads((RESULTS / "LLM_PILOT_97.json").read_text(encoding="utf-8"))
    baseline = baseline_ids()
    exact = set(full["exact_solved_task_ids"])
    new_ids = sorted(exact - baseline)
    aggregate_statuses: dict[str, int] = {}
    for split in (confirmation, held, challenge, full):
        for status, count in split["status_counts"].items():
            aggregate_statuses[status] = aggregate_statuses.get(status, 0) + count
    total_runtime = sum(item["runtime_seconds"] for item in (confirmation, held, challenge, full))
    primary = "LLM capability-API comprehension bottleneck" if full["train_consistent_rate"] == 0 else "Multiple bottlenecks"
    result = {
        "experiment_id": "LLM_HYPOTHESIS_GENERATOR_V1_REAL",
        "status": "COMPLETE",
        "frozen_config": config,
        "frozen_deterministic_baseline": {"cumulative_solved": 22, "task_ids": sorted(baseline), "regression": json.loads((RESULTS / "ARC2_CAPABILITY_SYSTEM_V1.json").read_text(encoding="utf-8"))["frozen_regression"]},
        "pilot_exposed": pilot,
        "split_evaluations": {"confirmation_development_100": confirmation, "held_out": held, "challenge_like": challenge},
        "full_1000": full,
        "llm_structured_program_exact_solved": full["exact_solved"],
        "newly_solved_vs_22": len(new_ids),
        "newly_solved_task_ids": new_ids,
        "composition_only_task_ids": new_ids,
        "deterministic_plus_llm_union": len(baseline | exact),
        "aggregate_status_counts": aggregate_statuses,
        "total_runtime_seconds": total_runtime,
        "leakage_audit": [
            "Each inference condition opened only training challenges and checkpointed predictions before scoring.",
            "Each finalizer refuses solution access until every declared task record is present and frozen.",
            "The old 113-task pilot predates the canonical-ID schema infrastructure fix and is excluded from final pooled scores.",
            "No task-ID branching, capability addition, router, or ranker was used after configuration freeze.",
        ],
        "bottleneck_conclusion": {
            "primary": primary,
            "evidence": {
                "full_schema_valid_rate": full["schema_valid_rate"], "full_type_valid_rate": full["type_valid_rate"],
                "full_executable_rate": full["executable_rate"], "full_train_consistent_rate": full["train_consistent_rate"],
                "composition_only_gain": len(new_ids), "direct_grid_pilot_exact": pilot["shared_25_comparison"]["direct_grid_diagnostic_only"]["exact_solved"],
            },
        },
    }
    RESULTS.joinpath("LLM_HYPOTHESIS_GENERATOR_V1_REAL.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORTS.mkdir(exist_ok=True)
    REPORTS.joinpath("llm_hypothesis_generator_v1.md").write_text("# LLM Hypothesis Generator V1 — Final Frozen Evaluation\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    REPORTS.joinpath("llm_capability_gap_candidates.md").write_text("# LLM Capability Gap Candidates\n\nNo new capability was added. The final frozen evidence is interpreted only after all split and full results were scored; it distinguishes API-comprehension failure from missing capability evidence in `LLM_HYPOTHESIS_GENERATOR_V1_REAL.json`.\n", encoding="utf-8")
    REPORTS.joinpath("data_exposure_audit.md").write_text("# Data Exposure Audit\n\n" + "\n".join(f"- {line}" for line in result["leakage_audit"]) + "\n", encoding="utf-8")
    update_csv(result)
    print(json.dumps({"status": result["status"], "full_exact": full["exact_solved"], "newly_solved": len(new_ids)}, indent=2))


if __name__ == "__main__":
    main()
