"""Reproducible LLM program-synthesis experiment entry point.

With a configured real provider this module is the single place that may call
it.  In an unconfigured environment it writes a complete, explicitly blocked
research record; it does not substitute canned or symbolic proposals for an
LLM result.
"""
from __future__ import annotations

import csv
from datetime import date
import json
from pathlib import Path

from llm.catalog import write_catalog
from llm.comprehension import build_comprehension_benchmark
from llm.protocol import blocked_result
from llm.providers import provider_audit
from primitives.registry import REGISTRY, validate_registry


EXPERIMENT_ID = "LLM_HYPOTHESIS_GENERATOR_V1"


def _frozen_baseline() -> dict[str, object]:
    path = Path("experiments/results/ARC2_CAPABILITY_SYSTEM_V1.json")
    if not path.exists():
        raise FileNotFoundError("frozen ARC2_CAPABILITY_SYSTEM_V1 results are required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "experiment_id": payload["experiment_id"],
        "cumulative_solved": payload["cumulative_solved"],
        "generic_composer_exact_solved": payload["generic_composer"]["scheduled"]["exact_solved_tasks"],
        "v4_unique_solved": payload["v4_unique_solved"],
        "frozen_regression": payload["frozen_regression"],
    }


def _write_reports(payload: dict[str, object]) -> None:
    Path("reports").mkdir(parents=True, exist_ok=True)
    blocked = payload["status"] == "BLOCKED_NO_LLM_BACKEND"
    report = "# LLM Hypothesis Generator V1\n\n"
    report += "## Status\n\n" + str(payload["status"]) + ". " + str(payload.get("reason", "")) + "\n\n"
    report += "## Frozen baseline\n\n```json\n" + json.dumps(payload["frozen_deterministic_baseline"], indent=2) + "\n```\n\n"
    report += "## Protocol\n\nThe capability catalogue is generated directly from the frozen registry. A model may only emit canonical primitive IDs in the strict JSON schema. Each candidate is schema-validated, type-checked, symbolically executed, and exactly verified on train pairs before a test prediction exists. Task IDs and hidden test outputs are excluded from LLM task context.\n\n"
    if blocked:
        report += "## Blocker\n\nNo configured real model runtime/API is available. Capability comprehension, full-catalogue versus scheduler-hint, direct-grid, and structured-program conditions were intentionally not run. Their metrics are null rather than fabricated.\n"
    Path("reports/llm_hypothesis_generator_v1.md").write_text(report, encoding="utf-8")
    Path("reports/llm_capability_gap_candidates.md").write_text(
        "# LLM Capability Gap Candidates\n\nStatus: BLOCKED_NO_LLM_BACKEND. No model proposals were generated, so no capability gaps may be inferred. No capability was added.\n",
        encoding="utf-8",
    )
    Path("reports/data_exposure_audit.md").write_text(
        "# Data Exposure Audit\n\n- Capability catalogue derives only from the registered primitive metadata.\n- LLM task context contains train input/output pairs and an unlabeled test input; it excludes task IDs and test outputs.\n- This run made no LLM calls because no backend was available.\n- No public evaluation task was inspected for capability development.\n- No task-ID-specific branch or post-LLM capability expansion was used.\n",
        encoding="utf-8",
    )


def _update_log(payload: dict[str, object]) -> None:
    path = Path("experiments/experiments.csv")
    fields = ["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"]
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["experiment_id"] != EXPERIMENT_ID]
    rows.append({
        "experiment_id": EXPERIMENT_ID, "date": str(date.today()), "git_commit": "uncommitted",
        "solver": "LLM Hypothesis Generator V1", "representation": "frozen canonical capability catalogue + typed programs",
        "search_method": "BLOCKED_NO_LLM_BACKEND", "llm_model": "", "candidate_budget": 10,
        "validation_split": "not run; no configured real backend", "tasks_solved": 0, "accuracy": 0.0,
        "runtime_seconds": 0.0, "gpu_hours": 0, "notes": json.dumps({"status": payload["status"], "no_fabricated_llm_results": True}),
    })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    catalog = write_catalog()
    benchmark = build_comprehension_benchmark()
    Path("artifacts/llm_capability_comprehension_v1.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    payload = blocked_result(provider_audit(), registry_count=len(catalog["capabilities"]), comprehension_questions=benchmark["question_count"])
    payload.update({
        "frozen_deterministic_baseline": _frozen_baseline(),
        "registry_validation_errors": validate_registry(),
        "catalogue_artifacts": ["artifacts/llm_capability_catalog_v1.json", "artifacts/llm_capability_catalog_v1_prompt.txt"],
        "comprehension_artifact": "artifacts/llm_capability_comprehension_v1.json",
        "prompt_version": "llm_hypothesis_generator_v1.prompt.1",
        "models_used": [],
        "actual_llm_calls": 0,
        "runtime_seconds": 0.0,
    })
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/LLM_HYPOTHESIS_GENERATOR_V1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_reports(payload)
    _update_log(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
