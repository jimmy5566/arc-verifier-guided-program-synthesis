"""CPU-only provenance audit for existing native ARC candidate artifacts.

This script deliberately analyzes only already-frozen candidates.  It never
loads a model and opens training solutions only after the candidate artifacts
have been read in full.  Evaluation60 has task-level scoring but no persisted
per-candidate target labels, so the script reports that limitation rather than
inventing representation-level correctness for that cohort.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GEOMETRIES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_transpose",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _is_exact(prediction: Any, solution: Any) -> bool:
    return prediction == solution


def _as_augmentations(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    support = candidate.get("support_augmentations")
    if isinstance(support, list) and support:
        return [item for item in support if isinstance(item, dict)]
    augmentation = candidate.get("augmentation")
    return [augmentation] if isinstance(augmentation, dict) else []


def _empty_view_stats() -> dict[str, Any]:
    return {
        "raw_support_events": 0,
        "unique_candidate_contributions": 0,
        "correct_support_events": 0,
        "correct_unique_candidate_contributions": 0,
        "tasks_with_correct_candidate": 0,
        "task_anyk": 0,
    }


def analyze_training(
    artifact: dict[str, Any], solutions: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    """Compute exact representation hits from training-side frozen candidates."""
    view_stats = {geometry: _empty_view_stats() for geometry in GEOMETRIES}
    order_stats: Counter[str] = Counter()
    color_stats: Counter[str] = Counter()
    correct_by_view: dict[str, set[str]] = {geometry: set() for geometry in GEOMETRIES}
    task_oracle: set[str] = set()
    task_count = 0
    invalid_total = 0
    generated_total = 0

    records = artifact["records"]
    for task_id, record in records.items():
        task_count += 1
        generated_total += int(record.get("generated_candidate_count", 0))
        invalid_total += int(record.get("invalid_candidate_count", 0))
        solution = solutions.get(task_id)
        if solution is None:
            raise KeyError(f"training solution missing frozen task {task_id}")
        candidate_views_with_hit: set[str] = set()
        for candidate in record.get("candidates", []):
            augmentations = _as_augmentations(candidate)
            exact = _is_exact(candidate.get("prediction"), solution)
            for augmentation in augmentations:
                geometry = augmentation.get("geometry")
                if geometry in view_stats:
                    view_stats[geometry]["raw_support_events"] += 1
                    view_stats[geometry]["unique_candidate_contributions"] += 1
                    if exact:
                        view_stats[geometry]["correct_support_events"] += 1
                        view_stats[geometry]["correct_unique_candidate_contributions"] += 1
                        candidate_views_with_hit.add(geometry)
                order_stats[str(augmentation.get("pair_order", "UNAVAILABLE"))] += 1
                color_stats[str(augmentation.get("color_offset", "UNAVAILABLE"))] += 1
        for geometry in candidate_views_with_hit:
            view_stats[geometry]["tasks_with_correct_candidate"] += 1
            correct_by_view[geometry].add(task_id)
        if candidate_views_with_hit:
            task_oracle.add(task_id)

    for geometry, stats in view_stats.items():
        stats["task_anyk"] = len(correct_by_view[geometry])
        events = stats["raw_support_events"]
        stats["correct_generation_rate"] = (
            stats["correct_support_events"] / events if events else None
        )
        stats["invalid_rate"] = "UNAVAILABLE_PER_VIEW"
        stats["runtime_contribution"] = "UNAVAILABLE_PER_VIEW"

    best_view = max(
        GEOMETRIES,
        key=lambda geometry: (len(correct_by_view[geometry]), -GEOMETRIES.index(geometry)),
    )
    return (
        {
            "task_count": task_count,
            "task_anyk": len(task_oracle),
            "generated_candidate_count": generated_total,
            "invalid_candidate_count": invalid_total,
            "pair_order_support_events": dict(sorted(order_stats.items())),
            "color_offset_support_events": dict(sorted(color_stats.items())),
            "views": view_stats,
            "best_global_view": best_view,
            "best_global_view_anyk": len(correct_by_view[best_view]),
            "oracle_view_anyk": len(task_oracle),
            "routing_headroom": len(task_oracle) - len(correct_by_view[best_view]),
        },
        correct_by_view,
    )


def analyze_evaluation_artifacts(
    candidates: dict[str, Any], scored_csv: Path
) -> dict[str, Any]:
    """Summarize view provenance without reopening unavailable per-candidate labels."""
    task_status: dict[str, dict[str, str]] = {}
    with scored_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task_status[row["task_id"]] = row

    view_support: Counter[str] = Counter()
    view_unique: Counter[str] = Counter()
    task_count_by_view: Counter[str] = Counter()
    invalid = 0
    total = 0
    for task_id, record in candidates["records"].items():
        total += 1
        invalid += int(record.get("invalid_candidate_count", 0))
        seen: set[str] = set()
        for candidate in record.get("candidates", []):
            for augmentation in _as_augmentations(candidate):
                geometry = augmentation.get("geometry")
                if geometry in GEOMETRIES:
                    view_support[geometry] += 1
                    view_unique[geometry] += 1
                    seen.add(geometry)
        for geometry in seen:
            task_count_by_view[geometry] += 1

    scored_anyk = sum(row["any_of_k"].lower() == "true" for row in task_status.values())
    return {
        "task_count": total,
        "task_anyk_from_frozen_scoring": scored_anyk,
        "invalid_candidate_count": invalid,
        "view_support_events": {view: view_support[view] for view in GEOMETRIES},
        "view_unique_candidate_contributions": {view: view_unique[view] for view in GEOMETRIES},
        "tasks_represented_by_view": {view: task_count_by_view[view] for view in GEOMETRIES},
        "per_candidate_correct_view_labels": "UNAVAILABLE_IN_EXISTING_ARTIFACTS",
        "preference_stability": "UNAVAILABLE: Evaluation60 retains task-level Any-of-K labels, not candidate-to-view exact labels.",
    }


def build_report(
    training: dict[str, Any], evaluation: dict[str, Any], paths: dict[str, str]
) -> dict[str, Any]:
    return {
        "experiment_id": "ARC2_REPRESENTATION_PREFERENCE_ROUTING_FEASIBILITY",
        "status": "COMPLETE_CPU_ONLY_ARTIFACT_ANALYSIS",
        "protocol": (
            "All candidate artifacts were already frozen. Training solutions were read only "
            "after the training artifact was loaded. No model/CUDA/inference was used."
        ),
        "artifact_paths": paths,
        "training_frozen60": training,
        "evaluation60": evaluation,
        "best_global_view": training["best_global_view"],
        "best_global_view_anyk": training["best_global_view_anyk"],
        "oracle_view_anyk": training["oracle_view_anyk"],
        "routing_headroom": training["routing_headroom"],
        "view_preference_stability": evaluation["preference_stability"],
        "router_feasibility": "INSUFFICIENT_DATA_FOR_ROUTER",
        "router_feasibility_rationale": [
            "Training Frozen60 candidate provenance permits retrospective view-level exactness.",
            "Evaluation60 persists only task-level Any-of-K after scoring, so it cannot test view-level preference stability.",
            "The available cohorts therefore cannot support a target-blind routing rule without a new, explicitly scoped provenance-preserving experiment.",
        ],
        "limitations": [
            "A deduplicated candidate can be supported by multiple views; view contribution is therefore not exclusive.",
            "Per-view invalid-rate and runtime attribution are not persisted in the frozen artifacts.",
            "No router was fitted, and no evaluation target was used as a feature.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    train = report["training_frozen60"]
    lines = [
        "# Representation Preference / Routing Feasibility",
        "",
        "CPU-only analysis of already-frozen candidate artifacts. No model was loaded.",
        "",
        f"- BEST_GLOBAL_VIEW: `{report['best_global_view']}`",
        f"- BEST_GLOBAL_VIEW_ANYK: `{report['best_global_view_anyk']}/{train['task_count']}`",
        f"- ORACLE_VIEW_ANYK: `{report['oracle_view_anyk']}/{train['task_count']}`",
        f"- ROUTING_HEADROOM: `{report['routing_headroom']}` tasks",
        f"- ROUTER_FEASIBILITY: `{report['router_feasibility']}`",
        "",
        "## Training Frozen60 view evidence",
        "",
        "| View | Support events | Correct support events | Tasks with an exact candidate | Correct-event rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for view in GEOMETRIES:
        stats = train["views"][view]
        rate = stats["correct_generation_rate"]
        formatted_rate = "N/A" if rate is None else f"{rate:.4f}"
        lines.append(
            f"| {view} | {stats['raw_support_events']} | "
            f"{stats['correct_support_events']} | {stats['tasks_with_correct_candidate']} | {formatted_rate} |"
        )
    lines += [
        "",
        "## Stability conclusion",
        "",
        report["view_preference_stability"],
        "",
        "Because Evaluation60 lacks persisted candidate-to-view target labels, a target-blind router cannot be justified from this evidence alone.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-candidates", type=Path, required=True)
    parser.add_argument("--training-solutions", type=Path, required=True)
    parser.add_argument("--evaluation-candidates", type=Path, required=True)
    parser.add_argument("--evaluation-results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_artifact = _load_json(args.training_candidates)
    eval_artifact = _load_json(args.evaluation_candidates)
    solutions = _load_json(args.training_solutions)
    training, _ = analyze_training(train_artifact, solutions)
    evaluation = analyze_evaluation_artifacts(eval_artifact, args.evaluation_results_csv)
    report = build_report(
        training,
        evaluation,
        {
            "training_candidates": str(args.training_candidates),
            "training_solutions": str(args.training_solutions),
            "evaluation_candidates": str(args.evaluation_candidates),
            "evaluation_results_csv": str(args.evaluation_results_csv),
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "REPRESENTATION_PREFERENCE_REPORT.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.output_dir / "REPRESENTATION_PREFERENCE_REPORT.md")
    print(json.dumps({
        "event": "REPRESENTATION_PREFERENCE_COMPLETE",
        "best_global_view": report["best_global_view"],
        "best_global_view_anyk": report["best_global_view_anyk"],
        "oracle_view_anyk": report["oracle_view_anyk"],
        "router_feasibility": report["router_feasibility"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
