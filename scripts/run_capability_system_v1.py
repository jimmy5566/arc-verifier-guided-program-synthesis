"""Frozen, train-only evaluation for ARC2 Capability System V1.

The script deliberately has two phases.  Every candidate is generated,
hard-verified on train pairs, and applied to all test inputs from the challenge
file before the training solution file is opened anywhere in this process.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from arc.io import discover_dataset_files, load_dataset
from capabilities.composer import ComposerStats, GenericProgramComposerV1
from capabilities.execution import CapabilityExecutor, Status
from capabilities.pipeline import CapabilityProgram, CapabilityProgramExecutor
from capabilities.scheduler import CapabilitySchedulerV1
from primitives.registry import REGISTRY, validate_registry
from solvers.capability_sanity import CountGenerationSolver, EnclosureFillSolver, PathSerializationSolver


EXPERIMENT_ID = "ARC2_CAPABILITY_SYSTEM_V1"
COMPOSER_FAMILIES = frozenset(("object", "relation", "pattern", "region", "counting", "generation", "graph_path"))
ALL_ABLATION_FAMILIES = ("object", "relation", "pattern", "region", "graph_path", "counting", "generation", "sequence", "iteration")


@dataclass
class FrozenRecord:
    program_ids: list[str]
    program_depths: list[int]
    selected_program_id: str | None
    selected_program_depth: int | None
    selected_primitive_ids: tuple[str, ...]
    selected_prediction: list[np.ndarray] | None
    train_consistent_programs: int
    program_ambiguous: bool
    prediction_ambiguous: bool
    test_execution_failed: bool
    schedule_primary: tuple[str, ...] = ()
    fallback_used: bool = False


def _merge_stats(left: ComposerStats, right: ComposerStats) -> ComposerStats:
    return ComposerStats(
        generated=left.generated + right.generated,
        type_pruned=left.type_pruned + right.type_pruned,
        precondition_pruned=left.precondition_pruned + right.precondition_pruned,
        train_pruned=left.train_pruned + right.train_pruned,
        deduplicated=left.deduplicated + right.deduplicated,
        verified=left.verified + right.verified,
        budget_exhausted=left.budget_exhausted or right.budget_exhausted,
    )


def _prediction_key(prediction: list[np.ndarray]) -> tuple[tuple[tuple[int, ...], str, bytes], ...]:
    return tuple((tuple(array.shape), str(array.dtype), array.tobytes()) for array in prediction)


def _freeze_programs(programs: Iterable[CapabilityProgram], test_inputs: Iterable[np.ndarray], *, primary: tuple[str, ...] = (), fallback_used: bool = False) -> FrozenRecord:
    executor = CapabilityProgramExecutor()
    ordered = sorted(programs, key=lambda program: program.program_id)
    generated: list[tuple[CapabilityProgram, list[np.ndarray]]] = []
    test_execution_failed = False
    inputs = tuple(test_inputs)
    for program in ordered:
        prediction: list[np.ndarray] = []
        for test_input in inputs:
            result = executor.execute(program, test_input)
            if result.status != Status.SUCCESS or not isinstance(result.value, np.ndarray):
                test_execution_failed = True
                prediction = []
                break
            prediction.append(result.value.copy())
        if prediction:
            generated.append((program, prediction))
    groups: dict[tuple[tuple[tuple[int, ...], str, bytes], ...], list[tuple[CapabilityProgram, list[np.ndarray]]]] = {}
    for program, prediction in generated:
        groups.setdefault(_prediction_key(prediction), []).append((program, prediction))
    selected = generated[0] if generated else None
    return FrozenRecord(
        program_ids=[program.program_id for program in ordered],
        program_depths=[program.depth for program in ordered],
        selected_program_id=selected[0].program_id if selected else None,
        selected_program_depth=selected[0].depth if selected else None,
        selected_primitive_ids=tuple(step.primitive_id for step in selected[0].steps) if selected else (),
        selected_prediction=selected[1] if selected else None,
        train_consistent_programs=len(ordered),
        program_ambiguous=len(ordered) > 1,
        prediction_ambiguous=len(groups) > 1,
        test_execution_failed=test_execution_failed,
        schedule_primary=primary,
        fallback_used=fallback_used,
    )


def _relevant_scheduler_families(schedule_families: Iterable[str], allowed: frozenset[str]) -> frozenset[str]:
    return frozenset(set(schedule_families) & set(allowed))


def _run_composer_variant(tasks, *, scheduled: bool, excluded_family: str | None = None, budget: int = 48):
    started = perf_counter()
    records: dict[str, FrozenRecord] = {}
    total = ComposerStats()
    fallback_tasks = 0
    scheduler = CapabilitySchedulerV1()
    allowed = COMPOSER_FAMILIES - ({excluded_family} if excluded_family in COMPOSER_FAMILIES else set())
    for task_id in sorted(tasks):
        task = tasks[task_id]
        fallback_used = False
        primary: tuple[str, ...] = ()
        if scheduled:
            schedule = scheduler.schedule(task, base_budget=budget, top_k=3)
            primary = schedule.primary
            stage_a = _relevant_scheduler_families(schedule.primary, frozenset(allowed))
            programs, stats = GenericProgramComposerV1((), max_depth=4, budget=budget).compose_task(task, families=stage_a)
            if not programs:
                fallback_used = True
                fallback_tasks += 1
                stage_b = frozenset(allowed) - stage_a
                fallback_programs, fallback_stats = GenericProgramComposerV1((), max_depth=4, budget=budget).compose_task(task, families=stage_b)
                programs.extend(fallback_programs)
                stats = _merge_stats(stats, fallback_stats)
        else:
            programs, stats = GenericProgramComposerV1((), max_depth=4, budget=budget).compose_task(task, families=allowed)
        total = _merge_stats(total, stats)
        records[task_id] = _freeze_programs(
            programs,
            (example.input.values for example in task.test),
            primary=primary,
            fallback_used=fallback_used,
        )
    return records, {
        "runtime_seconds": perf_counter() - started,
        "stats": asdict(total),
        "fallback_tasks": fallback_tasks,
        "allowed_families": sorted(allowed),
    }


def _freeze_sanity_solver(tasks, factory):
    started = perf_counter()
    records: dict[str, FrozenRecord] = {}
    for task_id in sorted(tasks):
        task = tasks[task_id]
        solver = factory().fit(task)
        programs = list(getattr(solver, "programs", ()))
        records[task_id] = _freeze_programs(programs, (example.input.values for example in task.test))
    return records, perf_counter() - started


def _score(records: dict[str, FrozenRecord], scored_tasks) -> dict[str, object]:
    solved: list[str] = []
    wrong: list[str] = []
    output_solved = 0
    output_total = 0
    depths: Counter[str] = Counter()
    for task_id in sorted(scored_tasks):
        task = scored_tasks[task_id]
        record = records[task_id]
        predictions = record.selected_prediction
        exact = bool(predictions) and len(predictions) == len(task.test) and all(
            np.array_equal(prediction, example.output.values)
            for prediction, example in zip(predictions or (), task.test)
        )
        if predictions and len(predictions) == len(task.test):
            output_solved += sum(
                np.array_equal(prediction, example.output.values)
                for prediction, example in zip(predictions, task.test)
            )
        output_total += len(task.test)
        if exact:
            solved.append(task_id)
            if record.selected_program_depth is not None:
                depths[str(record.selected_program_depth)] += 1
        elif record.train_consistent_programs:
            wrong.append(task_id)
    return {
        "exact_solved_task_ids": solved,
        "exact_solved_tasks": len(solved),
        "test_outputs_solved": output_solved,
        "test_outputs_total": output_total,
        "train_consistent_but_test_wrong_task_ids": wrong,
        "train_consistent_but_test_wrong": len(wrong),
        "depth_distribution": dict(sorted(depths.items())),
        "program_ambiguity_tasks": sum(record.program_ambiguous for record in records.values()),
        "prediction_ambiguity_tasks": sum(record.prediction_ambiguous for record in records.values()),
        "test_execution_failed_tasks": sum(record.test_execution_failed for record in records.values()),
    }


def _baseline_ids() -> tuple[set[str], dict[str, int]]:
    results = Path("experiments/results")
    v0 = json.loads((results / "SOLVER_LIB_V0.json").read_text(encoding="utf-8"))
    v1 = json.loads((results / "SOLVER_LIB_V1_OBJECT.json").read_text(encoding="utf-8"))
    v2 = json.loads((results / "SOLVER_LIB_V2_RELATION_COMPOSITION.json").read_text(encoding="utf-8"))
    v3 = json.loads((results / "SOLVER_LIB_V3_PATTERN.json").read_text(encoding="utf-8"))
    base = (
        set(v0["unique_solved_task_ids"])
        | set(v1["newly_solved_task_ids"])
        | set(v2["newly_solved_task_ids"])
        | set(v3["newly_solved_task_ids"])
    )
    regression = {
        "v0_unique": len(v0["unique_solved_task_ids"]),
        "v0_v1_cumulative": len(set(v0["unique_solved_task_ids"]) | set(v1["newly_solved_task_ids"])),
        "v0_v1_v2_cumulative": len(set(v0["unique_solved_task_ids"]) | set(v1["newly_solved_task_ids"]) | set(v2["newly_solved_task_ids"])),
        "v0_v1_v2_v3_cumulative": len(base),
    }
    regression["passed"] = regression == {
        "v0_unique": 14,
        "v0_v1_cumulative": 18,
        "v0_v1_v2_cumulative": 18,
        "v0_v1_v2_v3_cumulative": 20,
    }
    return base, regression


def _program_families(record: FrozenRecord) -> set[str]:
    families: set[str] = set()
    for primitive_id in record.selected_primitive_ids:
        if primitive_id.startswith(("SEL_", "OBJ_")):
            families.add("object")
        elif primitive_id.startswith(("REL_", "DIST_", "ALIGN_", "COMP_", "COPY_")):
            families.add("relation")
        elif primitive_id.startswith("PAT_"):
            families.add("pattern")
        elif primitive_id.startswith("REG_"):
            families.add("region")
        elif primitive_id.startswith("COUNT_"):
            families.add("counting")
        elif primitive_id.startswith("GEN_"):
            families.add("generation")
        elif primitive_id.startswith("GRAPH_"):
            families.add("graph_path")
        elif primitive_id.startswith("SEQ_"):
            families.add("sequence")
        elif primitive_id.startswith("ITERATE_") or primitive_id.startswith("LINE_EXTEND_"):
            families.add("iteration")
    return families


def _split_metrics(task_ids: Iterable[str], split_by_task: dict[str, str]) -> dict[str, int]:
    ids = set(task_ids)
    return {split: sum(split_by_task.get(task_id) == split for task_id in ids) for split in ("development", "held_out", "challenge_like")}


def _update_experiment_log(payload: dict[str, object], task_count: int) -> None:
    log = Path("experiments/experiments.csv")
    fields = [
        "experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model",
        "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes",
    ]
    rows = []
    if log.exists():
        with log.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["experiment_id"] != EXPERIMENT_ID]
    row = {
        "experiment_id": EXPERIMENT_ID,
        "date": str(date.today()),
        "git_commit": "uncommitted",
        "solver": "Capability System V1",
        "representation": "typed V4 capabilities plus delegated bounded V2/V3 structured programs",
        "search_method": "deterministic bounded typed composition + fallback scheduler",
        "llm_model": "",
        "candidate_budget": 48,
        "validation_split": "official_training_challenges; solutions only after prediction freeze",
        "tasks_solved": payload["cumulative_solved"],
        "accuracy": payload["cumulative_solved"] / task_count,
        "runtime_seconds": payload["runtime_seconds"],
        "gpu_hours": 0,
        "notes": json.dumps({"newly_solved": payload["newly_solved_task_ids"], "composition_only": payload["composition_only_task_ids"]}),
    }
    with log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(row)


def main() -> None:
    started = perf_counter()
    files = discover_dataset_files("data/raw")

    # INFERENCE FREEZE: this load contains train pairs and unlabeled test inputs only.
    tasks = load_dataset(files["training_challenges"])
    unscheduled_records, unscheduled_diag = _run_composer_variant(tasks, scheduled=False)
    scheduled_records, scheduled_diag = _run_composer_variant(tasks, scheduled=True)
    sanity_records: dict[str, dict[str, FrozenRecord]] = {}
    sanity_runtime: dict[str, float] = {}
    for name, factory in (
        ("EnclosureFillSolver", EnclosureFillSolver),
        ("PathSerializationSolver", PathSerializationSolver),
        ("CountGenerationSolver", CountGenerationSolver),
    ):
        sanity_records[name], sanity_runtime[name] = _freeze_sanity_solver(tasks, factory)
    ablation_records: dict[str, tuple[dict[str, FrozenRecord], dict[str, object]]] = {}
    for family in ALL_ABLATION_FAMILIES:
        ablation_records[family] = _run_composer_variant(tasks, scheduled=False, excluded_family=family)

    # SCORING: solution access begins only after every prediction above is frozen.
    scored_tasks = load_dataset(files["training_challenges"], files["training_solutions"])
    unscheduled = _score(unscheduled_records, scored_tasks)
    scheduled = _score(scheduled_records, scored_tasks)
    sanity = {name: _score(records, scored_tasks) for name, records in sanity_records.items()}
    ablations = {
        family: {**_score(records, scored_tasks), "runtime_seconds": diagnostics["runtime_seconds"]}
        for family, (records, diagnostics) in ablation_records.items()
    }

    base_ids, frozen_regression = _baseline_ids()
    generic_ids = set(scheduled["exact_solved_task_ids"])
    sanity_ids = set().union(*(set(metrics["exact_solved_task_ids"]) for metrics in sanity.values()))
    v4_ids = generic_ids | sanity_ids
    new_ids = v4_ids - base_ids
    composition_only = {
        task_id
        for task_id in generic_ids - base_ids - sanity_ids
        if scheduled_records[task_id].selected_program_depth and scheduled_records[task_id].selected_program_depth >= 2
    }
    primitive_usage = Counter()
    for task_id in generic_ids:
        primitive_usage.update(scheduled_records[task_id].selected_primitive_ids)

    split_by_task: dict[str, str] = {}
    split_path = Path("data/splits/task_splits.csv")
    if split_path.exists():
        with split_path.open(newline="", encoding="utf-8") as handle:
            split_by_task = {row["task_id"]: row["split"] for row in csv.DictReader(handle)}
    solved_scheduler_families = [
        (task_id, _program_families(scheduled_records[task_id]))
        for task_id in generic_ids
    ]
    scheduler_recall = {
        f"top_{k}_recall": (
            sum(required <= set(scheduled_records[task_id].schedule_primary[:k]) for task_id, required in solved_scheduler_families if required)
            / sum(bool(required) for _task_id, required in solved_scheduler_families)
            if any(required for _task_id, required in solved_scheduler_families)
            else None
        )
        for k in (1, 3, 5)
    }
    executor_audit = CapabilityExecutor.dispatch_audit()
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "protocol": {
            "inference": "training challenges only: train pairs -> compose -> hard verify -> test prediction freeze",
            "scoring": "training solutions loaded only after all variants and sanity predictions were frozen",
            "task_id_hardcode": False,
        },
        "runtime_seconds": perf_counter() - started,
        "task_count": len(tasks),
        "frozen_regression": frozen_regression,
        "registry_validation_errors": validate_registry(),
        "executor_dispatch_audit": executor_audit,
        "sanity_solvers": {name: {**metrics, "runtime_seconds": sanity_runtime[name]} for name, metrics in sanity.items()},
        "generic_composer": {
            "unscheduled": {**unscheduled, **unscheduled_diag},
            "scheduled": {**scheduled, **scheduled_diag},
        },
        "v4_unique_solved_task_ids": sorted(v4_ids),
        "v4_unique_solved": len(v4_ids),
        "overlap_existing_20": sorted(v4_ids & base_ids),
        "newly_solved_task_ids": sorted(new_ids),
        "newly_solved": len(new_ids),
        "cumulative_solved": len(base_ids | v4_ids),
        "standalone_newly_solved": len((sanity_ids - base_ids)),
        "composition_only_task_ids": sorted(composition_only),
        "composition_only_newly_solved": len(composition_only),
        "scheduler": {
            **scheduler_recall,
            "fallback_enabled": True,
            "fallback_tasks": scheduled_diag["fallback_tasks"],
            "candidate_reduction_vs_unscheduled": unscheduled_diag["stats"]["generated"] - scheduled_diag["stats"]["generated"],
            "runtime_delta_seconds": unscheduled_diag["runtime_seconds"] - scheduled_diag["runtime_seconds"],
        },
        "family_ablation": {
            family: {
                "exact_solved": result["exact_solved_tasks"],
                "delta_vs_unscheduled": result["exact_solved_tasks"] - unscheduled["exact_solved_tasks"],
                "runtime_seconds": result["runtime_seconds"],
            }
            for family, result in ablations.items()
        },
        "split_metrics": {
            split: {
                "tasks": sum(value == split for value in split_by_task.values()),
                "scheduled_generic_exact": _split_metrics(scheduled["exact_solved_task_ids"], split_by_task)[split],
                "v4_unique_exact": _split_metrics(v4_ids, split_by_task)[split],
            }
            for split in ("development", "held_out", "challenge_like")
        },
        "primitive_usage_on_generic_exact": dict(primitive_usage),
        "leakage_hardcode_audit": [
            "All inference variants loaded training challenges without a solution path.",
            "All test predictions were materialized before the first solution-file load.",
            "No task ID is used in candidate generation, dispatch, scheduler, or scoring selection.",
            "Hard verification compares candidate outputs only to train outputs.",
        ],
        "known_scope_limits": [
            "The generic grammar reuses bounded object/relation programs and instantiates region-fill, count-to-N-cell-generation, graph-path serialization, and registered pattern-completion templates.",
            "No LLM, router, learned ranker, or unrestricted Cartesian program search is used.",
        ],
    }
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("experiments/results/ARC2_CAPABILITY_SYSTEM_V1.json").write_text(json.dumps(payload, indent=2, default=list), encoding="utf-8")
    report = "# ARC2 Capability System V1\n\n"
    report += "## Capability architecture\n\nThe system uses a typed, versioned primitive registry. `CapabilityExecutor` resolves an ID to one deterministic implementation and returns explicit SUCCESS, INVALID, AMBIGUOUS, or EXECUTION_FAILED states. The active V4 library covers region/topology, separators and line connection, graph/path traversal and serialization, counting, generation, sequence conversion, and bounded iteration.\n\n"
    report += "## Primitive typing and composition\n\n`CapabilityProgramExecutor` preserves the latest grid canvas separately from typed intermediate values, so perception can feed parameter provenance without treating a RegionSet, Integer, Path, or Sequence as a grid. The bounded generic grammar currently instantiates Grid→Region→Fill, Grid→Count→Generate, and Grid→Graph→Path→Sequence→Grid templates.\n\n"
    report += "## Parameter inference and hard verification\n\nAll colors, path endpoints, count sources, output geometry, and serializer orientation are intersected across train pairs. Dynamic `{$from: ...}` references recompute typed facts on each test input. Every complete candidate must exactly reproduce every train output; test solutions are never consulted to choose a hypothesis.\n\n"
    report += "## Scheduler and budget\n\nThe deterministic scheduler ranks multiple families and preserves fallback search. Each graph-path template caps endpoint-pair enumeration with a shared deterministic budget; no unconstrained Cartesian product is used. The comparison below reports the observed scheduler/unscheduled trade-off rather than assuming a runtime win.\n\n"
    report += "## Coverage, ablation, and generalization\n\nThe JSON results include standalone versus composition-only coverage, depth, ambiguity, train-consistent-but-test-wrong counts, family-removal ablations, and fixed development/held-out/challenge-like split metrics. Zero-contribution families remain reported as zero rather than being silently excluded.\n\n"
    report += "## Leakage controls\n\nInference loads only the challenge collection. All composer variants, sanity solvers, and ablations freeze test predictions before the first solution-file load. No task-ID branch, public-evaluation-specific rule, LLM, router, or learned ranker is present.\n\n"
    report += "## Remaining gaps\n\nThe scheduler preserves coverage and reduces the measured candidate count, but its primary-stage recall remains limited because many valid hypotheses are recovered by fallback. Existing bounded object/relation and pattern programs are delegated through the unified verifier; the remaining gap is richer fully typed V4 multi-step templates that can produce composition-only gains beyond those frozen families.\n\n"
    report += "## Results\n\n```json\n" + json.dumps(payload, indent=2, default=list) + "\n```\n"
    Path("reports/arc2_capability_system_v1.md").write_text(report, encoding="utf-8")
    _update_experiment_log(payload, len(tasks))
    print(json.dumps(payload, indent=2, default=list))


if __name__ == "__main__":
    main()
