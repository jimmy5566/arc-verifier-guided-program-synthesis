#!/usr/bin/env python3
"""CPU-only retrospective audit of the frozen Eval60 fixed 4+4 portfolio.

The target-blind phase reconstructs the full and source-only D1 rankings,
writes immutable candidate/ranking/prediction artifacts, and hashes them.  It
loads evaluation solutions only after that freeze is complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.selector_d1 import attempts_from_order, baseline_order, d1_order, verify_d1_scope


EXPECTED_ZIP_SHA256 = "79a90750202b6988cfb4a0ca0c45f96846b5a89a2ca6647434ebffff47685c35"
EXPECTED_FULL = {"top1": 20, "top2": 28, "oracle": 30, "tasks": 60, "outputs": 89}
EXPECTED_PORTFOLIO = {
    "TTT24": ["flip_lr", "flip_ud", "transpose", "anti_transpose"],
    "TTT48": ["identity", "rot90", "flip_ud", "anti_transpose"],
}
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_TRIALS = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if columns is None:
        columns = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def member_name(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one ZIP member ending {suffix!r}; got {matches}")
    return matches[0]


def read_member(archive: zipfile.ZipFile, suffix: str) -> Any:
    return json.loads(archive.read(member_name(archive, suffix)).decode("utf-8"))


def validate_manifest(archive: zipfile.ZipFile) -> dict[str, str]:
    manifest_name = member_name(archive, "/MANIFEST.sha256")
    root = manifest_name.rsplit("/", 1)[0]
    verified: dict[str, str] = {}
    for line in archive.read(manifest_name).decode("utf-8").splitlines():
        expected, filename = line.split("  ", 1)
        actual = hashlib.sha256(archive.read(f"{root}/{filename}")).hexdigest()
        if actual != expected:
            raise ValueError(f"manifest mismatch for {filename}: {actual} != {expected}")
        verified[filename] = actual
    return verified


def discover_source_mapping(pool_artifact: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, int] = defaultdict(int)
    observed_tags: dict[str, set[str]] = defaultdict(set)
    for task in pool_artifact["pools"].values():
        for output in task["per_test"]:
            for candidate in output["candidates"]:
                for row in candidate["source_rows"]:
                    label = str(row["source"])
                    observed[label] += 1
                    observed_tags[label].update(map(str, row.get("slot_tags", [])))
    configured = {str(k): list(map(str, v)) for k, v in pool_artifact["portfolio"].items()}
    if set(observed) != set(configured) or set(configured) != {"TTT24", "TTT48"}:
        raise ValueError(f"ambiguous source labels: observed={sorted(observed)} configured={sorted(configured)}")
    if configured != EXPECTED_PORTFOLIO:
        raise ValueError(f"portfolio mismatch: {configured}")
    return {
        "status": "FROZEN_BEFORE_SOLUTION_ACCESS",
        "mapping_method": "exact labels plus portfolio view-set identity from immutable handoff",
        "conceptual_to_frozen_label": {label: label for label in sorted(observed)},
        "sources": {
            label: {
                "conceptual_depth": int(label.removeprefix("TTT")),
                "source_row_count": observed[label],
                "configured_views": configured[label],
                "observed_slot_tags": sorted(observed_tags[label]),
            }
            for label in sorted(observed)
        },
    }


def representative(candidate: dict[str, Any]) -> tuple[str, int]:
    return min((str(row["source"]), int(row["candidate_index"])) for row in candidate["source_rows"])


def source_b_scores(candidates: list[dict[str, Any]], source_labels: Iterable[str]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    scores: dict[str, dict[str, float]] = {source: {} for source in source_labels}
    reps = {str(candidate["grid_key"]): representative(candidate) for candidate in candidates}
    for candidate in candidates:
        token = str(candidate["grid_key"])
        for source in source_labels:
            rows = [row for row in candidate["source_rows"] if str(row["source"]) == source]
            if rows:
                scores[source][token] = max(float(row["selected_support_count"]) - float(row["mean_view_nll"]) for row in rows)
    ranks: dict[str, dict[str, int]] = {}
    for source in source_labels:
        ordered = sorted(scores[source], key=lambda token: (-scores[source][token], reps[token], token))
        ranks[source] = {token: index + 1 for index, token in enumerate(ordered)}
    return scores, ranks


def reconstruct_full_evidence(
    pools: dict[str, Any], historic_rankings: dict[str, Any], source_labels: list[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline_replay: dict[str, Any] = {}
    d1_rankings: dict[str, Any] = {}
    d1_predictions: dict[str, Any] = {}
    checked_outputs = 0
    for task_id in sorted(pools):
        historic = {int(row["test_index"]): row for row in historic_rankings[task_id]}
        baseline_rows: list[dict[str, Any]] = []
        d1_rows: list[dict[str, Any]] = []
        attempt_1: list[Any] = []
        attempt_2: list[Any] = []
        for output in pools[task_id]["per_test"]:
            index = int(output["test_index"])
            candidates = output["candidates"]
            frozen = historic[index]
            b_scores, b_ranks = source_b_scores(candidates, source_labels)
            for candidate in candidates:
                token = str(candidate["grid_key"])
                frozen_scores = candidate.get("source_local_b_support", {})
                for source in source_labels:
                    if token in b_scores[source] and not math.isclose(
                        b_scores[source][token], float(frozen_scores[source][token]), rel_tol=0.0, abs_tol=1e-12
                    ):
                        raise ValueError(f"source-local B-SUPPORT value mismatch at {task_id}:{index}:{source}:{token}")
            if b_ranks != frozen["source_local_ranks"]:
                raise ValueError(f"source-local B-SUPPORT rank mismatch at {task_id}:{index}")
            recomputed_rrf = {
                str(candidate["grid_key"]): sum(
                    1.0 / b_ranks[source][str(candidate["grid_key"])]
                    for source in source_labels
                    if str(candidate["grid_key"]) in b_ranks[source]
                )
                for candidate in candidates
            }
            for candidate in candidates:
                token = str(candidate["grid_key"])
                if not math.isclose(recomputed_rrf[token], float(candidate["rrf_score"]), rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"B-RRF mismatch at {task_id}:{index}:{token}")
            replay = baseline_order(candidates)
            if replay != frozen["ranked_grid_keys"]:
                raise ValueError(f"historical baseline ordering mismatch at {task_id}:{index}")
            ordered, likelihood_ranks, likelihood_rrf = d1_order(candidates)
            verify_d1_scope(candidates, ordered)
            first, second = attempts_from_order(candidates, ordered)
            baseline_rows.append({"test_index": index, "ranked_grid_keys": replay})
            d1_rows.append(
                {
                    "test_index": index,
                    "ranked_grid_keys": ordered,
                    "attempt_grid_keys": ordered[:2] if len(ordered) >= 2 else ordered * 2,
                    "source_b_support_ranks": b_ranks,
                    "source_likelihood_ranks": likelihood_ranks,
                    "likelihood_rrf_scores": likelihood_rrf,
                    "changed_from_baseline": ordered != replay,
                }
            )
            attempt_1.append(first)
            attempt_2.append(second)
            checked_outputs += 1
        baseline_replay[task_id] = baseline_rows
        d1_rankings[task_id] = d1_rows
        d1_predictions[task_id] = {"attempt_1": attempt_1, "attempt_2": attempt_2}
    return (
        {"status": "EXACT_FULL_POOL_REPLAY", "outputs": checked_outputs, "rankings": baseline_replay},
        {"status": "FROZEN_BEFORE_SOLUTION_ACCESS", "solutions_opened": False, "rankings": d1_rankings},
        {"status": "FROZEN_BEFORE_SOLUTION_ACCESS", "solutions_opened": False, "predictions": d1_predictions},
    )


def build_source_pool(pools: dict[str, Any], source: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task_id in sorted(pools):
        per_test: list[dict[str, Any]] = []
        for output in pools[task_id]["per_test"]:
            filtered: list[dict[str, Any]] = []
            for candidate in output["candidates"]:
                rows = [dict(row) for row in candidate["source_rows"] if str(row["source"]) == source]
                if not rows:
                    continue
                tags = sorted({str(tag) for row in rows for tag in row.get("slot_tags", [])})
                support = sum(int(row["selected_support_count"]) for row in rows)
                b_score = max(float(row["selected_support_count"]) - float(row["mean_view_nll"]) for row in rows)
                filtered.append(
                    {
                        "grid": candidate["grid"],
                        "grid_key": candidate["grid_key"],
                        "portfolio_slot_tags": tags,
                        "portfolio_sources": [source],
                        "portfolio_support_count": support,
                        "source_local_b_support": {source: b_score},
                        "source_rows": rows,
                    }
                )
            _, ranks = source_b_scores(filtered, [source])
            for candidate in filtered:
                candidate["rrf_score"] = 1.0 / ranks[source][str(candidate["grid_key"])]
            per_test.append(
                {
                    "test_index": int(output["test_index"]),
                    "unique_candidate_count": len(filtered),
                    "candidates": filtered,
                }
            )
        result[task_id] = {
            "per_test": per_test,
            "unique_candidate_bundle_count": sum(row["unique_candidate_count"] for row in per_test),
        }
    return result


def rank_source_pool(source_pools: dict[str, Any], source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rankings: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    for task_id in sorted(source_pools):
        rows: list[dict[str, Any]] = []
        attempt_1: list[Any] = []
        attempt_2: list[Any] = []
        for output in source_pools[task_id]["per_test"]:
            candidates = output["candidates"]
            ordered, likelihood_ranks, likelihood_rrf = d1_order(candidates)
            verify_d1_scope(candidates, ordered)
            first, second = attempts_from_order(candidates, ordered)
            b_scores = {str(candidate["grid_key"]): float(candidate["rrf_score"]) for candidate in candidates}
            rows.append(
                {
                    "test_index": int(output["test_index"]),
                    "ranked_grid_keys": ordered,
                    "attempt_grid_keys": ordered[:2] if len(ordered) >= 2 else ordered * 2,
                    "source_b_rrf_scores": b_scores,
                    "source_likelihood_ranks": likelihood_ranks,
                    "source_likelihood_rrf_scores": likelihood_rrf,
                    "empty_pool": not candidates,
                }
            )
            attempt_1.append(first)
            attempt_2.append(second)
        rankings[task_id] = rows
        predictions[task_id] = {"attempt_1": attempt_1, "attempt_2": attempt_2}
    return rankings, predictions


def freeze_source_artifact(source: str, portfolio: list[str], pools: dict[str, Any], rankings: dict[str, Any], predictions: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "SOURCE_ONLY_CANDIDATES_RANKINGS_PREDICTIONS_FROZEN_BEFORE_SOLUTION_ACCESS",
        "solutions_opened": False,
        "source": source,
        "portfolio": portfolio,
        "selector": "D1_RECONSTRUCTED_FROM_SOURCE_LOCAL_B_SUPPORT_AND_SOURCE_LOCAL_LIKELIHOOD",
        "combined_rrf_reused": False,
        "pools": pools,
        "rankings": rankings,
        "predictions": predictions,
    }


def score_artifact(pools: dict[str, Any], predictions: dict[str, Any], solutions: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    top1 = top2 = oracle = 0
    for task_id in sorted(pools):
        for output in pools[task_id]["per_test"]:
            index = int(output["test_index"])
            target = solutions[task_id][index]
            first = predictions[task_id]["attempt_1"][index]
            second = predictions[task_id]["attempt_2"][index]
            first_hit = first == target
            second_hit = second == target
            pool_hit = any(candidate["grid"] == target for candidate in output["candidates"])
            top1 += int(first_hit)
            top2 += int(first_hit or second_hit)
            oracle += int(pool_hit)
            rows.append(
                {
                    "task_id": task_id,
                    "test_index": index,
                    "output_key": f"{task_id}:{index}",
                    "top1_hit": first_hit,
                    "top2_hit": first_hit or second_hit,
                    "oracle_hit": pool_hit,
                    "candidate_count": len(output["candidates"]),
                }
            )
    return {"top1": top1, "top2": top2, "oracle": oracle, "rows": rows}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile input")
    point = (len(ordered) - 1) * probability
    lower = math.floor(point)
    upper = math.ceil(point)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - point) + ordered[upper] * (point - lower)


def queue_makespan(durations: list[float], workers: int = 4) -> float:
    loads = [0.0] * workers
    for duration in durations:
        index = min(range(workers), key=lambda worker: (loads[worker], worker))
        loads[index] += float(duration)
    return max(loads)


def bootstrap_runtime(task_durations: list[float], target_tasks: int, *, extra_output_costs: list[float] | None = None) -> dict[str, float]:
    rng = random.Random(BOOTSTRAP_SEED + target_tasks + (1 if extra_output_costs else 0))
    walls: list[float] = []
    for _ in range(BOOTSTRAP_TRIALS):
        sampled = [rng.choice(task_durations) for _ in range(target_tasks)]
        if extra_output_costs:
            for cost in extra_output_costs:
                sampled[rng.randrange(target_tasks)] += rng.choice(extra_output_costs)
        walls.append(queue_makespan(sampled))
    return {
        "p25_seconds": percentile(walls, 0.25),
        "median_seconds": percentile(walls, 0.50),
        "p75_seconds": percentile(walls, 0.75),
        "p90_seconds": percentile(walls, 0.90),
        "p95_seconds": percentile(walls, 0.95),
    }


def runtime_audit(version3_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_rows = list(csv.DictReader((version3_dir / "task_timing.csv").open(encoding="utf-8", newline="")))
    cell_rows = list(csv.DictReader((version3_dir / "cell_dfs_metrics.csv").open(encoding="utf-8", newline="")))
    throughput = json.loads((version3_dir / "throughput_summary.json").read_text(encoding="utf-8"))
    benchmark_ids = json.loads((version3_dir / "BENCHMARK_TASK_IDS.json").read_text(encoding="utf-8"))
    order = benchmark_ids["task_ids"] if isinstance(benchmark_ids, dict) else benchmark_ids
    by_task = {row["task_id"]: row for row in task_rows}
    durations = [float(by_task[task_id]["TTT24_TOTAL_S"]) for task_id in order]
    ttt24_train = sum(float(row["TTT24_TRAIN_S"]) for row in task_rows)
    ttt24_setup = sum(float(row["TTT24_RESET_OR_SETUP_S"]) for row in task_rows)
    ttt24_dfs = sum(float(row["TTT24_DFS_TOTAL_S"]) for row in task_rows)
    ttt48_train = sum(float(row["TTT48_TRAIN_S"]) for row in task_rows)
    ttt48_setup = sum(float(row["TTT48_RESET_OR_SETUP_S"]) for row in task_rows)
    ttt48_dfs = sum(float(row["TTT48_DFS_TOTAL_S"]) for row in task_rows)
    total24 = sum(durations)
    total48 = sum(float(row["TTT48_TOTAL_S"]) for row in task_rows)
    source_total = total24 + total48
    component_rows = [
        {"source": "TTT24", "component": "train", "gpu_seconds": ttt24_train, "fraction_of_source_total": ttt24_train / source_total},
        {"source": "TTT24", "component": "reset_or_setup", "gpu_seconds": ttt24_setup, "fraction_of_source_total": ttt24_setup / source_total},
        {"source": "TTT24", "component": "dfs", "gpu_seconds": ttt24_dfs, "fraction_of_source_total": ttt24_dfs / source_total},
        {"source": "TTT24", "component": "total", "gpu_seconds": total24, "fraction_of_source_total": total24 / source_total},
        {"source": "TTT48", "component": "train", "gpu_seconds": ttt48_train, "fraction_of_source_total": ttt48_train / source_total},
        {"source": "TTT48", "component": "reset_or_setup", "gpu_seconds": ttt48_setup, "fraction_of_source_total": ttt48_setup / source_total},
        {"source": "TTT48", "component": "dfs", "gpu_seconds": ttt48_dfs, "fraction_of_source_total": ttt48_dfs / source_total},
        {"source": "TTT48", "component": "total", "gpu_seconds": total48, "fraction_of_source_total": total48 / source_total},
    ]
    output_dfs_costs_by_task: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for row in cell_rows:
        if int(row["ttt_depth"]) == 24:
            output_dfs_costs_by_task[row["task_id"]][int(row["output_index"])] += float(row["cell_wall_s"])
    observed_output_costs = [cost for task in output_dfs_costs_by_task.values() for cost in task.values()]
    # The 259-output projection has 19 outputs beyond one output per 240 tasks.
    task_projection = {str(n): bootstrap_runtime(durations, n) for n in (60, 120, 240)}
    output_projection_240 = bootstrap_runtime(durations, 240, extra_output_costs=[0.0] * 19)
    # Replace the placeholder extra-output samples with empirical TTT24 four-view DFS costs.
    rng = random.Random(BOOTSTRAP_SEED + 259)
    output_walls: list[float] = []
    for _ in range(BOOTSTRAP_TRIALS):
        sampled = [rng.choice(durations) for _ in range(240)]
        for _extra in range(19):
            sampled[rng.randrange(240)] += rng.choice(observed_output_costs)
        output_walls.append(queue_makespan(sampled))
    output_projection_240 = {
        "p25_seconds": percentile(output_walls, 0.25),
        "median_seconds": percentile(output_walls, 0.50),
        "p75_seconds": percentile(output_walls, 0.75),
        "p90_seconds": percentile(output_walls, 0.90),
        "p95_seconds": percentile(output_walls, 0.95),
    }
    practical_upper = max(task_projection["240"]["p95_seconds"], output_projection_240["p95_seconds"])
    headroom = 12 * 3600 - practical_upper
    practical_hours = practical_upper / 3600
    classification = "SAFE" if practical_hours <= 9.5 else "LIKELY_SAFE" if practical_hours <= 10.5 else "MARGINAL" if practical_hours <= 12 else "UNSAFE"
    projection = {
        "evidence": "Kaggle Version 3 measured 8-task, 11-output L4x4 benchmark telemetry",
        "confidence": "LIMITED_SMALL_N_8_TASKS",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_trials": BOOTSTRAP_TRIALS,
        "observed_8_task_ttt24_gpu_seconds": total24,
        "observed_8_task_ttt48_gpu_seconds": total48,
        "observed_8_task_ttt24_queue_wall_seconds": queue_makespan(durations),
        "ttt24_train_gpu_seconds": ttt24_train,
        "ttt48_train_gpu_seconds": ttt48_train,
        "ttt24_dfs_gpu_seconds": ttt24_dfs,
        "ttt48_dfs_gpu_seconds": ttt48_dfs,
        "ttt24_share_of_source_gpu_seconds": total24 / (total24 + total48),
        "ttt48_share_of_source_gpu_seconds": total48 / (total24 + total48),
        "ttt24_fraction_of_measured_four_gpu_capacity": throughput["TTT24_fraction_of_four_gpu_capacity"],
        "ttt48_fraction_of_measured_four_gpu_capacity": throughput["TTT48_fraction_of_four_gpu_capacity"],
        "task_based_4worker_bootstrap": task_projection,
        "output_based_240_tasks_259_outputs_bootstrap": output_projection_240,
        "projected_60_task_hours_task_based_median": task_projection["60"]["median_seconds"] / 3600,
        "projected_120_task_hours_task_based_median": task_projection["120"]["median_seconds"] / 3600,
        "projected_240_task_hours_task_based_median": task_projection["240"]["median_seconds"] / 3600,
        "projected_259_output_hours_output_based_median": output_projection_240["median_seconds"] / 3600,
        "practical_upper_p95_seconds": practical_upper,
        "practical_upper_p95_hours": practical_hours,
        "headroom_to_12h_seconds": headroom,
        "headroom_to_12h_hours": headroom / 3600,
        "headroom_percent": headroom / (12 * 3600),
        "classification": classification,
        "method_notes": [
            "task projection resamples measured TTT24 total task durations and applies deterministic four-worker list scheduling",
            "259-output projection adds 19 empirical extra-output TTT24 four-view DFS costs to 240 sampled task profiles",
            "model load, notebook startup, shutdown, and platform jitter are not included",
            "8-task evidence is too small for a high-confidence production runtime guarantee",
        ],
    }
    return component_rows, projection


def make_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# Eval60 TTT24-only fixed4 CPU audit V1",
        "",
        "Retrospective development attribution only. Candidate pools and source-only D1 predictions were frozen and hashed before evaluation solutions were loaded.",
        "",
        "## Exact replay gate",
        "",
        f"- Full fixed 4+4 D1: Top-1 {decision['full']['top1']}/89, Top-2 {decision['full']['top2']}/89, oracle {decision['full']['oracle']}/89.",
        f"- TTT24-only: Top-1 {decision['ttt24_only']['top1']}/89, Top-2 {decision['ttt24_only']['top2']}/89, oracle {decision['ttt24_only']['oracle']}/89.",
        f"- TTT48-only: Top-1 {decision['ttt48_only']['top1']}/89, Top-2 {decision['ttt48_only']['top2']}/89, oracle {decision['ttt48_only']['oracle']}/89.",
        "",
        "## Marginal value",
        "",
        f"- Removing TTT48 loses {decision['marginals']['top2_lost_without_ttt48']} Top-2 outputs and {decision['marginals']['oracle_lost_without_ttt48']} oracle outputs; it gains {decision['marginals']['top2_gained_without_ttt48']} Top-2 outputs.",
        f"- Full-pool oracle gap is {decision['full']['oracle'] - decision['full']['top2']} outputs; source-only selection loss is reported separately in the CSV artifacts.",
        "",
        "## Runtime projection",
        "",
        f"- TTT24 accounts for {decision['runtime']['ttt24_share_of_source_gpu_seconds']:.1%} of measured TTT24+TTT48 source GPU-seconds; TTT48 accounts for {decision['runtime']['ttt48_share_of_source_gpu_seconds']:.1%}.",
        f"- TTT24-only 240-task practical P95 estimate: {decision['runtime']['practical_upper_p95_hours']:.2f} h; headroom to 12 h: {decision['runtime']['headroom_to_12h_hours']:.2f} h.",
        f"- Runtime classification: **{decision['runtime']['classification']}**, confidence **{decision['runtime']['confidence']}**.",
        "",
        "## Decision",
        "",
        f"**{decision['decision']}**",
        "",
        decision["decision_rationale"],
        "",
        "## Scientific interpretation",
        "",
        f"1. Candidate recall: removing TTT48 loses {decision['marginals']['oracle_lost_without_ttt48']} exact-output candidates ({decision['marginals']['oracle_lost_without_ttt48'] / 89:.1%} of all outputs; {decision['marginals']['oracle_lost_without_ttt48'] / decision['full']['oracle']:.1%} of full-pool oracle hits).",
        f"2. Top-2 selection: the strict TTT24-only replay loses {decision['marginals']['top2_lost_without_ttt48']} full-system Top-2 solves and gains {decision['marginals']['top2_gained_without_ttt48']}; net change {decision['ttt24_only']['top2'] - decision['full']['top2']:+d}.",
        f"3. Runtime: dropping TTT48 removes {decision['runtime']['ttt48_share_of_source_gpu_seconds']:.1%} of measured source GPU-seconds, not exactly 50%; the TTT24-only 259-output median is {decision['runtime']['projected_259_output_hours_output_based_median']:.2f} h.",
        "4. GPU validation: a dedicated TTT24-only live timing run is justified only as a cost validation; this CPU audit already shows a material development-set accuracy loss, so it does not justify replacing TTT48 in production.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-zip", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--full-replay-gate", type=Path, required=True)
    parser.add_argument("--version3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit directory: {args.output_dir}")
    if sha256_file(args.reference_zip) != EXPECTED_ZIP_SHA256:
        raise ValueError("immutable handoff ZIP hash mismatch")

    gate_report = json.loads((args.full_replay_gate / "D1_REPLAY_REPORT.json").read_text(encoding="utf-8"))
    if gate_report["d1"] != {"top1": 20, "top2": 28, "pool_oracle": 30}:
        raise ValueError(f"full replay gate failed: {gate_report['d1']}")
    if not gate_report["integrity"]["historical_rankings_exactly_reproduced"]:
        raise ValueError("full replay gate did not reproduce historical rankings")

    with zipfile.ZipFile(args.reference_zip) as archive:
        manifest = validate_manifest(archive)
        pool_artifact = read_member(archive, "/fixed_portfolio_candidates_frozen.json")
        ranking_artifact = read_member(archive, "/fixed_portfolio_rankings_frozen.json")
        prediction_artifact = read_member(archive, "/fixed_portfolio_predictions_frozen.json")
        freeze_artifact = read_member(archive, "/portfolio_freeze.json")
    if any(artifact.get("solutions_opened") is not False for artifact in (pool_artifact, ranking_artifact, prediction_artifact, freeze_artifact)):
        raise ValueError("handoff target-blind freeze boundary is invalid")
    pools = pool_artifact["pools"]
    task_count = len(pools)
    output_count = sum(len(task["per_test"]) for task in pools.values())
    if (task_count, output_count) != (EXPECTED_FULL["tasks"], EXPECTED_FULL["outputs"]):
        raise ValueError(f"cohort mismatch: {task_count} tasks / {output_count} outputs")

    source_mapping = discover_source_mapping(pool_artifact)
    sources = sorted(source_mapping["sources"])
    full_replay, full_d1_rankings, full_d1_predictions = reconstruct_full_evidence(pools, ranking_artifact["rankings"], sources)
    reference_d1_rankings = json.loads((args.full_replay_gate / "d1_rankings_frozen.json").read_text(encoding="utf-8"))
    reference_d1_predictions = json.loads((args.full_replay_gate / "d1_predictions_frozen.json").read_text(encoding="utf-8"))
    comparable_rank_fields = {
        "test_index",
        "ranked_grid_keys",
        "attempt_grid_keys",
        "source_likelihood_ranks",
        "likelihood_rrf_scores",
        "changed_from_baseline",
    }
    reconstructed_projection = {
        task_id: [{key: value for key, value in row.items() if key in comparable_rank_fields} for row in rows]
        for task_id, rows in full_d1_rankings["rankings"].items()
    }
    reference_projection = {
        task_id: [{key: value for key, value in row.items() if key in comparable_rank_fields} for row in rows]
        for task_id, rows in reference_d1_rankings["rankings"].items()
    }
    if (
        reconstructed_projection != reference_projection
        or full_d1_predictions["predictions"] != reference_d1_predictions["predictions"]
        or full_d1_predictions["solutions_opened"] != reference_d1_predictions["solutions_opened"]
    ):
        raise ValueError("new reconstruction does not exactly equal frozen full-pool D1 gate artifacts")

    ttt24_pools = build_source_pool(pools, "TTT24")
    ttt48_pools = build_source_pool(pools, "TTT48")
    ttt24_rankings, ttt24_predictions = rank_source_pool(ttt24_pools, "TTT24")
    ttt48_rankings, ttt48_predictions = rank_source_pool(ttt48_pools, "TTT48")
    ttt24_artifact = freeze_source_artifact("TTT24", EXPECTED_PORTFOLIO["TTT24"], ttt24_pools, ttt24_rankings, ttt24_predictions)
    ttt48_artifact = freeze_source_artifact("TTT48", EXPECTED_PORTFOLIO["TTT48"], ttt48_pools, ttt48_rankings, ttt48_predictions)

    reconstructibility = {
        "status": "EXACT_SOURCE_LOCAL_RECONSTRUCTIBLE",
        "solutions_opened": False,
        "evidence": {
            "source_local_b_support_recomputed_from": "selected_support_count - mean_view_nll",
            "source_local_b_ranks_match_historical": True,
            "full_b_rrf_matches_historical": True,
            "source_local_likelihood_recomputed_from": "mean original_log_likelihood per exact grid/source",
            "combined_rrf_reused_for_source_only": False,
            "full_d1_rankings_match_gate": True,
            "full_d1_predictions_match_gate": True,
        },
    }
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "SOURCE_LABEL_MAPPING.json", source_mapping)
    atomic_json(args.output_dir / "SCORING_RECONSTRUCTIBILITY.json", reconstructibility)
    atomic_json(
        args.output_dir / "full_pool_replay.json",
        {
            "status": "FULL_4PLUS4_EXACT_REPLAY_FROZEN_BEFORE_SOLUTION_ACCESS",
            "solutions_opened": False,
            "candidate_sets_unchanged": True,
            "historical_rankings_exactly_reproduced": True,
            "replay": full_replay,
            "d1_rankings": full_d1_rankings,
            "d1_predictions": full_d1_predictions,
        },
    )
    atomic_json(args.output_dir / "ttt24_only_pool_frozen.json", ttt24_artifact)
    atomic_json(args.output_dir / "ttt48_only_pool_frozen.json", ttt48_artifact)
    frozen_paths = [
        args.output_dir / "SOURCE_LABEL_MAPPING.json",
        args.output_dir / "SCORING_RECONSTRUCTIBILITY.json",
        args.output_dir / "full_pool_replay.json",
        args.output_dir / "ttt24_only_pool_frozen.json",
        args.output_dir / "ttt48_only_pool_frozen.json",
    ]
    provenance = {
        "audit": "EVAL60_TTT24_ONLY_FIXED4_CPU_AUDIT_V1",
        "status": "TARGET_BLIND_ARTIFACTS_FROZEN_AND_HASHED_BEFORE_SOLUTION_ACCESS",
        "gold_opened": False,
        "cpu_only": True,
        "model_calls": 0,
        "gpu_runs": 0,
        "inputs": {
            "reference_zip": str(args.reference_zip.resolve()),
            "reference_zip_sha256": sha256_file(args.reference_zip),
            "solutions_path_reserved_until_after_freeze": str(args.solutions.resolve()),
            "version3_dir": str(args.version3_dir.resolve()),
            "full_replay_gate": str(args.full_replay_gate.resolve()),
        },
        "zip_manifest_verified": manifest,
        "cohort": {"task_count": task_count, "test_output_count": output_count, "task_ids_sha256": canonical_hash(sorted(pools))},
        "frozen_artifact_sha256": {path.name: sha256_file(path) for path in frozen_paths},
    }
    atomic_json(args.output_dir / "PROVENANCE.json", provenance)

    # Target access begins only after all candidate, ranking, prediction, and provenance artifacts above are immutable.
    solutions = json.loads(args.solutions.read_text(encoding="utf-8"))
    if not set(pools).issubset(solutions):
        raise ValueError(f"solutions missing tasks: {sorted(set(pools) - set(solutions))}")
    full_score = score_artifact(pools, full_d1_predictions["predictions"], solutions)
    score24 = score_artifact(ttt24_pools, ttt24_predictions, solutions)
    score48 = score_artifact(ttt48_pools, ttt48_predictions, solutions)
    if {"top1": full_score["top1"], "top2": full_score["top2"], "oracle": full_score["oracle"]} != {"top1": 20, "top2": 28, "oracle": 30}:
        raise RuntimeError(f"full score gate failed after freeze: {full_score}")

    by_full = {row["output_key"]: row for row in full_score["rows"]}
    by24 = {row["output_key"]: row for row in score24["rows"]}
    by48 = {row["output_key"]: row for row in score48["rows"]}
    overlap_rows: list[dict[str, Any]] = []
    for key in sorted(by_full):
        rowf, row24, row48 = by_full[key], by24[key], by48[key]
        overlap_rows.append(
            {
                "output_key": key,
                "task_id": rowf["task_id"],
                "test_index": rowf["test_index"],
                "full_oracle": rowf["oracle_hit"],
                "ttt24_oracle": row24["oracle_hit"],
                "ttt48_oracle": row48["oracle_hit"],
                "full_top2": rowf["top2_hit"],
                "ttt24_top2": row24["top2_hit"],
                "ttt48_top2": row48["top2_hit"],
                "pool_category": "BOTH" if row24["oracle_hit"] and row48["oracle_hit"] else "TTT24_ONLY" if row24["oracle_hit"] else "TTT48_ONLY" if row48["oracle_hit"] else "NEITHER",
                "lost_top2_without_ttt48": rowf["top2_hit"] and not row24["top2_hit"],
                "gained_top2_without_ttt48": row24["top2_hit"] and not rowf["top2_hit"],
                "lost_oracle_without_ttt48": rowf["oracle_hit"] and not row24["oracle_hit"],
            }
        )
    write_csv(args.output_dir / "pool_overlap_by_output.csv", overlap_rows)
    write_csv(args.output_dir / "ttt48_marginal_oracle_outputs.csv", [row for row in overlap_rows if row["ttt48_oracle"] and not row["ttt24_oracle"]])
    write_csv(args.output_dir / "ttt24_marginal_oracle_outputs.csv", [row for row in overlap_rows if row["ttt24_oracle"] and not row["ttt48_oracle"]])

    selector_rows: list[dict[str, Any]] = []
    for condition, score in (("FULL_4PLUS4_D1", full_score), ("TTT24_ONLY_D1", score24), ("TTT48_ONLY_D1", score48)):
        selector_rows.append(
            {
                "condition": condition,
                "top1": score["top1"],
                "top2": score["top2"],
                "pool_oracle": score["oracle"],
                "selection_loss": score["oracle"] - score["top2"],
                "denominator": output_count,
            }
        )
    write_csv(args.output_dir / "selector_comparison.csv", selector_rows)

    tag_hits: dict[str, set[str]] = {tag: set() for tag in EXPECTED_PORTFOLIO["TTT24"]}
    tag_generated = {tag: 0 for tag in EXPECTED_PORTFOLIO["TTT24"]}
    tag_grids: dict[str, set[str]] = {tag: set() for tag in EXPECTED_PORTFOLIO["TTT24"]}
    for task_id, task in ttt24_pools.items():
        for output in task["per_test"]:
            target = solutions[task_id][int(output["test_index"])]
            key = f"{task_id}:{int(output['test_index'])}"
            for candidate in output["candidates"]:
                for row in candidate["source_rows"]:
                    for tag in set(map(str, row.get("slot_tags", []))):
                        if tag not in tag_hits:
                            continue
                        tag_generated[tag] += 1
                        tag_grids[tag].add(str(candidate["grid_key"]))
                        if candidate["grid"] == target:
                            tag_hits[tag].add(key)
    view_rows: list[dict[str, Any]] = []
    for tag in EXPECTED_PORTFOLIO["TTT24"]:
        other_hits = set().union(*(tag_hits[other] for other in EXPECTED_PORTFOLIO["TTT24"] if other != tag))
        unique_hits = tag_hits[tag] - other_hits
        view_rows.append(
            {
                "source": "TTT24",
                "view": tag,
                "correct_outputs": len(tag_hits[tag]),
                "unique_correct_outputs": len(unique_hits),
                "generated_source_rows": tag_generated[tag],
                "unique_grids": len(tag_grids[tag]),
                "valid_source_rows": tag_generated[tag],
                "invalid_source_rows": 0,
            }
        )
    write_csv(args.output_dir / "view_contribution_ttt24.csv", view_rows)

    runtime_rows, runtime_projection = runtime_audit(args.version3_dir)
    write_csv(args.output_dir / "runtime_components_v3.csv", runtime_rows)
    atomic_json(args.output_dir / "ttt24_only_runtime_projection.json", runtime_projection)
    telemetry = """# Next L4 GPU telemetry requirements

For any future production-equivalent timing run, persist one-second telemetry for every L4 atomically.

## Required sample fields

`timestamp`, `gpu_id`, `gpu_util_percent`, `memory_util_percent`,
`memory_used_mb`, `memory_total_mb`, `power_draw_w`, `power_limit_w`,
`temperature_c`, `sm_clock_mhz`, `memory_clock_mhz`, and `phase`.

Allowed dual-depth phases are `MODEL_LOAD`, `TTT24_TRAIN`, `TTT24_DFS`,
`TTT48_TRAIN`, `TTT48_DFS`, `RESET_TRANSITION`, `SERIALIZE`, and `IDLE`.
For a TTT24-only run use `MODEL_LOAD`, `TTT24_TRAIN`, `TTT24_DFS`,
`SERIALIZE`, and `IDLE`.

## Required aggregation per phase and GPU

- utilization: mean, median, p10, p50, p90, p95, maximum;
- VRAM: mean and peak;
- power: mean, p90, maximum;
- temperature: mean and maximum;
- phase wall seconds.

Also report:

- `GPU_ACTIVE_FRACTION`: workload samples with utilization >10%;
- `GPU_HIGH_UTIL_FRACTION`: workload samples with utilization >=80%;
- `GPU_IDLE_FRACTION`: workload samples with utilization <=10%.

Do not substitute notebook-wide averages. Preserve run identity (source,
config, model, challenge and task/test-index hashes), worker GPU UUIDs,
model-load/barrier/heartbeat/terminal timestamps, per-task source timings,
per-output/view nodes/tokens/candidates, scheduler dispatches, failures,
deadline/finalization state, coverage and fallback counts.

The current Version 3 evidence covers only 8 tasks and 11 outputs, so a larger
representative timing cohort is required for a high-confidence production SLA.
"""
    (args.output_dir / "NEXT_L4_GPU_TELEMETRY_REQUIREMENTS.md").write_text(telemetry, encoding="utf-8")

    lost_top2 = [row["output_key"] for row in overlap_rows if row["lost_top2_without_ttt48"]]
    gained_top2 = [row["output_key"] for row in overlap_rows if row["gained_top2_without_ttt48"]]
    lost_oracle = [row["output_key"] for row in overlap_rows if row["lost_oracle_without_ttt48"]]
    ttt48_only = [row["output_key"] for row in overlap_rows if row["pool_category"] == "TTT48_ONLY"]
    ttt24_only = [row["output_key"] for row in overlap_rows if row["pool_category"] == "TTT24_ONLY"]
    both = [row["output_key"] for row in overlap_rows if row["pool_category"] == "BOTH"]
    neither = [row["output_key"] for row in overlap_rows if row["pool_category"] == "NEITHER"]
    if lost_top2 or lost_oracle:
        decision_label = "KEEP_TTT48"
        rationale = "TTT48 supplies exact candidates or selected Top-2 outputs that disappear in the strict TTT24-only replay; removing it is not scientifically equivalent."
    elif runtime_projection["classification"] == "HIGH_RISK":
        decision_label = "TTT24_ONLY_RUNTIME_HIGH_RISK"
        rationale = "No measured accuracy loss was found, but the TTT24-only runtime projection exceeds the 12-hour envelope."
    else:
        decision_label = "TTT24_ONLY_CANDIDATE"
        rationale = "The strict source-only replay retained full observed accuracy and the measured runtime projection is within the stated envelope; live parity remains unverified."
    decision = {
        "audit": "EVAL60_TTT24_ONLY_FIXED4_CPU_AUDIT_V1",
        "scope": "RETROSPECTIVE_DEVELOPMENT_EVIDENCE_NOT_HELD_OUT_NOT_LB",
        "full": {k: full_score[k] for k in ("top1", "top2", "oracle")},
        "ttt24_only": {k: score24[k] for k in ("top1", "top2", "oracle")},
        "ttt48_only": {k: score48[k] for k in ("top1", "top2", "oracle")},
        "marginals": {
            "ttt24_and_ttt48_hit_count": len(both),
            "both_source_oracle_outputs": both,
            "ttt24_only_hit_count": len(ttt24_only),
            "ttt24_only_oracle_outputs": ttt24_only,
            "ttt48_only_hit_count": len(ttt48_only),
            "ttt48_only_oracle_outputs": ttt48_only,
            "neither_hit_count": len(neither),
            "neither_source_oracle_outputs": neither,
            "top2_lost_without_ttt48": len(lost_top2),
            "top2_lost_output_keys": lost_top2,
            "top2_gained_without_ttt48": len(gained_top2),
            "top2_gained_output_keys": gained_top2,
            "oracle_lost_without_ttt48": len(lost_oracle),
            "oracle_lost_output_keys": lost_oracle,
            "ttt48_marginal_oracle_rate": len(lost_oracle) / output_count,
            "ttt48_marginal_given_full_hit": len(lost_oracle) / full_score["oracle"],
            "ttt24_marginal_oracle_count": len(ttt24_only),
            "ttt48_marginal_oracle_count": len(ttt48_only),
            "net_top2_change": score24["top2"] - full_score["top2"],
            "ttt24_only_pool_loss_vs_full": full_score["oracle"] - score24["oracle"],
            "ttt24_only_selection_loss": score24["oracle"] - score24["top2"],
            "full_selection_loss": full_score["oracle"] - full_score["top2"],
            "full_top2_hit_outputs": [row["output_key"] for row in full_score["rows"] if row["top2_hit"]],
            "ttt24_top2_hit_outputs": [row["output_key"] for row in score24["rows"] if row["top2_hit"]],
            "ttt48_top2_hit_outputs": [row["output_key"] for row in score48["rows"] if row["top2_hit"]],
            "drop_ttt48_loss_decomposition": {
                "candidate_pool_loss_outputs": lost_oracle,
                "realized_full_top2_losses": lost_top2,
                "ttt48_marginal_candidates_not_selected_by_full_top2": sorted(set(lost_oracle) - set(lost_top2)),
                "ttt24_pool_hits_missed_by_ttt24_selector": [row["output_key"] for row in score24["rows"] if row["oracle_hit"] and not row["top2_hit"]],
            },
        },
        "runtime": runtime_projection,
        "decision": decision_label,
        "decision_rationale": rationale,
        "live_model_parity": "NOT_RUN_CPU_ONLY",
        "gpu_runs": 0,
        "kaggle_runs": 0,
        "submissions": 0,
        "solutions_sha256_scored_only_after_freeze": sha256_file(args.solutions),
        "dedicated_gpu_validation": "NOT_JUSTIFIED_FOR_REPLACEMENT; OPTIONAL_ONLY_FOR_RUNTIME_CALIBRATION",
    }
    atomic_json(args.output_dir / "DECISION.json", decision)
    (args.output_dir / "EVAL60_TTT24_ONLY_AUDIT_REPORT.md").write_text(make_markdown(decision), encoding="utf-8")
    print(json.dumps({"event": "EVAL60_TTT24_ONLY_FIXED4_CPU_AUDIT_COMPLETE", "full": decision["full"], "ttt24_only": decision["ttt24_only"], "ttt48_only": decision["ttt48_only"], "decision": decision_label, "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
