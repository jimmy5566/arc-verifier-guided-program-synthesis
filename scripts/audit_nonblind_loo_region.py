"""CPU-only LOFO competence-region audit for frozen Transfer30 checkpoints.

This script deliberately has no challenge or solution-file argument.  It only
reads an immutable local snapshot of ranking and production checkpoint JSON.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any


EPSILONS = (1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)


def cell_key(cell: dict[str, Any]) -> tuple[int, str]:
    return int(cell["depth"]), str(cell["gen_view"])


def canonical_grid(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def describe(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values) if values else None,
        "median": quantile(values, 0.5),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
        "max": max(values) if values else None,
    }


def fmt(value: float | None, digits: int = 6) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def snapshot_sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def ranking_for_task(task_payload: dict[str, Any], ranking_path: Path) -> dict[str, Any]:
    frozen = task_payload.get("task", {}).get("frozen_ranking", {})
    if isinstance(frozen, dict) and frozen.get("top4"):
        return frozen
    return json.loads(ranking_path.read_text(encoding="utf-8"))["ranking"]


def run(snapshot: Path, output_dir: Path) -> dict[str, Any]:
    ranking_paths = sorted((snapshot / "checkpoints" / "rankings").glob("*.json"))
    task_paths = sorted((snapshot / "checkpoints" / "tasks").glob("*.json"))
    if not ranking_paths:
        raise ValueError("No frozen ranking files found")

    exact_inclusion: dict[int, int] = {1: 0, 2: 0, 4: 0}
    held_ranks: dict[int, list[float]] = {1: [], 2: [], 4: []}
    regrets: dict[int, list[float]] = {1: [], 2: [], 4: []}
    basin: dict[float, dict[str | int, Any]] = {
        epsilon: {"sizes": [], 1: 0, 2: 0, 4: 0} for epsilon in EPSILONS
    }
    winner_depths: collections.Counter[int] = collections.Counter()
    winner_views: collections.Counter[str] = collections.Counter()
    top1_depth_match = top1_view_match = top2_depth_contains = top2_view_contains = 0
    held_fold_count = 0
    lofo_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []

    for ranking_path in ranking_paths:
        payload = json.loads(ranking_path.read_text(encoding="utf-8"))
        ranking = payload["ranking"]
        cells = ranking["all_40_cells"]
        if len(cells) != 40:
            raise ValueError(f"{ranking_path.name}: expected 40 cells, got {len(cells)}")
        fold_count = int(cells[0]["fold_count"])
        task_id = ranking_path.stem
        held_top_sets: dict[int, list[set[tuple[int, str]]]] = {1: [], 2: [], 4: [], 8: []}
        task_rows: list[dict[str, Any]] = []

        for fold in range(fold_count):
            lofo = sorted(
                cells,
                key=lambda cell: (
                    sum(cell["fold_nll_per_token"][index] for index in range(fold_count) if index != fold)
                    / (fold_count - 1),
                    cell_key(cell),
                ),
            )
            held = sorted(cells, key=lambda cell: (cell["fold_nll_per_token"][fold], cell_key(cell)))
            held_position = {cell_key(cell): index + 1 for index, cell in enumerate(held)}
            held_winner = held[0]
            winner_key = cell_key(held_winner)
            best_nll = float(held_winner["fold_nll_per_token"][fold])
            winner_depths[int(held_winner["depth"])] += 1
            winner_views[str(held_winner["gen_view"])] += 1
            fold_row: dict[str, Any] = {
                "fold": fold,
                "held_winner": {
                    "depth": int(held_winner["depth"]),
                    "view": str(held_winner["gen_view"]),
                    "nll": best_nll,
                },
                "lofo": {},
            }

            for top_k in (1, 2, 4):
                selected = lofo[:top_k]
                selected_keys = {cell_key(cell) for cell in selected}
                winner_included = winner_key in selected_keys
                best_rank = min(held_position[cell_key(cell)] for cell in selected)
                selected_nll = min(float(cell["fold_nll_per_token"][fold]) for cell in selected)
                exact_inclusion[top_k] += int(winner_included)
                held_ranks[top_k].append(float(best_rank))
                regrets[top_k].append(selected_nll - best_nll)
                fold_row["lofo"][f"top{top_k}"] = {
                    "winner_included": winner_included,
                    "best_held_rank": best_rank,
                    "nll_regret": selected_nll - best_nll,
                }
                for epsilon in EPSILONS:
                    competence_basin = {
                        cell_key(cell)
                        for cell in held
                        if float(cell["fold_nll_per_token"][fold]) - best_nll <= epsilon
                    }
                    basin[epsilon][top_k] += int(bool(selected_keys & competence_basin))
                    if top_k == 1:
                        basin[epsilon]["sizes"].append(len(competence_basin))

            top1, top2 = lofo[0], lofo[:2]
            top1_depth_match += int(int(top1["depth"]) == int(held_winner["depth"]))
            top1_view_match += int(str(top1["gen_view"]) == str(held_winner["gen_view"]))
            top2_depth_contains += int(any(int(cell["depth"]) == int(held_winner["depth"]) for cell in top2))
            top2_view_contains += int(any(str(cell["gen_view"]) == str(held_winner["gen_view"]) for cell in top2))
            for top_k in held_top_sets:
                held_top_sets[top_k].append({cell_key(cell) for cell in held[:top_k]})
            task_rows.append(fold_row)
            held_fold_count += 1

        stability: dict[str, Any] = {"task_id": task_id, "fold_count": fold_count}
        for top_k, sets in held_top_sets.items():
            overlaps = [len(left & right) / len(left | right) for left, right in itertools.combinations(sets, 2)]
            stability[f"top{top_k}_jaccard_mean"] = statistics.mean(overlaps) if overlaps else None
        winners = held_top_sets[1]
        stability["unique_rank1_cells"] = len(set().union(*winners))
        stability["unique_rank1_depths"] = len({next(iter(item))[0] for item in winners})
        stability["unique_rank1_views"] = len({next(iter(item))[1] for item in winners})
        stability_rows.append(stability)
        lofo_rows.append({"task_id": task_id, "folds": task_rows})

    production_rows: list[dict[str, Any]] = []
    same_output: list[bool] = []
    unique_outputs: dict[int, list[float]] = {1: [], 2: [], 4: []}
    same_delta: list[float] = []
    different_delta: list[float] = []
    for task_path in task_paths:
        task_payload = json.loads(task_path.read_text(encoding="utf-8"))
        task_id = task_path.stem
        ranking_path = snapshot / "checkpoints" / "rankings" / f"{task_id}.json"
        ranking = ranking_for_task(task_payload, ranking_path)
        predictions = task_payload.get("task", {}).get("production_predictions", [])
        outputs_by_label: dict[str, set[str]] = collections.defaultdict(set)
        for row in predictions:
            if row.get("parse_valid") is True:
                output = canonical_grid(row.get("prediction"))
                for label in row.get("labels", []):
                    if label in {"LOO_TOP1", "LOO_TOP2", "LOO_TOP4"}:
                        outputs_by_label[label].add(output)
        usable = bool(ranking.get("top4")) and "LOO_TOP1" in outputs_by_label and "LOO_TOP2" in outputs_by_label
        row: dict[str, Any] = {"task_id": task_id, "usable": usable, "prediction_record_count": len(predictions)}
        if usable:
            rank1, rank2 = ranking["top1"][0], ranking["top2"][1]
            equal = outputs_by_label["LOO_TOP1"] == outputs_by_label["LOO_TOP2"]
            delta = float(rank2["cv_nll_per_token_mean"]) - float(rank1["cv_nll_per_token_mean"])
            same_output.append(equal)
            (same_delta if equal else different_delta).append(delta)
            row.update(
                {
                    "rank1": {"depth": int(rank1["depth"]), "view": str(rank1["gen_view"]), "cv_nll": rank1["cv_nll_per_token_mean"]},
                    "rank2": {"depth": int(rank2["depth"]), "view": str(rank2["gen_view"]), "cv_nll": rank2["cv_nll_per_token_mean"]},
                    "delta_nll_12": delta,
                    "same_depth": int(rank1["depth"]) == int(rank2["depth"]),
                    "same_view": str(rank1["gen_view"]) == str(rank2["gen_view"]),
                    "same_canonical_output": equal,
                }
            )
            for top_k, labels in ((1, ("LOO_TOP1",)), (2, ("LOO_TOP1", "LOO_TOP2")), (4, ("LOO_TOP1", "LOO_TOP2", "LOO_TOP4"))):
                values: set[str] = set()
                for label in labels:
                    values.update(outputs_by_label[label])
                unique_outputs[top_k].append(float(len(values)))
        production_rows.append(row)

    metric = {
        "winner_inclusion": {
            f"top{top_k}": {"count": exact_inclusion[top_k], "fraction": exact_inclusion[top_k] / held_fold_count}
            for top_k in (1, 2, 4)
        },
        "held_fold_rank": {f"top{top_k}": describe(held_ranks[top_k]) for top_k in (1, 2, 4)},
        "nll_regret": {f"top{top_k}": describe(regrets[top_k]) for top_k in (1, 2, 4)},
        "near_optimal_basin": {
            str(epsilon): {
                "basin_size": describe(basin[epsilon]["sizes"]),
                **{
                    f"top{top_k}_region_retrieval": {
                        "count": basin[epsilon][top_k],
                        "fraction": basin[epsilon][top_k] / held_fold_count,
                    }
                    for top_k in (1, 2, 4)
                },
            }
            for epsilon in EPSILONS
        },
        "fold_stability": {
            "mean_jaccard": {
                f"top{top_k}": statistics.mean(
                    [row[f"top{top_k}_jaccard_mean"] for row in stability_rows if row[f"top{top_k}_jaccard_mean"] is not None]
                )
                for top_k in (1, 2, 4, 8)
            },
            "per_task": stability_rows,
        },
        "depth_view": {
            "top1_same_winner_depth": {"count": top1_depth_match, "fraction": top1_depth_match / held_fold_count},
            "top1_same_winner_view": {"count": top1_view_match, "fraction": top1_view_match / held_fold_count},
            "top2_contains_winner_depth": {"count": top2_depth_contains, "fraction": top2_depth_contains / held_fold_count},
            "top2_contains_winner_view": {"count": top2_view_contains, "fraction": top2_view_contains / held_fold_count},
            "held_fold_winner_depth_counts": dict(sorted(winner_depths.items())),
            "held_fold_winner_view_counts": dict(sorted(winner_views.items())),
        },
        "production_diversity": {
            "frozen_production_tasks": len(task_paths),
            "usable_tasks": len(same_output),
            "top1_top2_identical_output": {"count": sum(same_output), "fraction": sum(same_output) / len(same_output) if same_output else None},
            "unique_output_count": {f"top{top_k}": describe(unique_outputs[top_k]) for top_k in (1, 2, 4)},
            "delta_nll_12_same_output": describe(same_delta),
            "delta_nll_12_different_output": describe(different_delta),
            "per_task": production_rows,
        },
    }
    payload = {
        "experiment": "NONBLIND_DEVELOPMENT_LOO_TRANSFER30_V1",
        "audit": "CPU_ONLY_TARGET_BLIND_INTERIM_LOFO_REGION_AUDIT",
        "snapshot_path": str(snapshot),
        "frozen_ranking_task_count": len(ranking_paths),
        "frozen_production_task_count": len(task_paths),
        "ranking_snapshot_sha256": snapshot_sha(ranking_paths),
        "evaluation_solutions_opened": False,
        "fixed_epsilons": list(EPSILONS),
        "random_baseline": {
            "winner_inclusion": {"top1": 0.025, "top2": 0.05, "top4": 0.10},
            "expected_best_rank": {"top1": 20.5, "top2": 41 / 3, "top4": 8.2},
        },
        "held_fold_count": held_fold_count,
        "metrics": metric,
        "lofo_fold_rows": lofo_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "LOO_REGION_AUDIT_INTERIM.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Interim LOFO competence-region audit",
        "",
        "CPU-only and target-blind. This reads only the Git-frozen checkpoint snapshot; no evaluation solution or real-test correctness data was opened.",
        "",
        f"- Frozen ranking tasks: `{len(ranking_paths)}`",
        f"- Frozen production tasks: `{len(task_paths)}`",
        f"- Held LOO folds: `{held_fold_count}`",
        f"- Ranking snapshot SHA256: `{payload['ranking_snapshot_sha256']}`",
        "",
        "## LOFO region retrieval",
        "",
        "| K | Exact held-fold winner inclusion | Median held-fold rank | Median NLL regret |",
        "| --- | ---: | ---: | ---: |",
    ]
    for top_k in (1, 2, 4):
        inclusion = metric["winner_inclusion"][f"top{top_k}"]
        lines.append(
            f"| Top{top_k} | {inclusion['fraction']:.3f} ({inclusion['count']}/{held_fold_count}) | "
            f"{fmt(metric['held_fold_rank'][f'top{top_k}']['median'], 2)} | {fmt(metric['nll_regret'][f'top{top_k}']['median'])} |"
        )
    lines += ["", "## Near-optimal competence-basin retrieval", "", "| Epsilon | Median basin cells | Top1 | Top2 | Top4 |", "| --- | ---: | ---: | ---: | ---: |"]
    for epsilon in EPSILONS:
        item = metric["near_optimal_basin"][str(epsilon)]
        lines.append(
            f"| {epsilon:g} | {fmt(item['basin_size']['median'], 2)} | "
            f"{item['top1_region_retrieval']['fraction']:.3f} | {item['top2_region_retrieval']['fraction']:.3f} | {item['top4_region_retrieval']['fraction']:.3f} |"
        )
    diversity = metric["production_diversity"]
    lines += [
        "",
        "## Frozen production diversity",
        "",
        f"- Usable completed task predictions: `{diversity['usable_tasks']}/{len(task_paths)}`",
        f"- Top1/Top2 identical canonical-output rate: `{fmt(diversity['top1_top2_identical_output']['fraction'], 3)}` ({diversity['top1_top2_identical_output']['count']}/{diversity['usable_tasks']})",
        f"- Mean unique outputs: Top1 `{fmt(diversity['unique_output_count']['top1']['mean'], 3)}`, Top2 `{fmt(diversity['unique_output_count']['top2']['mean'], 3)}`, Top4 `{fmt(diversity['unique_output_count']['top4']['mean'], 3)}`.",
        "",
        "## Scope",
        "",
        "This measures transfer of train-pair LOO NLL regions only. It does not establish real-test output correctness and cannot authorize a production decision.",
    ]
    (output_dir / "LOO_REGION_AUDIT_INTERIM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.snapshot, args.output_dir)
    print(
        json.dumps(
            {
                "ranking_tasks": payload["frozen_ranking_task_count"],
                "production_tasks": payload["frozen_production_task_count"],
                "ranking_snapshot_sha256": payload["ranking_snapshot_sha256"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
