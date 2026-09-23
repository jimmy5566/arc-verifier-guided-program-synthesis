#!/usr/bin/env python3
"""CPU-only D1 replay from the immutable 4+4 selector-review ZIP.

D1 adds source-local likelihood reciprocal-rank fusion only as an exact
tie-break of the frozen B-RRF score.  Candidate pools are never regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.selector_d1 import attempts_from_order, baseline_order, d1_order, verify_d1_scope


DEFAULT_REFERENCE_ZIP = ROOT.parents[2] / "ARC2_TTT24_TTT48_4plus4_selector_handoff_20260924.zip"
DEFAULT_OUTPUT = ROOT / "artifacts" / "eval60_4plus4_d1_selector"
SOLUTIONS = ROOT / "data" / "raw" / "arc-agi_evaluation_solutions.json"
EXPECTED = {"baseline_top1": 20, "baseline_top2": 27, "d1_top1": 20, "d1_top2": 28, "pool_oracle": 30}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_member(archive: zipfile.ZipFile, suffix: str) -> Any:
    names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"reference ZIP must contain exactly one {suffix}; found {names}")
    return json.loads(archive.read(names[0]).decode("utf-8"))


def validate_reference_manifest(archive: zipfile.ZipFile) -> None:
    manifests = [name for name in archive.namelist() if name.endswith("/MANIFEST.sha256")]
    if len(manifests) != 1:
        raise ValueError("reference ZIP is missing MANIFEST.sha256")
    root = manifests[0].rsplit("/", 1)[0]
    for line in archive.read(manifests[0]).decode("utf-8").splitlines():
        expected, filename = line.split("  ", 1)
        actual = hashlib.sha256(archive.read(f"{root}/{filename}")).hexdigest()
        if actual != expected:
            raise ValueError(f"reference ZIP manifest mismatch for {filename}")


def grid_key(grid: Any) -> str:
    return json.dumps(grid, separators=(",", ":"))


def freeze_replay(*, pools: dict[str, Any], historic_rankings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline: dict[str, Any] = {}
    d1_rankings: dict[str, Any] = {}
    d1_predictions: dict[str, Any] = {}
    for task_id, task_pool in pools.items():
        historic_by_index = {int(row["test_index"]): row for row in historic_rankings[task_id]}
        baseline_rows: list[dict[str, Any]] = []
        d1_rows: list[dict[str, Any]] = []
        attempts1: list[Any] = []
        attempts2: list[Any] = []
        for per_test in task_pool["per_test"]:
            index = int(per_test["test_index"])
            candidates = list(per_test["candidates"])
            historic = historic_by_index.get(index)
            if historic is None:
                raise ValueError(f"missing historical ranking for {task_id}:{index}")
            replay_baseline = baseline_order(candidates)
            frozen_baseline = list(historic["ranked_grid_keys"])
            if replay_baseline != frozen_baseline:
                raise ValueError(f"historical reproduction mismatch for {task_id}:{index}")
            ordered, likelihood_ranks, likelihood_rrf = d1_order(candidates)
            verify_d1_scope(candidates, ordered)
            first, second = attempts_from_order(candidates, ordered)
            baseline_rows.append({"test_index": index, "ranked_grid_keys": replay_baseline, "attempt_grid_keys": list(historic["attempt_grid_keys"])})
            d1_rows.append(
                {
                    "test_index": index,
                    "ranked_grid_keys": ordered,
                    "attempt_grid_keys": ordered[:2] if len(ordered) >= 2 else ordered * 2,
                    "source_likelihood_ranks": likelihood_ranks,
                    "likelihood_rrf_scores": likelihood_rrf,
                    "changed_from_baseline": ordered != replay_baseline,
                }
            )
            attempts1.append(first)
            attempts2.append(second)
        baseline[task_id] = baseline_rows
        d1_rankings[task_id] = d1_rows
        d1_predictions[task_id] = {"attempt_1": attempts1, "attempt_2": attempts2}
    return baseline, d1_rankings, d1_predictions


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def score(*, pools: dict[str, Any], predictions: dict[str, Any], solutions: dict[str, Any]) -> dict[str, Any]:
    top1 = top2 = oracle = 0
    rows: list[dict[str, Any]] = []
    for task_id, task_pool in pools.items():
        prediction = predictions[task_id]
        for per_test in task_pool["per_test"]:
            index = int(per_test["test_index"])
            target = solutions[task_id][index]
            first = prediction["attempt_1"][index]
            second = prediction["attempt_2"][index]
            pool_hit = any(candidate["grid"] == target for candidate in per_test["candidates"])
            first_hit = first == target
            second_hit = second == target
            oracle += int(pool_hit)
            top1 += int(first_hit)
            top2 += int(first_hit or second_hit)
            rows.append({"task_id": task_id, "test_index": index, "pool_oracle_hit": pool_hit, "top1_hit": first_hit, "top2_hit": first_hit or second_hit})
    return {"top1": top1, "top2": top2, "pool_oracle": oracle, "rows": rows}


def report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Eval60 fixed 4+4 D1 selector replay",
            "",
            "CPU-only, retrospective development evidence. D1 rankings and predictions were persisted before solutions were opened.",
            "",
            f"- Historical baseline: Top-1 {report['baseline']['top1']}/89; Top-2 {report['baseline']['top2']}/89.",
            f"- D1: Top-1 {report['d1']['top1']}/89; Top-2 {report['d1']['top2']}/89; pool oracle {report['d1']['pool_oracle']}/89.",
            f"- New solves: {report['paired']['new_solves']}; regressions: {report['paired']['regressions']}; net: {report['paired']['net_gain']}.",
            f"- Outputs whose D1 order changed: {report['integrity']['changed_order_outputs']}/89.",
            "",
            "D1 only applies likelihood RRF as a deterministic tie-break for exact B-RRF score ties; no raw likelihood is compared across sources.",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-zip", type=Path, default=DEFAULT_REFERENCE_ZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--solutions", type=Path, default=SOLUTIONS)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing D1 output: {args.output_dir}")
    if not args.reference_zip.is_file():
        raise FileNotFoundError(args.reference_zip)

    with zipfile.ZipFile(args.reference_zip) as archive:
        validate_reference_manifest(archive)
        pool_artifact = read_member(archive, "/fixed_portfolio_candidates_frozen.json")
        ranking_artifact = read_member(archive, "/fixed_portfolio_rankings_frozen.json")
        prediction_artifact = read_member(archive, "/fixed_portfolio_predictions_frozen.json")
    if pool_artifact.get("solutions_opened") is not False or ranking_artifact.get("solutions_opened") is not False:
        raise ValueError("reference candidate/ranking artifacts are not target-blind frozen artifacts")

    pools = pool_artifact["pools"]
    historic_rankings = ranking_artifact["rankings"]
    historic_predictions = prediction_artifact["predictions"]
    if set(pools) != set(historic_rankings) or set(pools) != set(historic_predictions):
        raise ValueError("reference task coverage mismatch")

    baseline, d1_rankings, d1_predictions = freeze_replay(pools=pools, historic_rankings=historic_rankings)
    config = {
        "selector": "D1_L_RRF_EXACT_B_RRF_TIE_BREAK",
        "rule": "(-B_RRF, -L_RRF, historical_representative, grid_key)",
        "likelihood_aggregation": "mean original_log_likelihood per unique grid per source",
        "cross_source_raw_likelihood_comparison": False,
        "candidate_generation": False,
        "model_calls": 0,
        "reference_zip": str(args.reference_zip),
        "reference_zip_sha256": sha256(args.reference_zip),
        "expected": EXPECTED,
        "solutions_opened": False,
    }
    atomic_json(args.output_dir / "d1_config_frozen.json", config)
    atomic_json(args.output_dir / "baseline_replay_frozen.json", {"solutions_opened": False, "rankings": baseline})
    atomic_json(args.output_dir / "d1_rankings_frozen.json", {"solutions_opened": False, "rankings": d1_rankings})
    atomic_json(args.output_dir / "d1_predictions_frozen.json", {"solutions_opened": False, "predictions": d1_predictions})
    frozen_hashes = {path.name: sha256(path) for path in sorted(args.output_dir.glob("*_frozen.json"))}

    # The freeze above is deliberately complete before target access.
    solutions = json.loads(args.solutions.read_text(encoding="utf-8"))
    baseline_predictions = historic_predictions
    baseline_score = score(pools=pools, predictions=baseline_predictions, solutions=solutions)
    d1_score = score(pools=pools, predictions=d1_predictions, solutions=solutions)
    if (baseline_score["top1"], baseline_score["top2"]) != (EXPECTED["baseline_top1"], EXPECTED["baseline_top2"]):
        raise RuntimeError(f"historical score mismatch: {baseline_score['top1']}/{baseline_score['top2']}")
    if (d1_score["top1"], d1_score["top2"], d1_score["pool_oracle"]) != (EXPECTED["d1_top1"], EXPECTED["d1_top2"], EXPECTED["pool_oracle"]):
        raise RuntimeError(f"D1 expected replay mismatch: {d1_score['top1']}/{d1_score['top2']}/{d1_score['pool_oracle']}")

    baseline_hits = {(row["task_id"], row["test_index"]) for row in baseline_score["rows"] if row["top2_hit"]}
    d1_hits = {(row["task_id"], row["test_index"]) for row in d1_score["rows"] if row["top2_hit"]}
    changed = [
        f"{task_id}:{row['test_index']}"
        for task_id, rows in d1_rankings.items()
        for row in rows
        if row["changed_from_baseline"]
    ]
    report = {
        "status": "COMPLETE_SCORED_AFTER_D1_FREEZE",
        "scope": "RETROSPECTIVE_DEVELOPMENT_EVIDENCE_NOT_HELD_OUT_NOT_LB_PERFORMANCE",
        "baseline": {key: baseline_score[key] for key in ("top1", "top2", "pool_oracle")},
        "d1": {key: d1_score[key] for key in ("top1", "top2", "pool_oracle")},
        "paired": {"new_solves": sorted(f"{task_id}:{index}" for task_id, index in d1_hits - baseline_hits), "regressions": sorted(f"{task_id}:{index}" for task_id, index in baseline_hits - d1_hits), "net_gain": len(d1_hits) - len(baseline_hits)},
        "integrity": {"candidate_sets_unchanged": True, "historical_rankings_exactly_reproduced": True, "cross_b_rrf_reordering": False, "changed_order_outputs": len(changed), "changed_outputs": changed, "frozen_hashes_before_scoring": frozen_hashes},
        "reference_zip_sha256": sha256(args.reference_zip),
    }
    atomic_json(args.output_dir / "D1_REPLAY_REPORT.json", report)
    (args.output_dir / "D1_REPLAY_REPORT.md").write_text(report_markdown(report), encoding="utf-8")
    print(json.dumps({"event": "D1_SELECTOR_REPLAY_COMPLETE", "baseline_top2": baseline_score["top2"], "d1_top2": d1_score["top2"], "new_solves": report["paired"]["new_solves"], "regressions": report["paired"]["regressions"], "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"D1 replay failed: {exc}", file=sys.stderr)
        raise
