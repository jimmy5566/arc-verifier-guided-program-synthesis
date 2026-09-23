"""Freeze G1 per-output selections, then score paired Smoke30 diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_eval3_reference_ttt import _read
# Selection is CPU-only and operates on frozen evidence.  Keep this import
# pinned to its actual definition rather than the GPU evidence producer.
from scripts.score_eval60_per_output_rescore import _select
from scripts.score_eval60_per_output_rescore import _output_hits
from scripts.run_smoke30_representation_alignment import FROZEN_STATUS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outputs_anyk(candidates: list[dict[str, Any]], target: list[Any]) -> list[bool]:
    return [any(candidate["prediction"][index] == value for candidate in candidates) for index, value in enumerate(target)]


def _prediction_from_evidence(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = record["candidates"]
    selections: list[dict[str, Any]] = []
    first: list[Any] = []
    second: list[Any] = []
    evidence_by_index = {int(item["test_index"]): item["candidates"] for item in record["per_output_evidence"]}
    expected_indices = set(range(len(candidates[0]["prediction"]))) if candidates else set(range(len(record["per_output_evidence"])))
    if set(evidence_by_index) != expected_indices:
        raise ValueError(f"{record['task_id']}: invalid per-output evidence")
    for index in sorted(expected_indices):
        pool = [{"prediction": candidate["prediction"][index]} for candidate in candidates]
        selection, output1, output2 = _select(evidence_by_index[index], pool)
        selections.append({"test_index": index, **selection})
        first.append(output1); second.append(output2)
    return {"reused_bundle_selection": False, "per_test": selections}, {"attempt_1": first, "attempt_2": second}


def _task_hit(prediction: dict[str, Any], target: list[Any]) -> bool:
    return prediction.get("attempt_1") == target or prediction.get("attempt_2") == target


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "baseline_candidates", "baseline_predictions", "candidates", "challenge_path", "solutions_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    required = ("per_output_selection_frozen.json", "predictions_frozen.json", "paired_results.csv", "SMOKE30_REPRESENTATION_REPORT.json", "SMOKE30_REPRESENTATION_REPORT.md")
    if any((args.output_dir / name).exists() for name in required):
        raise FileExistsError("refusing to overwrite frozen Smoke30 representation scoring")
    manifest, baseline_candidates, baseline_predictions, candidates = map(_read, (args.manifest, args.baseline_candidates, args.baseline_predictions, args.candidates))
    task_ids = list(manifest.get("task_ids", ()))
    if len(task_ids) != 30 or candidates.get("status") != FROZEN_STATUS or candidates.get("task_ids") != task_ids or set(candidates.get("records", ())) != set(task_ids):
        raise ValueError("incomplete target-blind Smoke30 G1 candidate freeze")
    if any(task_id not in baseline_candidates.get("records", {}) for task_id in task_ids) or any(task_id not in baseline_predictions.get("records", {}).get("TTT24", {}) for task_id in task_ids):
        raise ValueError("G0 baseline does not cover frozen Smoke30 cohort")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections: dict[str, Any] = {}; predictions: dict[str, Any] = {}
    for task_id in task_ids:
        selection, prediction = _prediction_from_evidence(candidates["records"][task_id])
        selections[task_id], predictions[task_id] = selection, prediction
    selection_artifact = {"experiment_id": manifest["experiment_id"], "status": "SMOKE30_G1_PER_OUTPUT_SELECTION_FROZEN_BEFORE_EXACT_SCORING", "solutions_opened": False, "task_ids": task_ids, "source_candidate_sha256": _sha256(args.candidates), "records": selections}
    prediction_artifact = {"experiment_id": manifest["experiment_id"], "status": "SMOKE30_G1_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING", "solutions_opened": False, "task_ids": task_ids, "selection_sha256": hashlib.sha256(json.dumps(selection_artifact, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "records": predictions}
    atomic_write_json(args.output_dir / "per_output_selection_frozen.json", selection_artifact)
    atomic_write_json(args.output_dir / "predictions_frozen.json", prediction_artifact)
    # Exact targets are only opened after all G1 selections/predictions exist.
    solutions = _read(args.solutions_path)
    challenge = _read(args.challenge_path)
    if set(task_ids) - set(solutions) or any(len(challenge[task_id]["test"]) != len(solutions[task_id]) for task_id in task_ids):
        raise ValueError("Smoke30 challenge/solution mapping mismatch")
    counts = {key: 0 for key in ("g0_output_anyk", "g1_output_anyk", "g0_top2", "g1_top2", "g0_task_anyk", "g1_task_anyk", "g0_task_top2", "g1_task_top2", "new_output", "lost_output")}
    rows: list[dict[str, Any]] = []; g0_keys: set[str] = set(); g1_keys: set[str] = set(); new_outputs: list[str] = []; regressions: list[str] = []
    g0_unique = g1_unique = g1_invalid = 0
    for task_id in task_ids:
        target = solutions[task_id]
        base_pool = baseline_candidates["records"][task_id]["candidates"]
        g1_pool = candidates["records"][task_id]["candidates"]
        g0_prediction = baseline_predictions["records"]["TTT24"][task_id]
        g1_prediction = predictions[task_id]
        base_anyk, g1_anyk = _outputs_anyk(base_pool, target), _outputs_anyk(g1_pool, target)
        base1, base2 = _output_hits(g0_prediction, target); new1, new2 = _output_hits(g1_prediction, target)
        counts["g0_output_anyk"] += sum(base_anyk); counts["g1_output_anyk"] += sum(g1_anyk)
        counts["g0_top2"] += sum(base2); counts["g1_top2"] += sum(new2)
        base_task_anyk, new_task_anyk = any(item["prediction"] == target for item in base_pool), any(item["prediction"] == target for item in g1_pool)
        counts["g0_task_anyk"] += base_task_anyk; counts["g1_task_anyk"] += new_task_anyk
        counts["g0_task_top2"] += _task_hit(g0_prediction, target); counts["g1_task_top2"] += _task_hit(g1_prediction, target)
        g0_unique += len(base_pool); g1_unique += len(g1_pool); g1_invalid += int(candidates["records"][task_id]["invalid_candidate_count"])
        g0_keys.update(json.dumps(item["prediction"], separators=(",", ":")) for item in base_pool)
        g1_keys.update(json.dumps(item["prediction"], separators=(",", ":")) for item in g1_pool)
        for index, (old_pool, new_pool, old_top2, new_top2) in enumerate(zip(base_anyk, g1_anyk, base2, new2, strict=True)):
            token = f"{task_id}:{index}"
            if new_pool and not old_pool:
                counts["new_output"] += 1; new_outputs.append(token)
            if old_pool and not new_pool:
                counts["lost_output"] += 1; regressions.append(token)
            rows.append({"task_id": task_id, "test_index": index, "g0_output_anyk": old_pool, "g1_output_anyk": new_pool, "g0_top2": old_top2, "g1_top2": new_top2, "candidate_recall_change": "NEW" if new_pool and not old_pool else "LOST" if old_pool and not new_pool else "UNCHANGED"})
    with (args.output_dir / "paired_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    output_total = len(rows); intersection = len(g0_keys & g1_keys); union = len(g0_keys | g1_keys)
    top2_gain = counts["g1_top2"] - counts["g0_top2"]
    net_output_gain = counts["new_output"] - counts["lost_output"]
    if counts["new_output"] >= 4 and top2_gain > 0:
        decision = "REPRESENTATION_ALIGNMENT_STRONG_GO"
    elif counts["new_output"] in {2, 3} and top2_gain > 0:
        decision = "REPRESENTATION_ALIGNMENT_GO"
    elif counts["new_output"] == 1 and net_output_gain > 0:
        decision = "WEAK_SIGNAL"
    else:
        decision = "NO_GAIN"
    task_seconds = sum(float(record["elapsed_seconds"]) for record in candidates["records"].values())
    report = {"experiment_id": manifest["experiment_id"], "status": "COMPLETE_SCORED_AFTER_TARGET_BLIND_FREEZE", "cohort_task_count": 30, "cohort_output_count": output_total, "G0_OUTPUT_ANYK": f"{counts['g0_output_anyk']}/{output_total}", "G1_OUTPUT_ANYK": f"{counts['g1_output_anyk']}/{output_total}", "G0_TASK_ANYK": f"{counts['g0_task_anyk']}/30", "G1_TASK_ANYK": f"{counts['g1_task_anyk']}/30", "G0_TOP2_OUTPUTS": f"{counts['g0_top2']}/{output_total}", "G1_TOP2_OUTPUTS": f"{counts['g1_top2']}/{output_total}", "G0_TOP2_TASKS": f"{counts['g0_task_top2']}/30", "G1_TOP2_TASKS": f"{counts['g1_task_top2']}/30", "NEW_OUTPUT_RECOVERIES": new_outputs, "REGRESSIONS": regressions, "NET_OUTPUT_GAIN": net_output_gain, "TOP2_OUTPUT_GAIN": top2_gain, "COMPLEMENTARY_RECOVERIES": new_outputs, "G0_UNIQUE_CANDIDATES": g0_unique, "G1_UNIQUE_CANDIDATES": g1_unique, "POOL_OVERLAP_JACCARD_GLOBAL": 0.0 if not union else intersection / union, "G1_INVALID_RATE": f"{g1_invalid}/240", "G1_GPU_SECONDS": task_seconds, "G1_WALL_CLOCK_SECONDS": float(candidates["runtime_seconds"]), "G1_SECONDS_PER_TASK": task_seconds / 30.0, "G1_PEAK_VRAM_GB": max(float(record["peak_allocated_vram_mb"]) for record in candidates["records"].values()) / 1024.0, "PROJECTED_240_RUNTIME_SECONDS": float(candidates["runtime_seconds"]) / 30.0 * 240.0, "candidate_generation_count_per_task": 8, "solutions_opened_only_after_predictions_frozen": True, "DECISION": decision}
    atomic_write_json(args.output_dir / "SMOKE30_REPRESENTATION_REPORT.json", report)
    (args.output_dir / "SMOKE30_REPRESENTATION_REPORT.md").write_text("# Smoke30 reference-like representation alignment\n\n" + "\n".join(f"- {key} = `{value}`" for key, value in report.items()) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
