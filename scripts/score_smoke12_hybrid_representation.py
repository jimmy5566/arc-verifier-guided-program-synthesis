"""CPU-only selection freeze and post-freeze scoring for Smoke12 hybrid views."""
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
from scripts.run_smoke12_hybrid_representation import FROZEN_STATUS
from scripts.score_eval60_per_output_rescore import _output_hits, _select


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output_anyk(candidates: list[dict[str, Any]], target: list[Any]) -> list[bool]:
    return [any(candidate["prediction"][index] == value for candidate in candidates) for index, value in enumerate(target)]


def _selection(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = record["candidates"]
    evidence = {int(item["test_index"]): item["candidates"] for item in record["per_output_evidence"]}
    indices = set(range(len(candidates[0]["prediction"]))) if candidates else set(range(len(record["per_output_evidence"])))
    if set(evidence) != indices:
        raise ValueError(f"{record['task_id']}: incomplete per-output evidence")
    selections: list[dict[str, Any]] = []; first: list[Any] = []; second: list[Any] = []
    for index in sorted(indices):
        pool = [{"prediction": candidate["prediction"][index]} for candidate in candidates]
        chosen, output1, output2 = _select(evidence[index], pool)
        selections.append({"test_index": index, **chosen}); first.append(output1); second.append(output2)
    return {"reused_bundle_selection": False, "per_test": selections}, {"attempt_1": first, "attempt_2": second}


def _task_top2(prediction: dict[str, Any], target: list[Any]) -> bool:
    return prediction.get("attempt_1") == target or prediction.get("attempt_2") == target


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "baseline_candidates", "baseline_predictions", "candidates", "challenge_path", "solutions_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    required = ("per_output_selection_frozen.json", "predictions_frozen.json", "paired_results.csv", "SMOKE12_HYBRID_REPORT.json", "SMOKE12_HYBRID_REPORT.md")
    if any((args.output_dir / name).exists() for name in required):
        raise FileExistsError("refusing to overwrite frozen Smoke12 hybrid score")
    manifest, baseline_candidates, baseline_predictions, candidates = map(_read, (args.manifest, args.baseline_candidates, args.baseline_predictions, args.candidates))
    task_ids = list(manifest.get("task_ids", ()))
    if len(task_ids) != 12 or candidates.get("status") != FROZEN_STATUS or candidates.get("task_ids") != task_ids or set(candidates.get("records", ())) != set(task_ids):
        raise ValueError("incomplete target-blind hybrid candidate freeze")
    if any(task_id not in baseline_candidates.get("records", {}) for task_id in task_ids) or any(task_id not in baseline_predictions.get("records", {}).get("TTT24", {}) for task_id in task_ids):
        raise ValueError("G0 baseline does not cover frozen Smoke12 cohort")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections: dict[str, Any] = {}; predictions: dict[str, Any] = {}
    for task_id in task_ids:
        selections[task_id], predictions[task_id] = _selection(candidates["records"][task_id])
    frozen_selection = {"experiment_id": manifest["experiment_id"], "status": "SMOKE12_HYBRID_PER_OUTPUT_SELECTION_FROZEN_BEFORE_EXACT_SCORING", "solutions_opened": False, "task_ids": task_ids, "source_candidate_sha256": _sha256(args.candidates), "records": selections}
    frozen_predictions = {"experiment_id": manifest["experiment_id"], "status": "SMOKE12_HYBRID_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING", "solutions_opened": False, "task_ids": task_ids, "selection_sha256": hashlib.sha256(json.dumps(frozen_selection, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "records": predictions}
    atomic_write_json(args.output_dir / "per_output_selection_frozen.json", frozen_selection)
    atomic_write_json(args.output_dir / "predictions_frozen.json", frozen_predictions)
    # Targets are opened only after all G1 selections and predictions are persisted.
    challenge, solutions = _read(args.challenge_path), _read(args.solutions_path)
    if set(task_ids) - set(solutions) or any(len(challenge[task_id]["test"]) != len(solutions[task_id]) for task_id in task_ids):
        raise ValueError("Smoke12 challenge/solution mapping mismatch")
    counts = {key: 0 for key in ("g0_anyk", "g1_anyk", "g0_top2", "g1_top2", "new", "lost", "g0_task_top2", "g1_task_top2")}
    rows: list[dict[str, Any]] = []; old_keys: set[str] = set(); new_keys: set[str] = set(); new_outputs: list[str] = []; regressions: list[str] = []
    old_unique = new_unique = invalid = 0
    for task_id in task_ids:
        target = solutions[task_id]; old_pool = baseline_candidates["records"][task_id]["candidates"]; new_pool = candidates["records"][task_id]["candidates"]
        old_anyk, new_anyk = _output_anyk(old_pool, target), _output_anyk(new_pool, target)
        _, old_top2 = _output_hits(baseline_predictions["records"]["TTT24"][task_id], target); _, new_top2 = _output_hits(predictions[task_id], target)
        counts["g0_anyk"] += sum(old_anyk); counts["g1_anyk"] += sum(new_anyk); counts["g0_top2"] += sum(old_top2); counts["g1_top2"] += sum(new_top2)
        counts["g0_task_top2"] += _task_top2(baseline_predictions["records"]["TTT24"][task_id], target); counts["g1_task_top2"] += _task_top2(predictions[task_id], target)
        old_unique += len(old_pool); new_unique += len(new_pool); invalid += int(candidates["records"][task_id]["invalid_candidate_count"])
        old_keys.update(json.dumps(item["prediction"], separators=(",", ":")) for item in old_pool); new_keys.update(json.dumps(item["prediction"], separators=(",", ":")) for item in new_pool)
        for index, (old_recall, new_recall, old_selected, new_selected) in enumerate(zip(old_anyk, new_anyk, old_top2, new_top2, strict=True)):
            token = f"{task_id}:{index}"
            if new_recall and not old_recall: counts["new"] += 1; new_outputs.append(token)
            if old_recall and not new_recall: counts["lost"] += 1; regressions.append(token)
            rows.append({"task_id": task_id, "test_index": index, "g0_output_anyk": old_recall, "g1_output_anyk": new_recall, "g0_top2": old_selected, "g1_top2": new_selected, "candidate_recall_change": "NEW" if new_recall and not old_recall else "LOST" if old_recall and not new_recall else "UNCHANGED"})
    with (args.output_dir / "paired_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    output_count = len(rows); net_recall = counts["new"] - counts["lost"]; net_top2 = counts["g1_top2"] - counts["g0_top2"]
    if counts["new"] >= 2 and net_recall > 0 and net_top2 > 0:
        decision = "HYBRID_GO"
    elif net_recall == 1:
        decision = "WEAK_SIGNAL"
    else:
        decision = "NO_GAIN"
    aggregate_seconds = sum(float(record["elapsed_seconds"]) for record in candidates["records"].values())
    union = old_keys | new_keys
    report = {"experiment_id": manifest["experiment_id"], "status": "COMPLETE_SCORED_AFTER_TARGET_BLIND_FREEZE", "cohort_task_count": 12, "cohort_output_count": output_count, "G0_OUTPUT_ANYK": f"{counts['g0_anyk']}/{output_count}", "G1_OUTPUT_ANYK": f"{counts['g1_anyk']}/{output_count}", "G0_TOP2_OUTPUTS": f"{counts['g0_top2']}/{output_count}", "G1_TOP2_OUTPUTS": f"{counts['g1_top2']}/{output_count}", "G0_TOP2_TASKS": f"{counts['g0_task_top2']}/12", "G1_TOP2_TASKS": f"{counts['g1_task_top2']}/12", "NEW_OUTPUT_RECOVERIES": new_outputs, "REGRESSIONS": regressions, "NET_RECALL_GAIN": net_recall, "NET_TOP2_GAIN": net_top2, "POOL_JACCARD_GLOBAL": 0.0 if not union else len(old_keys & new_keys) / len(union), "UNIQUE_CANDIDATES_G0": old_unique, "UNIQUE_CANDIDATES_G1": new_unique, "INVALID_RATE": f"{invalid}/96", "GPU_SECONDS": aggregate_seconds, "WALL_CLOCK_SECONDS": float(candidates["runtime_seconds"]), "SECONDS_PER_TASK": aggregate_seconds / 12.0, "PEAK_VRAM_GB": max(float(record["peak_allocated_vram_mb"]) for record in candidates["records"].values()) / 1024.0, "solutions_opened_only_after_predictions_frozen": True, "DECISION": decision}
    atomic_write_json(args.output_dir / "SMOKE12_HYBRID_REPORT.json", report)
    (args.output_dir / "SMOKE12_HYBRID_REPORT.md").write_text("# Smoke12 hybrid representation\n\n" + "\n".join(f"- {key} = `{value}`" for key, value in report.items()) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
