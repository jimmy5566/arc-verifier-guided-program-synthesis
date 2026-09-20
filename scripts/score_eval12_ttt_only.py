"""Score the frozen Eval12 TTT-only candidates after the target-blind barrier."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


TTT_FROZEN = "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"
BASELINE_FROZEN = "EVAL12_REUSED_AUG8_GREEDY_BASELINE_FROZEN"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_hash(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def _candidate_hits(record: dict[str, Any], target: Any) -> list[int]:
    return [index for index, candidate in enumerate(record.get("candidates", ())) if candidate.get("prediction") == target]


def _validate_before_solution_access(manifest: dict[str, Any], config: dict[str, Any], baseline: dict[str, Any], ttt: dict[str, Any], challenge_path: Path) -> list[str]:
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "EVAL12_COHORT_FROZEN_BEFORE_TTT_GENERATION" or len(task_ids) != 12 or len(task_ids) != len(set(task_ids)) or manifest.get("task_ids_hash") != _task_hash(task_ids):
        raise ValueError("invalid immutable Eval12 manifest")
    if _sha256(challenge_path) != manifest.get("source_challenge_sha256"):
        raise ValueError("evaluation challenge differs from frozen Eval12 source")
    expected_ttt = {"train_pairs_only": True, "rank": 8, "alpha": 16, "steps": 24, "learning_rate": 5e-4, "target_suffixes": ["q_proj", "v_proj"]}
    if config.get("status") != "FROZEN_BEFORE_TTT_GENERATION" or {key: config.get("ttt", {}).get(key) for key in expected_ttt} != expected_ttt:
        raise ValueError("TTT configuration differs from the frozen contract")
    if any(int(config.get(key, -1)) != value for key, value in {"augmentation_count": 8, "search_beams": 1, "generation_micro_batch_size": 1, "likelihood_micro_batch_size": 1, "worker_count": 4}.items()):
        raise ValueError("Eval12 generation settings differ from the frozen contract")
    expected = set(task_ids)
    if baseline.get("status") != BASELINE_FROZEN or baseline.get("task_ids_hash") != manifest["task_ids_hash"] or set(baseline.get("records", ())) != expected:
        raise ValueError("invalid reused baseline candidate pool")
    if ttt.get("status") != TTT_FROZEN or not ttt.get("ttt_enabled") or ttt.get("task_ids_hash") != manifest["task_ids_hash"] or int(ttt.get("stage_augmentation_count", -1)) != 8 or int(ttt.get("stage_worker_count", -1)) != 4 or int(ttt.get("search", {}).get("beams_per_augmentation", -1)) != 1 or set(ttt.get("records", ())) != expected:
        raise ValueError("incomplete or non-frozen TTT candidate artifact")
    for task_id in task_ids:
        baseline_record, ttt_record = baseline["records"][task_id], ttt["records"][task_id]
        if not isinstance(baseline_record.get("candidates"), list) or not baseline_record["candidates"]:
            raise ValueError(f"{task_id}: empty baseline candidate pool")
        metrics = ttt_record.get("ttt", {})
        if ttt_record.get("status") != "SUCCESS" or not isinstance(ttt_record.get("candidates"), list) or not ttt_record["candidates"]:
            raise ValueError(f"{task_id}: TTT candidate generation incomplete")
        if {key: metrics.get(key) for key in ("rank", "alpha", "steps")} != {"rank": 8, "alpha": 16, "steps": 24} or metrics.get("training_pairs_only") is not True or metrics.get("base_model_unchanged") is not True or metrics.get("adapter_updated") is not True or metrics.get("loss_finite") is not True:
            raise ValueError(f"{task_id}: TTT safety/contract validation failed")
    return task_ids


def _classification(recovered: int) -> str:
    if recovered >= 3:
        return "TTT_WORKS"
    if recovered:
        return "WEAK_TTT_GAIN"
    return "TTT_FAILURE"


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "ttt_config", "baseline", "ttt", "challenge_path", "solutions_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if any((args.output_dir / name).exists() for name in ("per_task_results.csv", "EVAL12_TTT_REPORT.json", "EVAL12_TTT_REPORT.md")):
        raise FileExistsError("refusing to overwrite a completed Eval12 TTT score")
    manifest, config, baseline, ttt = _read(args.manifest), _read(args.ttt_config), _read(args.baseline), _read(args.ttt)
    task_ids = _validate_before_solution_access(manifest, config, baseline, ttt, args.challenge_path)
    # Sole target boundary: all candidate artifacts and their TTT safety data
    # have passed the complete immutable freeze gate above.
    solutions = _read(args.solutions_path)
    if set(task_ids) - set(solutions):
        raise ValueError("evaluation solutions omit an Eval12 task")
    rows: list[dict[str, Any]] = []
    baseline_hits: dict[str, list[int]] = {}
    ttt_hits: dict[str, list[int]] = {}
    for task_id in task_ids:
        target = solutions[task_id]
        baseline_hits[task_id] = _candidate_hits(baseline["records"][task_id], target)
        ttt_hits[task_id] = _candidate_hits(ttt["records"][task_id], target)
        record = ttt["records"][task_id]
        rows.append({"task_id": task_id, "baseline_any_of_k": bool(baseline_hits[task_id]), "baseline_correct_candidate_indices": json.dumps(baseline_hits[task_id]), "ttt_any_of_k": bool(ttt_hits[task_id]), "ttt_correct_candidate_indices": json.dumps(ttt_hits[task_id]), "ttt_unique_candidate_count": int(record.get("unique_candidate_count", len(record["candidates"]))), "ttt_invalid_candidate_count": int(record.get("invalid_candidate_count", 0)), "ttt_seconds": float(record["ttt"]["seconds"]), "generation_seconds": float(record.get("generation_seconds", 0.0)), "task_elapsed_seconds": float(record.get("elapsed_seconds", 0.0)), "ttt_adapter_update_l1": float(record["ttt"]["adapter_update_l1"])})
    if any(baseline_hits.values()):
        raise AssertionError("Eval12 violates its immutable all-baseline-pool-miss definition")
    recovered = [task_id for task_id in task_ids if ttt_hits[task_id]]
    runtime = float(ttt.get("runtime_seconds", sum(row["task_elapsed_seconds"] for row in rows)))
    report = {
        "experiment_id": "ARC2_EVAL12_TTT_ONLY_DIAGNOSTIC",
        "status": "COMPLETE_SCORED_AFTER_TTT_CANDIDATES_FROZEN",
        "protocol": "Train-pair-only task-local LoRA and greedy Aug8 candidate generation; evaluation solutions opened only by this post-freeze scorer.",
        "task_ids": task_ids,
        "task_ids_hash": manifest["task_ids_hash"],
        "source_challenge_sha256": manifest["source_challenge_sha256"],
        "artifact_sha256": {"manifest": _sha256(args.manifest), "ttt_config": _sha256(args.ttt_config), "baseline": _sha256(args.baseline), "ttt_candidates": _sha256(args.ttt)},
        "baseline_any_of_k": 0,
        "ttt_any_of_k": len(recovered),
        "new_recoveries": len(recovered),
        "new_recovery_task_ids": recovered,
        "total_runtime_seconds": runtime,
        "mean_ttt_overhead_seconds_per_task": sum(row["ttt_seconds"] for row in rows) / len(rows),
        "mean_task_runtime_seconds": sum(row["task_elapsed_seconds"] for row in rows) / len(rows),
        "mean_unique_candidates_per_task": sum(row["ttt_unique_candidate_count"] for row in rows) / len(rows),
        "total_invalid_candidates": sum(row["ttt_invalid_candidate_count"] for row in rows),
        "classification": _classification(len(recovered)),
        "fixed_ttt": config["ttt"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_task_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "EVAL12_TTT_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "EVAL12_TTT_REPORT.md").write_text(
        "# Eval12 TTT-only diagnostic\n\n"
        f"- Cohort hash: `{manifest['task_ids_hash']}`\n"
        "- The scorer validated the complete candidate freeze and TTT contract before opening evaluation solutions.\n"
        f"- BASELINE_ANYK=0/12\n- TTT_ANYK={report['ttt_any_of_k']}/12\n- NEW_RECOVERIES={report['new_recoveries']}\n"
        f"- TOTAL_RUNTIME={runtime:.3f}s\n- TTT_OVERHEAD_PER_TASK={report['mean_ttt_overhead_seconds_per_task']:.3f}s\n"
        f"- Classification: **{report['classification']}**\n",
        encoding="utf-8",
    )
    print(json.dumps({"BASELINE_ANYK": "0/12", "TTT_ANYK": f"{report['ttt_any_of_k']}/12", "NEW_RECOVERIES": report["new_recoveries"], "TOTAL_RUNTIME": runtime, "TTT_OVERHEAD_PER_TASK": report["mean_ttt_overhead_seconds_per_task"], "CLASSIFICATION": report["classification"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
