"""Post-freeze scorer for the Eval6 public-reference-style TTT pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


FROZEN = "EVAL6_REFERENCE_TTT_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_hash(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def _hits(record: dict[str, Any], target: Any) -> list[int]:
    return [index for index, candidate in enumerate(record.get("candidates", ())) if candidate.get("prediction") == target]


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "reference_config", "baseline", "candidates", "challenge_path", "solutions_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if any((args.output_dir / name).exists() for name in ("per_task_results.csv", "EVAL6_REFERENCE_TTT_REPORT.json", "EVAL6_REFERENCE_TTT_REPORT.md")):
        raise FileExistsError("refusing to overwrite a completed Eval6 reference TTT score")
    manifest, config, baseline, candidates = _read(args.manifest), _read(args.reference_config), _read(args.baseline), _read(args.candidates)
    ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "EVAL6_REFERENCE_TTT_COHORT_FROZEN" or len(ids) != 6 or manifest.get("task_ids_hash") != _task_hash(ids):
        raise ValueError("invalid frozen Eval6 manifest")
    if hashlib.sha256(args.challenge_path.read_bytes()).hexdigest() != manifest.get("source_challenge_sha256"):
        raise ValueError("evaluation challenge differs from frozen cohort source")
    if config.get("style") != "public_nvarc_documented_ttt" or int(config.get("rank", -1)) != 256 or int(config.get("alpha", -1)) != 32:
        raise ValueError("reference-style TTT config is not frozen")
    if set(baseline.get("records", ())) != set(ids) or set(candidates.get("records", ())) != set(ids) or candidates.get("status") != FROZEN or candidates.get("task_ids_hash") != manifest["task_ids_hash"]:
        raise ValueError("candidate freeze incomplete")
    for task_id in ids:
        metrics = candidates["records"][task_id].get("ttt", {})
        if metrics.get("style") != "public_nvarc_documented_ttt" or int(metrics.get("rank", -1)) != 256 or int(metrics.get("alpha", -1)) != 32 or metrics.get("train_pairs_only") is not True or metrics.get("full_dialogue_assistant_only_loss") is not True or metrics.get("loss_finite") is not True:
            raise ValueError(f"{task_id}: reference-style TTT safety/config freeze failed")
    # Sole target boundary: all six target-blind candidate records passed the
    # complete immutable reference-TTT contract above.
    solutions = _read(args.solutions_path)
    if set(ids) - set(solutions):
        raise ValueError("solutions omit Eval6 task")
    rows: list[dict[str, Any]] = []
    recovered: list[str] = []
    for task_id in ids:
        base = _hits(baseline["records"][task_id], solutions[task_id])
        found = _hits(candidates["records"][task_id], solutions[task_id])
        if base:
            raise AssertionError("Eval6 baseline is not all pool misses")
        if found:
            recovered.append(task_id)
        record, metrics = candidates["records"][task_id], candidates["records"][task_id]["ttt"]
        rows.append({"task_id": task_id, "baseline_any_of_k": False, "reference_ttt_any_of_k": bool(found), "correct_candidate_indices": json.dumps(found), "first_loss": metrics["first_loss"], "last_loss": metrics["last_loss"], "ttt_seconds": metrics["seconds"], "adapter_parameter_count": metrics["adapter_parameter_count"], "generated_candidates": record["generated_candidate_count"], "unique_candidates": record["unique_candidate_count"], "invalid_candidates": record["invalid_candidate_count"], "generation_seconds": record["generation_seconds"], "elapsed_seconds": record["elapsed_seconds"], "peak_allocated_vram_mb": record["peak_allocated_vram_mb"]})
    signal = "STRONG_TTT_SIGNAL" if len(recovered) >= 2 else "WEAK_TTT_SIGNAL" if recovered else "NO_TTT_SIGNAL"
    report = {"experiment_id": "ARC2_EVAL6_REFERENCE_TTT", "status": "COMPLETE_SCORED_AFTER_REFERENCE_TTT_CANDIDATES_FROZEN", "protocol": "Documented public NVARC-style TTT followed by unchanged Aug8 greedy generation; solutions were opened only by this scorer.", "task_ids": ids, "task_ids_hash": manifest["task_ids_hash"], "BASELINE_ANYK": 0, "STRONG_TTT_ANYK": len(recovered), "NEW_RECOVERIES": len(recovered), "new_recovery_task_ids": recovered, "TOTAL_RUNTIME": candidates["runtime_seconds"], "TTT_SECONDS": sum(float(row["ttt_seconds"]) for row in rows), "GENERATION_SECONDS": sum(float(row["generation_seconds"]) for row in rows), "PEAK_VRAM_MB": max(int(row["peak_allocated_vram_mb"]) for row in rows), "signal": signal, "artifact_sha256": {"candidates": _sha256(args.candidates), "baseline": _sha256(args.baseline), "config": _sha256(args.reference_config)}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_task_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "EVAL6_REFERENCE_TTT_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "EVAL6_REFERENCE_TTT_REPORT.md").write_text("# Eval6 reference-style TTT pilot\n\n" + "\n".join(f"- {key} = `{value}`" for key, value in report.items() if key in {"BASELINE_ANYK", "STRONG_TTT_ANYK", "NEW_RECOVERIES", "TOTAL_RUNTIME", "TTT_SECONDS", "GENERATION_SECONDS", "PEAK_VRAM_MB", "signal"}) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("BASELINE_ANYK", "STRONG_TTT_ANYK", "NEW_RECOVERIES", "TOTAL_RUNTIME", "TTT_SECONDS", "GENERATION_SECONDS", "PEAK_VRAM_MB", "signal")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
