"""Score the frozen Eval3 reference-style TTT candidate pools after freezing."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_eval3_reference_ttt import FROZEN_STATUS, _task_hash


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hits(record: dict[str, Any], target: Any) -> list[int]:
    return [index for index, candidate in enumerate(record.get("candidates", ())) if candidate.get("prediction") == target]


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "reference_config", "baseline", "candidates", "challenge_path", "solutions_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    names = ("per_task_results.csv", "EVAL3_REFERENCE_TTT_REPORT.json", "EVAL3_REFERENCE_TTT_REPORT.md")
    if any((args.output_dir / name).exists() for name in names):
        raise FileExistsError("refusing to overwrite a completed Eval3 score")
    manifest, config, baseline, candidates = (_read(args.manifest), _read(args.reference_config), _read(args.baseline), _read(args.candidates))
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "EVAL3_REFERENCE_TTT_COHORT_FROZEN" or len(task_ids) != 3 or manifest.get("task_ids_hash") != _task_hash(task_ids):
        raise ValueError("invalid Eval3 manifest")
    if hashlib.sha256(args.challenge_path.read_bytes()).hexdigest() != manifest.get("source_challenge_sha256"):
        raise ValueError("challenge source differs from frozen manifest")
    if candidates.get("status") != FROZEN_STATUS or set(candidates.get("records", ())) != set(task_ids) or set(baseline.get("records", ())) != set(task_ids):
        raise ValueError("candidate freeze incomplete")
    if int(config.get("rank", -1)) != 256 or int(config.get("alpha", -1)) != 32 or int(config.get("ttt_steps", -1)) != 24:
        raise ValueError("TTT configuration differs from frozen Eval3 contract")
    # Target boundary: candidate pools and checkpoints are complete above.
    solutions = _read(args.solutions_path)
    rows: list[dict[str, Any]] = []; recovered: list[str] = []
    for task_id in task_ids:
        baseline_hits, ttt_hits = _hits(baseline["records"][task_id], solutions[task_id]), _hits(candidates["records"][task_id], solutions[task_id])
        if baseline_hits:
            raise AssertionError(f"{task_id}: immutable baseline is not a pool miss")
        if ttt_hits:
            recovered.append(task_id)
        record, ttt = candidates["records"][task_id], candidates["records"][task_id]["ttt"]
        if not (ttt.get("train_pairs_only") and ttt.get("full_dialogue_assistant_only_loss") and ttt.get("loss_finite") and ttt.get("adapter_updated") and ttt.get("base_model_unchanged")):
            raise ValueError(f"{task_id}: TTT integrity violation")
        optimizer_seconds = float(sum(ttt["step_seconds"]))
        rows.append({"task_id": task_id, "baseline_any_of_k": False, "ttt_any_of_k": bool(ttt_hits), "correct_candidate_indices": json.dumps(ttt_hits), "ttt_optimizer_seconds": optimizer_seconds, "seconds_per_ttt_step": ttt["seconds_per_step"], "generation_seconds": record["generation_seconds"], "unique_candidates": record["unique_candidate_count"], "invalid_candidates": record["invalid_candidate_count"], "peak_allocated_vram_mb": record["peak_allocated_vram_mb"]})
    report = {"experiment_id": "ARC2_REFERENCE_STYLE_TTT_EVAL3_PILOT", "status": "COMPLETE_SCORED_AFTER_CANDIDATE_FREEZE", "BASELINE_ANYK": "0/3", "TTT_ANYK": f"{len(recovered)}/3", "NEW_RECOVERIES": len(recovered), "new_recovery_task_ids": recovered, "TTT_SECONDS_PER_TASK": sum(float(row["ttt_optimizer_seconds"]) for row in rows) / len(rows), "GENERATION_SECONDS_PER_TASK": sum(float(row["generation_seconds"]) for row in rows) / len(rows), "PEAK_VRAM_GB": max(int(row["peak_allocated_vram_mb"]) for row in rows) / 1024.0, "TTT_SIGNAL": "TTT_SIGNAL" if recovered else "NO_TTT_SIGNAL", "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_task_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "EVAL3_REFERENCE_TTT_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "EVAL3_REFERENCE_TTT_REPORT.md").write_text("# Eval3 reference-style TTT pilot\n\n" + "\n".join(f"- {key} = `{value}`" for key, value in report.items() if key in {"BASELINE_ANYK", "TTT_ANYK", "NEW_RECOVERIES", "TTT_SECONDS_PER_TASK", "GENERATION_SECONDS_PER_TASK", "PEAK_VRAM_GB", "TTT_SIGNAL"}) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
