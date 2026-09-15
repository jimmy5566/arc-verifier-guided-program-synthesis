"""Build the one-shot low-cost Speed V2 Kaggle experiment notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SMOKE_TASK_IDS = (
    "f8b3ba0a", "3c9b0459", "c8f0f002", "e3fe1151",
    "996ec1f3", "6f8cd79b", "6d58a25d", "8dae5dfc",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = f'''import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

SMOKE_TASK_IDS = {list(SMOKE_TASK_IDS)!r}
MICRO_BATCH_SIZES = (1, 2, 4, 8)
input_root = Path("/kaggle/input")
source_script = next(input_root.rglob("compare_speed_v2_artifacts.py"), None)
if source_script is None:
    # Kaggle CLI stores source directories as an archive.  Extract only to
    # /kaggle/working, then execute exclusively from that mutable copy.
    source_archive = next(input_root.rglob("ARC2.zip"), None)
    if source_archive is None: raise RuntimeError("Speed V2 source is neither mounted nor archived")
    extracted = Path("/kaggle/working/speed_v2_source")
    if not (extracted / "ARC2").exists():
        with zipfile.ZipFile(source_archive) as archive: archive.extractall(extracted)
    root = extracted / "ARC2"
else:
    root = source_script.parents[1]
baselines = next(input_root.rglob("old_smoke_A.json")).parent
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json")
solutions = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json")
model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
native_config = root / "configs/nvarc_native_846d0198"
config = root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"
frozen30 = root / "artifacts/speed_v2_frozen30_manifest.json"
out = Path("/kaggle/working/artifacts/speed_v2")
out.mkdir(parents=True, exist_ok=True)

def run(command, event, env=None):
    print(json.dumps({{"event": event, "command": command}}, sort_keys=True), flush=True)
    subprocess.run(command, env=env, check=True)

def frozen_cohort(path, task_ids, experiment_id, source):
    if not path.exists():
        value = {{"experiment_id": experiment_id, "status": "COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS", "source": source, "task_count": len(task_ids), "task_ids": task_ids, "task_ids_hash": hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest(), "integrity": {{"solutions_opened": False, "targets_inspected": False}}}}
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return path

def b_attempt_freeze(selection_path, output):
    selected = json.loads(selection_path.read_text(encoding="utf-8")); records = selected.get("records", {{}})
    result = {{}}
    for task_id, record in sorted(records.items()):
        candidates = record["candidates"]; indices = [int(x) for x in record["public_reference_selection"]["attempt_candidate_indices"]]
        values = [candidates[index]["prediction"] for index in indices]
        result[task_id] = {{"attempt_1": values[0], "attempt_2": values[1] if len(values) > 1 else values[0], "attempt_candidate_indices": indices}}
    output.write_text(json.dumps({{"status": "B_ATTEMPTS_FROZEN_BEFORE_EXACT_SCORING", "task_ids": sorted(result), "records": result, "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest()}}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

def subset(source_path, task_ids, output):
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if set(task_ids) - set(source.get("records", {{}})): raise RuntimeError("old frozen60 artifact misses Speed V2 task")
    source["records"] = {{task_id: source["records"][task_id] for task_id in task_ids}}
    source["task_ids_hash"] = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    source["stage_task_count"] = len(task_ids)
    # A full-job wall time is not a wall time for an arbitrary subset of its
    # dynamically scheduled tasks.  The predictions remain reusable, but a
    # matched OLD timing control is required before claiming a speedup.
    source["runtime_seconds"] = None
    source["runtime_scope"] = {{"available": False, "reason": "subset_of_larger_dynamic_run"}}
    output.write_text(json.dumps(source, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

print(json.dumps({{"event": "SPEED_V2_START", "source": str(root), "gpus": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines(), "solutions_opened": False, "phase1_task_ids": SMOKE_TASK_IDS}}, sort_keys=True), flush=True)

# Phase 1: exactly four 8-task runs.  No ARC solutions are loaded anywhere in
# this phase; batch selection depends only on speed, OOM/retry state, and the
# target-blind candidate/B-attempt equivalence reports below.
phase1 = out / "phase1"; phase1.mkdir(exist_ok=True)
smoke_cohort = frozen_cohort(phase1 / "cohort.json", SMOKE_TASK_IDS, "ARC2_SPEED_V2_8_TASK_BATCH_GATE", "fixed predeclared engineering smoke")
phase1_failures = {{}}
for batch_size in MICRO_BATCH_SIZES:
    condition = phase1 / f"batch_{{batch_size}}"; condition.mkdir(exist_ok=True)
    candidate, selection = condition / "A_candidates_frozen.json", condition / "B_selection_frozen.json"
    checkpoints, comparison = condition / "checkpoints", condition / "comparison_vs_batch1.json"
    if not candidate.exists():
        try:
            run([sys.executable, str(root / "scripts/run_qwen4b_native_augmentation_search.py"), "--cohort", str(smoke_cohort), "--config", str(config), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(candidate), "--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4", "--search-beams", "1", "--generation-micro-batch-size", str(batch_size), "--likelihood-micro-batch-size", str(batch_size), "--checkpoint-dir", str(checkpoints), "--resume"], f"PHASE1_BATCH_{{batch_size}}_GENERATION", dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"))
        except subprocess.CalledProcessError as exc:
            phase1_failures[str(batch_size)] = {{"stage": "generation", "returncode": exc.returncode}}
            (condition / "FAILED.json").write_text(json.dumps(phase1_failures[str(batch_size)], indent=2, sort_keys=True) + "\\n", encoding="utf-8")
            print(json.dumps({{"event": "PHASE1_BATCH_FAILED", "batch_size": batch_size, **phase1_failures[str(batch_size)]}}, sort_keys=True), flush=True)
            continue
    if not selection.exists():
        try:
            run([sys.executable, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(candidate), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(selection)], f"PHASE1_BATCH_{{batch_size}}_B_SELECTION", dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        except subprocess.CalledProcessError as exc:
            phase1_failures[str(batch_size)] = {{"stage": "cpu_selection", "returncode": exc.returncode}}
            (condition / "FAILED.json").write_text(json.dumps(phase1_failures[str(batch_size)], indent=2, sort_keys=True) + "\\n", encoding="utf-8")
            continue
    baseline_candidate = phase1 / "batch_1" / "A_candidates_frozen.json" if batch_size != 1 else baselines / "old_smoke_A.json"
    baseline_selection = phase1 / "batch_1" / "B_selection_frozen.json" if batch_size != 1 else baselines / "old_smoke_B.json"
    if not comparison.exists():
        try:
            run([sys.executable, str(root / "scripts/compare_speed_v2_artifacts.py"), "--baseline-candidates", str(baseline_candidate), "--candidate-candidates", str(candidate), "--baseline-selection", str(baseline_selection), "--candidate-selection", str(selection), "--output", str(comparison)], f"PHASE1_BATCH_{{batch_size}}_TARGET_BLIND_COMPARISON", dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        except subprocess.CalledProcessError as exc:
            phase1_failures[str(batch_size)] = {{"stage": "target_blind_comparison", "returncode": exc.returncode}}
            (condition / "FAILED.json").write_text(json.dumps(phase1_failures[str(batch_size)], indent=2, sort_keys=True) + "\\n", encoding="utf-8")

comparisons = {{batch_size: json.loads((phase1 / f"batch_{{batch_size}}" / "comparison_vs_batch1.json").read_text(encoding="utf-8")) for batch_size in MICRO_BATCH_SIZES if (phase1 / f"batch_{{batch_size}}" / "comparison_vs_batch1.json").exists()}}
eligible = []
for batch_size, report in comparisons.items():
    runtime = report["candidate_runtime"]
    exact = report["candidate_pool_support_exact_tasks"] == len(SMOKE_TASK_IDS) and report["b_attempt_exact_tasks"] == len(SMOKE_TASK_IDS)
    safe = runtime["retries"] == 0 and not runtime["failed_task_ids"]
    if exact and safe: eligible.append((runtime["wall_seconds"], batch_size))
if not eligible: raise RuntimeError("no Speed V2 micro-batch setting preserved target-blind 8-task candidate/B outputs")
selected_batch = min(eligible)[1]
selection_gate = {{"status": "SPEED_V2_BATCH_CONFIGURATION_FROZEN_TARGET_BLIND", "selected_batch_size": selected_batch, "eligible": [batch for _runtime, batch in sorted(eligible)], "comparisons": {{str(batch): {{key: comparisons[batch][key] for key in ("candidate_pool_support_exact_tasks", "candidate_order_exact_tasks", "b_attempt_exact_tasks", "max_abs_likelihood_delta", "baseline_runtime", "candidate_runtime")}} for batch in comparisons}}, "failed_or_rejected_conditions": phase1_failures, "solutions_opened": False}}
(phase1 / "selected_config.json").write_text(json.dumps(selection_gate, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps({{"event": "PHASE1_SELECTED", "batch_size": selected_batch, "solutions_opened": False}}, sort_keys=True), flush=True)

# Phase 3: one and only one NEW 30-task confirmation, using the batch choice
# frozen above.  OLD is a reusable deterministic subset of Frozen60, so it is
# copied locally rather than regenerated.
phase3 = out / "phase3"; phase3.mkdir(exist_ok=True)
cohort = json.loads(frozen30.read_text(encoding="utf-8")); task_ids = list(cohort["task_ids"])
if cohort.get("status") != "COHORT_FROZEN_BEFORE_SPEED_V2_INFERENCE_AND_TARGET_ACCESS" or len(task_ids) != 30: raise RuntimeError("invalid immutable Speed V2 frozen30 manifest")
old_candidate, old_selection = phase3 / "OLD_candidates_frozen.json", phase3 / "OLD_B_selection_frozen.json"
if not old_candidate.exists(): subset(baselines / "old_frozen60_A.json", task_ids, old_candidate)
if not old_selection.exists(): subset(baselines / "old_frozen60_B.json", task_ids, old_selection)
new_candidate, new_selection = phase3 / "NEW_candidates_frozen.json", phase3 / "NEW_B_selection_frozen.json"
if not new_candidate.exists():
    run([sys.executable, str(root / "scripts/run_qwen4b_native_augmentation_search.py"), "--cohort", str(frozen30), "--config", str(config), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(new_candidate), "--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4", "--search-beams", "1", "--generation-micro-batch-size", str(selected_batch), "--likelihood-micro-batch-size", str(selected_batch), "--checkpoint-dir", str(phase3 / "checkpoints"), "--resume"], "PHASE3_NEW_SPEED_V2_GENERATION", dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"))
if not new_selection.exists():
    run([sys.executable, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(new_candidate), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(new_selection)], "PHASE3_NEW_B_SELECTION", dict(os.environ, CUDA_VISIBLE_DEVICES=""))
comparison = phase3 / "OLD_vs_NEW_target_blind_comparison.json"
if not comparison.exists():
    run([sys.executable, str(root / "scripts/compare_speed_v2_artifacts.py"), "--baseline-candidates", str(old_candidate), "--candidate-candidates", str(new_candidate), "--baseline-selection", str(old_selection), "--candidate-selection", str(new_selection), "--output", str(comparison)], "PHASE3_TARGET_BLIND_OLD_NEW_COMPARISON", dict(os.environ, CUDA_VISIBLE_DEVICES=""))
old_attempts, new_attempts = phase3 / "OLD_B_attempts_frozen.json", phase3 / "NEW_B_attempts_frozen.json"
if not old_attempts.exists(): b_attempt_freeze(old_selection, old_attempts)
if not new_attempts.exists(): b_attempt_freeze(new_selection, new_attempts)
print(json.dumps({{"event": "PHASE3_NEW_PREDICTIONS_FROZEN", "new_attempts": str(new_attempts), "solutions_opened": False}}, sort_keys=True), flush=True)

# This is deliberately the first and only target boundary in the notebook.
score = phase3 / "OLD_NEW_SCORED_AFTER_FREEZE.json"
if not score.exists():
    run([sys.executable, str(root / "scripts/score_speed_v2_frozen30.py"), "--manifest", str(frozen30), "--old-candidates", str(old_candidate), "--old-selection", str(old_selection), "--new-candidates", str(new_candidate), "--new-selection", str(new_selection), "--solutions-path", str(solutions), "--output", str(score)], "PHASE3_POST_FREEZE_SCORING")
old_new = json.loads(comparison.read_text(encoding="utf-8")); scored = json.loads(score.read_text(encoding="utf-8"))
old_wall, new_wall = old_new["baseline_runtime"]["wall_seconds"], old_new["candidate_runtime"]["wall_seconds"]
if old_wall is None or new_wall is None:
    raise RuntimeError("Speed V2 wall-time comparison requires matched full-run OLD and NEW artifacts")
summary = {{"status": "SPEED_V2_COMPLETE", "selected_batch_size": selected_batch, "old_runtime_seconds": old_wall, "new_runtime_seconds": new_wall, "speedup": old_wall / new_wall if new_wall else None, "runtime_reduction_pct": (1 - new_wall / old_wall) * 100 if old_wall else None, "candidate_pool_support_exact_tasks": old_new["candidate_pool_support_exact_tasks"], "b_attempt_exact_tasks": old_new["b_attempt_exact_tasks"], "old_two_attempt_accuracy": scored["old"]["two_attempt_exact"], "new_two_attempt_accuracy": scored["new"]["two_attempt_exact"], "phase1": selection_gate, "per_gpu_load": old_new["candidate_runtime"]["worker_load"], "peak_vram_mb": old_new["candidate_runtime"]["peak_allocated_vram_mb"], "retries": old_new["candidate_runtime"]["retries"], "solutions_loaded_only_after_new_predictions_frozen": True}}
(out / "SPEED_V2_REPORT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps({{"event": "SPEED_V2_COMPLETE", **summary, "report": str(out / "SPEED_V2_REPORT.json")}}, sort_keys=True), flush=True)
'''
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
