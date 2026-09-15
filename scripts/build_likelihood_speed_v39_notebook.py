"""Build ARC2 Version 39: a likelihood-only GPU micro-batch gate.

Generation deliberately remains micro-batch one in every condition.  The
notebook freezes every B output before it opens ARC solutions, and its choice
of a retained setting is entirely target-blind.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SMOKE_TASK_IDS = (
    "f8b3ba0a", "3c9b0459", "c8f0f002", "e3fe1151",
    "996ec1f3", "6f8cd79b", "6d58a25d", "8dae5dfc",
)
LIKELIHOOD_MICRO_BATCH_SIZES = (1, 2, 4, 8, 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = f'''import hashlib, json, os, subprocess, sys, time, zipfile
from pathlib import Path

SMOKE_TASK_IDS = {list(SMOKE_TASK_IDS)!r}
LIKELIHOOD_MICRO_BATCH_SIZES = {LIKELIHOOD_MICRO_BATCH_SIZES!r}
GENERATION_MICRO_BATCH_SIZE = 1
input_root = Path("/kaggle/input")
source_script = next(input_root.rglob("run_qwen4b_native_augmentation_search.py"), None)
if source_script is None:
    archive = next(input_root.rglob("ARC2.zip"), None)
    if archive is None: raise RuntimeError("ARC2 source input is missing")
    extracted = Path("/kaggle/working/arc2_v39_source")
    if not (extracted / "ARC2").exists():
        with zipfile.ZipFile(archive) as value: value.extractall(extracted)
    root = extracted / "ARC2"
else:
    root = source_script.parents[1]
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json")
solutions = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json")
model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
native_config = root / "configs/nvarc_native_846d0198"
config = root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"
out = Path("/kaggle/working/artifacts/likelihood_speed_v39")
out.mkdir(parents=True, exist_ok=True)

def run(command, event, env=None):
    print(json.dumps({{"event": event, "command": command}}, sort_keys=True), flush=True)
    subprocess.run(command, env=env, check=True)

def write_once(path, value):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

def frozen_cohort(path):
    value = {{"experiment_id": "ARC2_VERSION_39_GPU_LIKELIHOOD_ONLY_SPEED", "status": "COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS", "source": "fixed predeclared 8-task engineering smoke", "task_count": len(SMOKE_TASK_IDS), "task_ids": SMOKE_TASK_IDS, "task_ids_hash": hashlib.sha256(json.dumps(sorted(SMOKE_TASK_IDS), separators=(",", ":")).encode()).hexdigest(), "integrity": {{"solutions_opened": False, "targets_inspected": False}}}}
    if not path.exists(): write_once(path, value)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if loaded != value: raise RuntimeError("Version 39 cohort is not immutable")
    return path

def freeze_attempts(selection_path, output):
    selected = json.loads(selection_path.read_text(encoding="utf-8")); records = {{}}
    for task_id, record in sorted(selected["records"].items()):
        candidates = record["candidates"]
        indices = [int(value) for value in record["public_reference_selection"]["attempt_candidate_indices"]]
        values = [candidates[index]["prediction"] for index in indices]
        records[task_id] = {{"attempt_1": values[0], "attempt_2": values[1] if len(values) > 1 else values[0], "attempt_candidate_indices": indices}}
    write_once(output, {{"status": "B_ATTEMPTS_FROZEN_BEFORE_EXACT_SCORING", "task_ids": sorted(records), "records": records, "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest()}})

def score_after_all_freezes(conditions, output):
    # This is intentionally the one target boundary: every successful
    # likelihood condition has a persisted candidate, selection, and attempt
    # freeze before solutions are imported.
    from arc.io import load_solutions
    targets = load_solutions(solutions)
    result = {{"status": "LIKELIHOOD_V39_SCORED_AFTER_ALL_PREDICTIONS_FROZEN", "solutions_loaded_after_all_successful_prediction_freezes": True, "conditions": {{}}}}
    for batch, paths in sorted(conditions.items()):
        candidate = json.loads(paths["candidate"].read_text(encoding="utf-8")); attempts = json.loads(paths["attempts"].read_text(encoding="utf-8"))
        top1 = two = any_k = 0
        for task_id in SMOKE_TASK_IDS:
            expected = targets[task_id]; record = attempts["records"][task_id]
            top1 += int(record["attempt_1"] == expected)
            two += int(record["attempt_1"] == expected or record["attempt_2"] == expected)
            any_k += int(any(item["prediction"] == expected for item in candidate["records"][task_id]["candidates"]))
        result["conditions"][str(batch)] = {{"top1_exact": top1, "two_attempt_exact": two, "any_of_k_diagnostic": any_k}}
    write_once(output, result)
    return result

def metrics(artifact):
    records = artifact["records"]
    return {{"wall_time_seconds": float(artifact["runtime_seconds"]), "generation_wall_time_seconds": float(artifact.get("generation_wall_seconds_sum", 0.0)), "original_likelihood_time_seconds": float(artifact.get("original_likelihood_seconds_sum", 0.0)), "b_support_scoring_time_seconds": float(artifact.get("b_support_scoring_seconds_sum", 0.0)), "seconds_per_task": float(artifact["runtime_seconds"]) / len(SMOKE_TASK_IDS), "retries": int(artifact.get("retry_count", 0)), "failed_task_ids": list(artifact.get("deadline", {{}}).get("failed_task_ids", ())), "peak_vram_mb": int(artifact.get("peak_allocated_vram_mb", 0))}}

print(json.dumps({{"event": "LIKELIHOOD_V39_START", "gpus": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines(), "source": str(root), "generation_micro_batch_size": GENERATION_MICRO_BATCH_SIZE, "likelihood_micro_batch_sizes": LIKELIHOOD_MICRO_BATCH_SIZES, "solutions_opened": False}}, sort_keys=True), flush=True)
cohort = frozen_cohort(out / "cohort.json")
conditions, failures = {{}}, {{}}
for likelihood_batch_size in LIKELIHOOD_MICRO_BATCH_SIZES:
    condition = out / f"likelihood_batch_{{likelihood_batch_size}}"; condition.mkdir(exist_ok=True)
    candidate = condition / "A_candidates_frozen.json"; selection = condition / "B_selection_frozen.json"; attempts = condition / "B_attempts_frozen.json"
    try:
        if not candidate.exists():
            run([sys.executable, str(root / "scripts/run_qwen4b_native_augmentation_search.py"), "--cohort", str(cohort), "--config", str(config), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(candidate), "--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4", "--search-beams", "1", "--generation-micro-batch-size", "1", "--likelihood-micro-batch-size", str(likelihood_batch_size), "--checkpoint-dir", str(condition / "checkpoints"), "--resume"], f"LIKELIHOOD_BATCH_{{likelihood_batch_size}}_GENERATION", dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"))
        if not selection.exists():
            run([sys.executable, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(candidate), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(selection), "--require-cached-evidence"], f"LIKELIHOOD_BATCH_{{likelihood_batch_size}}_B_SUPPORT_SELECTION", dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        if not attempts.exists(): freeze_attempts(selection, attempts)
        conditions[likelihood_batch_size] = {{"candidate": candidate, "selection": selection, "attempts": attempts}}
        print(json.dumps({{"event": "LIKELIHOOD_BATCH_PREDICTIONS_FROZEN", "likelihood_micro_batch_size": likelihood_batch_size, "solutions_opened": False}}, sort_keys=True), flush=True)
    except subprocess.CalledProcessError as exc:
        failures[str(likelihood_batch_size)] = {{"stage": "subprocess", "returncode": exc.returncode}}
    except Exception as exc:
        failures[str(likelihood_batch_size)] = {{"stage": "freeze_or_selection", "error": f"{{type(exc).__name__}}: {{exc}}"}}

# All target-blind comparisons are frozen before the sole scoring boundary.
comparisons = {{}}
if 1 not in conditions: raise RuntimeError("likelihood batch=1 baseline did not freeze")
baseline = conditions[1]
for batch, paths in sorted(conditions.items()):
    comparison = out / f"likelihood_batch_{{batch}}" / "comparison_vs_batch1.json"
    if not comparison.exists():
        run([sys.executable, str(root / "scripts/compare_speed_v2_artifacts.py"), "--baseline-candidates", str(baseline["candidate"]), "--candidate-candidates", str(paths["candidate"]), "--baseline-selection", str(baseline["selection"]), "--candidate-selection", str(paths["selection"]), "--output", str(comparison)], f"LIKELIHOOD_BATCH_{{batch}}_TARGET_BLIND_COMPARISON", dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    comparisons[batch] = json.loads(comparison.read_text(encoding="utf-8"))
scores = score_after_all_freezes(conditions, out / "SCORED_AFTER_ALL_FREEZES.json")
rows, eligible = [], []
for batch, paths in sorted(conditions.items()):
    artifact = json.loads(paths["candidate"].read_text(encoding="utf-8")); comparison = comparisons[batch]; row = {{"likelihood_micro_batch_size": batch, **metrics(artifact), "candidate_pool_exact_tasks": comparison["candidate_pool_support_exact_tasks"], "candidate_order_exact_tasks": comparison["candidate_order_exact_tasks"], "support_count_exact_tasks": comparison["candidate_pool_support_exact_tasks"], "ranked_candidate_indices_exact_tasks": comparison["ranked_candidate_indices_exact_tasks"], "b_attempt_1_2_exact_tasks": comparison["b_attempt_exact_tasks"], "b_attempt_indices_exact_tasks": comparison["b_attempt_indices_exact_tasks"], "max_abs_likelihood_delta": comparison["max_abs_likelihood_delta"], "accuracy": scores["conditions"][str(batch)]}}
    row["speedup_vs_batch1"] = metrics(json.loads(baseline["candidate"].read_text(encoding="utf-8")))["wall_time_seconds"] / row["wall_time_seconds"]
    row["exact_pass"] = all(row[key] == len(SMOKE_TASK_IDS) for key in ("candidate_pool_exact_tasks", "candidate_order_exact_tasks", "support_count_exact_tasks", "ranked_candidate_indices_exact_tasks", "b_attempt_1_2_exact_tasks", "b_attempt_indices_exact_tasks")) and row["retries"] == 0 and not row["failed_task_ids"] and row["accuracy"] == scores["conditions"]["1"]
    rows.append(row)
    if row["exact_pass"]: eligible.append((row["wall_time_seconds"], batch))
if not eligible: raise RuntimeError("no likelihood micro-batch condition passed the exactness and safety gate")
selected = min(eligible)[1]
report = {{"experiment_id": "ARC2_VERSION_39_GPU_LIKELIHOOD_ONLY_SPEED", "status": "COMPLETE", "task_ids": SMOKE_TASK_IDS, "generation_micro_batch_size": 1, "selected_likelihood_micro_batch_size": selected, "selection_rule": "fastest zero-retry/OOM target-blind exact-equivalent condition", "conditions": rows, "failures": failures, "integrity": {{"generation_config_frozen": True, "target_blind_selection": True, "solutions_opened_only_after_all_successful_prediction_freezes": True, "no_generation_batch_experiment": True}}}}
write_once(out / "LIKELIHOOD_SPEED_V39_REPORT.json", report)
print(json.dumps({{"event": "LIKELIHOOD_V39_COMPLETE", "selected_likelihood_micro_batch_size": selected, "report": str(out / "LIKELIHOOD_SPEED_V39_REPORT.json")}}, sort_keys=True), flush=True)
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
