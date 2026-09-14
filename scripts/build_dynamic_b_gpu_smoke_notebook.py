"""Build the fixed, target-blind 8-task Dynamic-B L4x4 engineering smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# Chosen before this smoke from historical runtime telemetry only.  They are
# all development-split training tasks; no correctness or target data informed
# selection.  The mix deliberately includes short and long examples.
TASK_IDS = (
    "f8b3ba0a", "3c9b0459", "c8f0f002", "e3fe1151",
    "996ec1f3", "6f8cd79b", "6d58a25d", "8dae5dfc",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = f'''import hashlib, json, os, subprocess, sys, time
from pathlib import Path

TASK_IDS = {list(TASK_IDS)!r}
input_root = Path("/kaggle/input")
root = next(input_root.rglob("run_qwen4b_native_augmentation_search.py")).parents[1]
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json")
model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
native_config = root / "configs/nvarc_native_846d0198"
config = root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"
out = Path("/kaggle/working/artifacts/dynamic_b_gpu_smoke")
out.mkdir(parents=True, exist_ok=True)
cohort = out / "cohort.json"; candidate = out / "A_candidates_frozen.json"; selection = out / "B_selection_frozen.json"; predictions = out / "B_attempts_frozen.json"; report = out / "GPU_SMOKE_REPORT.json"; checkpoints = out / "generation_checkpoints"
deadline_unix = time.time() + 30 * 60

def run(command, event, env=None):
    print(json.dumps({{"event": event, "command": command}}, sort_keys=True), flush=True)
    subprocess.run(command, env=env, check=True)

def grid(value):
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value): raise ValueError("empty/non-grid attempt")
    width = len(value[0])
    if any(len(row) != width or any(not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row) for row in value): raise ValueError("invalid ARC grid")
    return value

if set(TASK_IDS) - set(json.loads(challenge.read_text(encoding="utf-8"))): raise RuntimeError("smoke tasks absent from training challenges")
if not cohort.exists():
    frozen = {{"experiment_id": "DYNAMIC_B_GPU_SMOKE_V1", "status": "COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS", "source": "development training challenges", "task_count": len(TASK_IDS), "task_ids": TASK_IDS, "task_ids_hash": hashlib.sha256(json.dumps(sorted(TASK_IDS), separators=(",", ":")).encode()).hexdigest(), "selection": "predeclared mixed historical runtime only", "solutions_opened": False}}
    cohort.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps({{"event": "GPU_SMOKE_START", "task_ids": TASK_IDS, "task_count": len(TASK_IDS), "deadline_unix": deadline_unix, "gpus": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines(), "solutions_opened": False}}, sort_keys=True), flush=True)
run([sys.executable, str(root / "scripts/run_qwen4b_native_augmentation_search.py"), "--cohort", str(cohort), "--config", str(config), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(candidate), "--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4", "--search-beams", "1", "--checkpoint-dir", str(checkpoints), "--resume", "--deadline-unix", str(deadline_unix)], "DYNAMIC_B_A_GENERATION", dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"))
run([sys.executable, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(candidate), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(selection), "--deadline-unix", str(deadline_unix)], "DYNAMIC_B_METHOD_B_SELECTION", dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
a = json.loads(candidate.read_text(encoding="utf-8")); b = json.loads(selection.read_text(encoding="utf-8")); records = a.get("records", {{}}); selected = b.get("records", {{}})
if a.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or b.get("status") != "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING" or set(records) != set(TASK_IDS) or set(selected) != set(TASK_IDS): raise RuntimeError("smoke pipeline did not complete all eight tasks")
attempts = {{}}
for task_id in TASK_IDS:
    record = selected[task_id]; candidates = record.get("candidates", []); indices = [int(x) for x in record.get("public_reference_selection", {{}}).get("attempt_candidate_indices", [])]
    if not indices or len(indices) > 2 or any(index < 0 or index >= len(candidates) for index in indices): raise RuntimeError(f"{{task_id}}: invalid Method B attempts")
    values = [candidates[index]["prediction"] for index in indices]
    if len(values) == 1: values.append(values[0])
    for prediction in values:
        if not isinstance(prediction, list) or not prediction: raise RuntimeError(f"{{task_id}}: no test prediction")
        for output in prediction: grid(output)
    attempts[task_id] = {{"attempt_1": values[0], "attempt_2": values[1], "attempt_candidate_indices": indices}}
predictions.write_text(json.dumps({{"status": "B_ATTEMPTS_FROZEN", "task_ids": TASK_IDS, "records": attempts}}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
files = {{path.stem: json.loads(path.read_text(encoding="utf-8")) for path in (checkpoints / "tasks").glob("*.json")}}
if set(files) != set(TASK_IDS) or any(value.get("config_sha256") != a.get("config_sha256") or value.get("record", {{}}).get("task_id") != task_id for task_id, value in files.items()): raise RuntimeError("invalid or incomplete atomic checkpoints")
by_worker = {{str(worker): [] for worker in range(4)}}
for task_id, record in records.items(): by_worker[str(record["worker_id"])].append(task_id)
summary = {{"status": "PASS", "task_ids": TASK_IDS, "task_count": len(records), "unique_task_count": len(set(records)), "duplicate_tasks": [], "worker_gpu_mapping": {{str(item["worker_id"]): {{"physical_gpu_id": item["physical_gpu_id"], "gpu_name": item["gpu_name"], "model_instances": item["model_instances"]}} for item in a["worker_ready"]}}, "tasks_by_worker": by_worker, "per_task_runtime_seconds": {{task_id: records[task_id]["elapsed_seconds"] for task_id in TASK_IDS}}, "checkpoint_validation": "PASS", "attempt_validation": "PASS", "parent_cuda_model_initialized": False, "worker_failures": a["deadline"]["failed_task_ids"], "deadline_skipped": a["deadline"]["skipped_task_ids"], "runtime_seconds": a["runtime_seconds"], "dynamic_scheduling_observed": any(len(values) > 2 for values in by_worker.values()), "warnings": []}}
report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps({{"event": "GPU_SMOKE_COMPLETE", **summary, "report": str(report), "attempts": str(predictions)}}, sort_keys=True), flush=True)
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
