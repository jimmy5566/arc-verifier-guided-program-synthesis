"""Build the matched pre-Speed-V2 30-task timing-control notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = '''import hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path

input_root = Path("/kaggle/input")
source_script = next(input_root.rglob("run_qwen4b_native_augmentation_search.py"), None)
if source_script is None:
    archive = next(input_root.rglob("ARC2.zip"), None)
    if archive is None: raise RuntimeError("OLD Dynamic-B reference source missing")
    extracted = Path("/kaggle/working/old_dynamic_b_source")
    if not (extracted / "ARC2").exists():
        with zipfile.ZipFile(archive) as value: value.extractall(extracted)
    root = extracted / "ARC2"
else:
    root = source_script.parents[1]
manifest = root / "artifacts/speed_v2_frozen30_manifest.json"
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json")
model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
native_config = root / "configs/nvarc_native_846d0198"
config = root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"
out = Path("/kaggle/working/artifacts/speed_v2_old_reference")
out.mkdir(parents=True, exist_ok=True)
candidate, selection, attempts = out / "OLD_candidates_frozen.json", out / "OLD_B_selection_frozen.json", out / "OLD_B_attempts_frozen.json"

def run(command, event, env=None):
    print(json.dumps({"event": event, "command": command}, sort_keys=True), flush=True)
    subprocess.run(command, env=env, check=True)

def freeze_attempts(path, output):
    value = json.loads(path.read_text(encoding="utf-8")); records = {}
    for task_id, record in sorted(value["records"].items()):
        candidates = record["candidates"]; indices = [int(index) for index in record["public_reference_selection"]["attempt_candidate_indices"]]
        rows = [candidates[index]["prediction"] for index in indices]
        records[task_id] = {"attempt_1": rows[0], "attempt_2": rows[1] if len(rows) > 1 else rows[0], "attempt_candidate_indices": indices}
    output.write_text(json.dumps({"status": "B_ATTEMPTS_FROZEN_BEFORE_EXACT_SCORING", "task_ids": sorted(records), "records": records, "selection_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

cohort = json.loads(manifest.read_text(encoding="utf-8")); ids = list(cohort.get("task_ids", ()))
if cohort.get("status") != "COHORT_FROZEN_BEFORE_SPEED_V2_INFERENCE_AND_TARGET_ACCESS" or len(ids) != 30: raise RuntimeError("invalid frozen Speed V2 cohort")
print(json.dumps({"event": "OLD_DYNAMIC_B_REFERENCE_START", "task_count": len(ids), "gpus": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines(), "solutions_opened": False, "source_commit": "8d1b880"}, sort_keys=True), flush=True)
if not candidate.exists():
    run([sys.executable, str(root / "scripts/run_qwen4b_native_augmentation_search.py"), "--cohort", str(manifest), "--config", str(config), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(candidate), "--stage", "external", "--external-augmentation-count", "32", "--external-worker-count", "4", "--search-beams", "1", "--checkpoint-dir", str(out / "checkpoints"), "--resume"], "OLD_DYNAMIC_B_GENERATION", dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"))
if not selection.exists():
    # This is the historical single-GPU B likelihood tail and is intentionally
    # retained in the control, not optimized away.
    run([sys.executable, str(root / "scripts/rerank_native_public_reference_selection.py"), "--frozen", str(candidate), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native_config), "--output", str(selection)], "OLD_DYNAMIC_B_SERIAL_B_SCORING", dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
if not attempts.exists(): freeze_attempts(selection, attempts)
a, b = json.loads(candidate.read_text(encoding="utf-8")), json.loads(selection.read_text(encoding="utf-8"))
if a.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or b.get("status") != "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING" or set(a.get("records", {})) != set(ids) or set(b.get("records", {})) != set(ids): raise RuntimeError("incomplete OLD reference artifact")
report = {{"status": "OLD_DYNAMIC_B_REFERENCE_COMPLETE", "task_count": len(ids), "runtime_seconds": a["runtime_seconds"] + b.get("public_reference_selection_runtime_seconds", 0.0), "generation_runtime_seconds": a["runtime_seconds"], "serial_b_scoring_runtime_seconds": b.get("public_reference_selection_runtime_seconds", 0.0), "worker_ready": a.get("worker_ready"), "attempts": str(attempts), "candidate": str(candidate), "selection": str(selection), "solutions_opened": False}}
(out / "OLD_DYNAMIC_B_REFERENCE_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps({"event": "OLD_DYNAMIC_B_REFERENCE_COMPLETE", **report}, sort_keys=True), flush=True)
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
