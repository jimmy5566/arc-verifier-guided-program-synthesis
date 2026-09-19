"""Prepare an isolated 4xL4 Kaggle Evaluation60 Aug8 experiment.

The staged source is an immutable archive of the requested frozen production
commit plus three experiment-only orchestration scripts.  It contains no ARC
solution files; evaluation targets stay in the Kaggle competition attachment
until the post-freeze scorer executes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "ec5a787ff16a6ff0240a76e77f6060b73e47fec3"
CONFIG = ROOT / "configs" / "QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"
NATIVE_CONFIG = ROOT / "configs" / "nvarc_native_846d0198"
EVALUATION_CHALLENGES = ROOT / "data" / "raw" / "arc-agi_evaluation_challenges.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _task_hash(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def _manifest() -> dict[str, Any]:
    challenges = json.loads(EVALUATION_CHALLENGES.read_text(encoding="utf-8"))
    if len(challenges) != 120:
        raise ValueError(f"expected 120 evaluation tasks, got {len(challenges)}")
    ordered = sorted(challenges)
    selected = sorted(ordered, key=lambda task_id: (hashlib.sha256(task_id.encode()).hexdigest(), task_id))[:60]
    return {
        "experiment_id": "ARC2_EVALUATION60_AUG8_GENERALIZATION_DIAGNOSIS",
        "status": "EVAL60_COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS",
        "source": "arc-agi_evaluation_challenges.json",
        "source_challenge_sha256": _sha256(EVALUATION_CHALLENGES),
        "source_task_count": len(challenges),
        "task_count": 60,
        "task_ids": selected,
        "task_ids_hash": _task_hash(selected),
        "selection_rule": "sort task IDs; compute SHA256(task_id); choose 60 smallest digest values; task_id lexicographic order breaks digest ties",
        "integrity": {"solutions_opened": False, "targets_inspected": False, "selection_is_immutable": True},
        "frozen_inference": {
            "source_commit": SOURCE_COMMIT, "augmentation_count": 8, "search_beams": 1,
            "generation_micro_batch_size": 1, "likelihood_micro_batch_size": 1, "worker_count": 4,
            "model": "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
        },
    }


def _archive(destination: Path) -> None:
    archive = destination.parent / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", SOURCE_COMMIT], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    archive.unlink()


def _notebook_source(dataset_slug: str) -> str:
    return f'''import json, os, subprocess, sys
from pathlib import Path

input_root = Path("/kaggle/input")
root = next(input_root.rglob("run_eval60_aug8_generalization.py")).parents[1]
baseline = next(input_root.rglob("eval60_manifest.json")).parent
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
solutions = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")
model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
native_config = root / "configs/nvarc_native_846d0198"
config = root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"
out = Path("/kaggle/working/artifacts/eval60_aug8")

gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
if len(gpus) != 4 or any("NVIDIA L4" not in line for line in gpus):
    raise RuntimeError(f"requires exactly 4 NVIDIA L4 GPUs, got: {{gpus}}")
if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {{"1", "true", "yes"}}:
    raise RuntimeError("Internet must be disabled")
print(json.dumps({{"event":"EVAL60_AUG8_HARDWARE_GATE","gpus":gpus,"internet":False,"dataset_source":"{dataset_slug}","solutions_opened":False}}, sort_keys=True), flush=True)

freeze = [sys.executable, str(root / "scripts/run_eval60_aug8_generalization.py"),
    "--manifest", str(baseline / "eval60_manifest.json"), "--challenge-path", str(challenge),
    "--model-path", str(model), "--native-config-dir", str(native_config), "--config", str(config), "--output-dir", str(out)]
if subprocess.run(freeze, env={{**os.environ, "CUDA_VISIBLE_DEVICES":"0,1,2,3"}}).returncode:
    raise RuntimeError("Evaluation60 target-blind freeze failed")
required_freeze = [out / "A_candidates_frozen.json", out / "B_support_selection_frozen.json", out / "predictions_frozen.json"]
if any(not item.is_file() for item in required_freeze):
    raise RuntimeError("missing Evaluation60 frozen artifact")
print(json.dumps({{"event":"EVAL60_AUG8_TARGET_BLIND_FREEZE_COMPLETE","solutions_opened":False,"artifacts":[str(item) for item in required_freeze]}}, sort_keys=True), flush=True)

score = [sys.executable, str(root / "scripts/score_eval60_aug8_generalization.py"),
    "--manifest", str(baseline / "eval60_manifest.json"), "--candidates", str(out / "A_candidates_frozen.json"),
    "--selection", str(out / "B_support_selection_frozen.json"), "--predictions", str(out / "predictions_frozen.json"),
    "--challenge-path", str(challenge), "--solutions-path", str(solutions), "--output-dir", str(out)]
if subprocess.run(score, env={{**os.environ, "CUDA_VISIBLE_DEVICES":""}}).returncode:
    raise RuntimeError("Evaluation60 post-freeze scoring failed")
required = required_freeze + [out / "per_task_results.csv", out / "EVAL60_AUG8_REPORT.md", out / "EVAL60_AUG8_REPORT.json"]
if any(not item.is_file() for item in required):
    raise RuntimeError("missing Evaluation60 required output")
print(json.dumps({{"event":"EVAL60_AUG8_COMPLETE","artifacts":[str(item) for item in required]}}, sort_keys=True), flush=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--dataset-slug", default="arc2-eval60-aug8-generalization-source")
    parser.add_argument("--kernel-slug", default="arc2-eval60-aug8-generalization")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite staging: {args.output}")
    # The active research branch may advance after the frozen Aug8 commit.
    # Archive the explicit commit below, rather than conflating the caller's
    # current HEAD with the inference source requested for this experiment.
    subprocess.run(["git", "cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}"], cwd=ROOT, check=True)
    manifest = _manifest()
    dataset_root = args.output / "dataset" / "ARC2"; dataset_root.mkdir(parents=True)
    _archive(dataset_root)
    # ``git archive`` supplied the exact source-commit config and tokenizer
    # contract.  Do not overlay current-worktree versions here.
    for name in ("run_eval60_aug8_generalization.py", "score_eval60_aug8_generalization.py"):
        shutil.copy2(ROOT / "scripts" / name, dataset_root / "scripts" / name)
    baseline = dataset_root / "eval60_aug8_baseline"; baseline.mkdir()
    _write(baseline / "eval60_manifest.json", manifest)
    if list(dataset_root.rglob("*solutions*.json")):
        raise RuntimeError("experiment source dataset must not contain solutions")
    _write(args.output / "dataset" / "dataset-metadata.json", {
        "title": "ARC2 Evaluation60 Aug8 Generalization Source", "subtitle": "Target-blind Aug8 Evaluation60 inference source", 
        "description": "Private source attachment. It contains no ARC solution file.", "id": f"{args.owner}/{args.dataset_slug}", "licenses": [{"name": "other"}],
    })
    kernel = args.output / "kernel"; kernel.mkdir()
    notebook_name = f"{args.kernel_slug}.ipynb"
    _write(kernel / notebook_name, {"cells":[{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[line + "\n" for line in _notebook_source(f"{args.owner}/{args.dataset_slug}").splitlines()]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"},"kaggle":{"accelerator":"nvidiaL4","isGpuEnabled":True,"isInternetEnabled":False,"language":"python","sourceType":"notebook"}},"nbformat":4,"nbformat_minor":4})
    _write(kernel / "kernel-metadata.json", {
        "id": f"{args.owner}/{args.kernel_slug}", "title": "ARC2 Evaluation60 Aug8 Generalization", "code_file": notebook_name,
        "language":"python", "kernel_type":"notebook", "is_private":True, "enable_gpu":True, "enable_tpu":False, "enable_internet":False,
        "keywords":["gpu"], "dataset_sources":[f"{args.owner}/{args.dataset_slug}"], "kernel_sources":[],
        "competition_sources":["arc-prize-2026-arc-agi-2"], "model_sources":["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"], "machine_shape":"NvidiaL4",
    })
    print(json.dumps({"status":"EVAL60_AUG8_STAGING_READY","manifest":manifest,"dataset":str(args.output / "dataset"),"kernel":str(kernel)}, sort_keys=True))


if __name__ == "__main__":
    main()
