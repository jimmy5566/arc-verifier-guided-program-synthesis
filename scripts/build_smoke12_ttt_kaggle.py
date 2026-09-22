"""Build a small immutable Kaggle package for one Smoke12 TTT condition.

The builder is deliberately experiment-only: it freezes no ARC solutions,
uses the existing Eval60 challenge, and writes candidate pools only.  S0 and
S3 are already frozen CPU artifacts; this package supports exactly S1 or S2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments" / "protocols" / "SMOKE12_TTT_COST_ACCURACY_R1_FREEZE.json"
REFERENCE_CONFIG = ROOT / "artifacts" / "eval60_reference_ttt_4gpu_kernel_v5_output" / "artifacts" / "eval60_reference_ttt_4gpu" / "ttt_config_frozen.json"

_CONDITIONS = {
    "ttt24_beam2": {
        "label": "s1",
        "ttt_steps": 24,
        "decode": "beam",
        "beam_width": 2,
        "estimated_wall_minutes": 40,
        "estimated_gpu_seconds": 7475.149,
    },
    "ttt48_greedy": {
        "label": "s2",
        "ttt_steps": 48,
        "decode": "greedy",
        "beam_width": 1,
        "estimated_wall_minutes": 27,
        "estimated_gpu_seconds": 4730.888,
    },
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _cohort(freeze: dict[str, Any]) -> tuple[list[str], str]:
    selection = freeze.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Smoke12 freeze has no selection")
    task_ids = list(selection.get("task_ids", ()))
    task_ids_hash = selection.get("task_ids_hash")
    calculated = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    if len(task_ids) != 12 or len(set(task_ids)) != 12 or task_ids_hash != calculated:
        raise ValueError("invalid frozen Smoke12 cohort")
    return task_ids, calculated


def _runner_config(condition: str, reference: dict[str, Any]) -> dict[str, Any]:
    if condition not in _CONDITIONS:
        raise ValueError(f"unsupported Smoke12 condition: {condition}")
    spec = _CONDITIONS[condition]
    return {
        "reference_ttt_base": reference,
        "conditions": {
            condition: {
                "ttt_steps": spec["ttt_steps"],
                "decode": spec["decode"],
                "beam_width": spec["beam_width"],
            }
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _notebook(*, dataset_slug: str, condition: str, source_commit: str) -> dict[str, Any]:
    code = f'''import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

dataset = Path("/kaggle/input/datasets/jimmy5566/{dataset_slug}")
required = {{
    "freeze": dataset / "freeze_manifest.json",
    "runner_config": dataset / "runner_experiment_config.json",
    "experiment": dataset / "experiment_config.json",
}}
missing = [name for name, path in required.items() if not path.is_file()]
if missing:
    raise RuntimeError(f"Smoke12 package attachment is incomplete: {{missing}}")
config = json.loads(required["experiment"].read_text(encoding="utf-8"))
for name in ("freeze", "runner_config"):
    if sha256_file(required[name]) != config["input_sha256"][name]:
        raise RuntimeError(f"Smoke12 {{name}} hash mismatch")
if config["condition"] != "{condition}" or config["source_commit"] != "{source_commit}":
    raise RuntimeError("Smoke12 frozen execution identity mismatch")
if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {{"1", "true", "yes"}}:
    raise RuntimeError("Internet must be disabled")
gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
if len(gpus) != 4 or any("NVIDIA L4" not in row for row in gpus):
    raise RuntimeError(f"Smoke12 requires exactly four NVIDIA L4 GPUs: {{gpus}}")
ptxas = Path("/usr/local/cuda-12.5/bin/ptxas")
if not ptxas.is_file():
    raise RuntimeError("verified ptxas is unavailable")
work = Path("/kaggle/working")
source = dataset / "ARC2"
runner = source / "scripts" / "run_smoke12_ttt_cost_ablation.py"
native = source / "configs" / "nvarc_native_846d0198"
challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
if not runner.is_file() or not native.is_dir() or not challenge.is_file() or not model.is_dir():
    raise RuntimeError("required source/config/challenge/model attachment is missing")
out = work / "artifacts" / "smoke12_ttt_cost_accuracy_r1" / "{condition}"
if out.exists():
    raise FileExistsError("refusing to overwrite frozen Smoke12 candidates")
out.mkdir(parents=True)
for name in ("freeze", "runner_config", "experiment"):
    shutil.copy2(required[name], out / required[name].name)
print(json.dumps({{"event": "SMOKE12_TARGET_BLIND_START", "condition": "{condition}", "source_commit": "{source_commit}", "cohort_size": 12, "gpus": gpus, "solutions_opened": False}}, sort_keys=True), flush=True)
started = time.monotonic()
command = [
    sys.executable, str(runner), "--manifest", str(out / "freeze_manifest.json"),
    "--experiment-config", str(out / "runner_experiment_config.json"), "--condition", "{condition}",
    "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(native),
    "--output", str(out / "candidates_frozen.json"), "--checkpoint-dir", str(out / "checkpoints"), "--resume",
]
environment = {{**os.environ, "TRITON_PTXAS_PATH": str(ptxas), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}}
subprocess.run(command, env=environment, check=True, timeout=3600)
candidates = out / "candidates_frozen.json"
if not candidates.is_file():
    raise RuntimeError("Smoke12 candidate artifact was not written")
artifact = json.loads(candidates.read_text(encoding="utf-8"))
if artifact.get("status") != "SMOKE12_TTT_COST_ABLATION_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING" or artifact.get("condition") != "{condition}" or artifact.get("solutions_opened") is not False:
    raise RuntimeError("Smoke12 candidate freeze boundary failed")
receipt = {{"event": "SMOKE12_CANDIDATES_FROZEN", "condition": "{condition}", "task_count": len(artifact.get("records", {{}})), "wall_seconds": time.monotonic() - started, "candidate_sha256": sha256_file(candidates), "solutions_opened": False}}
(out / "candidate_freeze_receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True), flush=True)
'''
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def build(*, output: Path, condition: str, owner: str, source_commit: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite package: {output}")
    freeze, reference = _read(FREEZE), _read(REFERENCE_CONFIG)
    task_ids, task_ids_hash = _cohort(freeze)
    if freeze.get("status") != "SMOKE12_TTT_COST_ABLATION_COHORT_FROZEN":
        raise ValueError("Smoke12 cohort is not frozen")
    if reference.get("rank") != 256 or reference.get("alpha") != 32 or reference.get("ttt_steps") != 24 or reference.get("generation_augmentation_count") != 8:
        raise ValueError("reference TTT configuration differs from frozen Smoke12 contract")
    runner_config = _runner_config(condition, reference)
    runner_config["config_hash"] = _sha256_json(runner_config)
    spec = _CONDITIONS[condition]
    dataset_slug = f"arc2-smoke12-ttt-r1-{spec['label']}-source"
    kernel_slug = f"arc2-smoke12-ttt-r1-{spec['label']}"
    output.mkdir(parents=True)
    dataset, kernel = output / "dataset", output / "kernel"
    dataset.mkdir(); kernel.mkdir()
    archive = dataset / "ARC2.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", source_commit], cwd=ROOT, stdout=handle, check=True)
    runner_source = subprocess.check_output(["git", "show", f"{source_commit}:scripts/run_smoke12_ttt_cost_ablation.py"], cwd=ROOT)
    experiment = {
        "experiment_id": "ARC2_SMOKE12_TTT_COST_ACCURACY_ABLATION",
        "status": "FROZEN_PACKAGE_NOT_PUSHED_NOT_LAUNCHED",
        "condition": condition,
        "source_commit": source_commit,
        "cohort_task_ids": task_ids,
        "cohort_task_ids_hash": task_ids_hash,
        "scientific_contract": {
            "model": "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
            "rank": 256,
            "alpha": 32,
            "target_modules": reference["target_modules"],
            "augmentation_count": 8,
            "decode": spec["decode"],
            "beam_width": spec["beam_width"],
            "ttt_steps": spec["ttt_steps"],
            "no_dfs": True,
            "no_b_support": True,
        },
        "runtime_budget": {
            "max_wall_minutes": 60,
            "estimated_wall_minutes": spec["estimated_wall_minutes"],
            "estimated_gpu_seconds": spec["estimated_gpu_seconds"],
        },
        "input_sha256": {},
        "runner_source_sha256": hashlib.sha256(runner_source).hexdigest(),
        "solutions_opened": False,
    }
    _write_json(dataset / "freeze_manifest.json", freeze)
    _write_json(dataset / "runner_experiment_config.json", runner_config)
    for name in ("freeze_manifest.json", "runner_experiment_config.json"):
        experiment["input_sha256"][name.removesuffix(".json").replace("_manifest", "").replace("runner_experiment_config", "runner_config")] = _sha256_file(dataset / name)
    # The notebook uses short stable keys, while files retain descriptive names.
    experiment["input_sha256"] = {
        "freeze": _sha256_file(dataset / "freeze_manifest.json"),
        "runner_config": _sha256_file(dataset / "runner_experiment_config.json"),
    }
    _write_json(dataset / "experiment_config.json", experiment)
    _write_json(dataset / "dataset-metadata.json", {
        "id": f"{owner}/{dataset_slug}",
        "title": f"ARC2 Smoke12 TTT R1 {spec['label'].upper()} Source",
        "subtitle": f"Frozen target-blind {condition} source",
        "description": "Private experiment source; contains no ARC solution grids.",
        "licenses": [{"name": "other"}],
    })
    notebook_name = f"{kernel_slug}.ipynb"
    _write_json(kernel / notebook_name, _notebook(dataset_slug=dataset_slug, condition=condition, source_commit=source_commit))
    _write_json(kernel / "kernel-metadata.json", {
        "id": f"{owner}/{kernel_slug}",
        "title": f"ARC2 Smoke12 TTT R1 {spec['label'].upper()}",
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [f"{owner}/{dataset_slug}"],
        "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
        "machine_shape": "NvidiaL4",
    })
    manifest = {
        "status": "PACKAGE_BUILT_NOT_PUSHED_NOT_LAUNCHED",
        "condition": condition,
        "source_commit": source_commit,
        "cohort_task_ids_hash": task_ids_hash,
        "dataset_slug": f"{owner}/{dataset_slug}",
        "kernel_slug": f"{owner}/{kernel_slug}",
        "runner_config_hash": runner_config["config_hash"],
        "files": {
            str(path.relative_to(output)).replace("\\", "/"): _sha256_file(path)
            for path in sorted(path for path in output.rglob("*") if path.is_file())
        },
    }
    _write_json(output / "PACKAGE_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=sorted(_CONDITIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args()
    source_commit = args.source_commit or _git("rev-parse", "HEAD")
    manifest = build(output=args.output, condition=args.condition, owner=args.owner, source_commit=source_commit)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
