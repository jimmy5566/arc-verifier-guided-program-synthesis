#!/usr/bin/env python3
"""Build a private, review-only Kaggle package for Retention30.

No Kaggle API is used here.  The package is created from an explicit Git
commit and contains no solution or answer artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "eval60_dual_ttt_dfs1024_retention30_v1"
EXPERIMENT_ID = "EVAL60_DUAL_TTT_DFS1024_RETENTION30_V1"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"
MODEL_SOURCE = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
REFERENCE_KERNEL = "sorokin/pip-install-unsloth-flash-patch"
COMPETITION = "arc-prize-2026-arc-agi-2"
SOURCE_PATHS = (
    "src",
    "configs/nvarc_native_846d0198",
    "scripts/run_eval60_dual_ttt_dfs1024_retention30.py",
    "scripts/score_eval60_retention30.py",
    "scripts/run_l4_dual_ttt_dfs1024_benchmark.py",
    "scripts/run_adaptive_ttt_loo_transfer12.py",
    "scripts/run_adaptive_ttt_step1.py",
    "scripts/run_eval3_reference_ttt.py",
    "scripts/turbodfs_d1_common.py",
    "scripts/turbodfs_v4_common.py",
    "scripts/run_eval60_adaptive_inference_joint_v2.py",
    "experiments/eval60_dual_ttt_dfs1024_retention30_v1/TARGET_BLIND_RUN_MANIFEST.json",
    "experiments/eval60_dual_ttt_dfs1024_retention30_v1/RETENTION30_OUTPUTS.json",
    "experiments/eval60_dual_ttt_dfs1024_retention30_v1/RETENTION30_COHORT_SHA256.txt",
    "experiments/eval60_dual_ttt_dfs1024_retention30_v1/HISTORICAL_GREEDY_SOURCE_CLASSES.csv",
    "experiments/eval60_dual_ttt_dfs1024_retention30_v1/PROVENANCE.json",
    "experiments/eval60_dual_ttt_dfs1024_retention30_v1/retention_config.json",
    "experiments/eval60_dual_ttt_dfs1024_retention30_v1/README.md",
    "pyproject.toml",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def archive_source(destination: Path, commit: str) -> None:
    destination.mkdir(parents=True)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as handle:
        archive = Path(handle.name)
    try:
        with archive.open("wb") as handle:
            subprocess.run(["git", "archive", "--format=tar", commit, "--", *SOURCE_PATHS], cwd=ROOT, stdout=handle, check=True)
        with tarfile.open(archive) as handle:
            handle.extractall(destination, filter="data")
    finally:
        archive.unlink(missing_ok=True)


def files_manifest(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def notebook_source(owner: str, dataset_slug: str) -> str:
    dataset_id = f"{owner}/{dataset_slug}"
    required_outputs = [
        "PROVENANCE.json", "RETENTION30_OUTPUTS.json", "RETENTION30_COHORT_SHA256.txt",
        "HISTORICAL_GREEDY_SOURCE_CLASSES.csv", "TARGET_BLIND_RUN_MANIFEST.json",
        "dfs_candidates_frozen.json", "ttt24_union_candidates_frozen.json",
        "ttt48_union_candidates_frozen.json", "dual_union_candidates_frozen.json",
        "dfs_cells.csv", "phase_intervals.csv", "gpu_telemetry_1s.csv",
        "gpu_phase_summary.csv", "TELEMETRY_SUMMARY.json", "worker_summary.csv", "dfs_source_hit_overlap.csv",
        "DFS_HIT_OVERLAP_SUMMARY.json", "dfs_view_gold_contribution.csv",
        "historical_ttt48_marginal_under_dfs.csv", "historical_ttt24_marginal_under_dfs.csv",
        "historical_top2_retention.csv", "historical_oracle_only_retention.csv",
        "RETENTION_REPORT.md", "DECISION.json",
    ]
    return "\n".join([
        "import json, os, shutil, subprocess, sys, time",
        "from pathlib import Path",
        f'EXPERIMENT_ID = "{EXPERIMENT_ID}"',
        f'DATASET = Path("/kaggle/input/datasets/{dataset_id}")',
        'SOURCE = DATASET / "ARC2"',
        'CHALLENGE = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")',
        'MODEL = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")',
        'RUNNER = SOURCE / "scripts/run_eval60_dual_ttt_dfs1024_retention30.py"',
        'SCORER = SOURCE / "scripts/score_eval60_retention30.py"',
        'NATIVE_CONFIG = SOURCE / "configs/nvarc_native_846d0198"',
        'CONFIG = DATASET / "retention_config_resolved.json"',
        'MANIFEST = DATASET / "TARGET_BLIND_RUN_MANIFEST.json"',
        'RETENTION = DATASET / "RETENTION30_OUTPUTS.json"',
        'OUTPUT = Path("/kaggle/working/analysis/eval60_dual_ttt_dfs1024_retention30_v1")',
        'required = [SOURCE, CHALLENGE, MODEL, RUNNER, SCORER, NATIVE_CONFIG, CONFIG, MANIFEST, RETENTION]',
        'missing = [str(path) for path in required if not path.exists()]',
        'if missing: raise RuntimeError(f"retention inputs missing: {missing}")',
        'if OUTPUT.exists() and any(OUTPUT.iterdir()): raise RuntimeError("RERUN_ENABLED=FALSE: output already exists")',
        'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {"1", "true", "yes"}: raise RuntimeError("Internet must be disabled")',
        'gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()',
        'if len(gpus) != 4 or any("NVIDIA L4" not in row for row in gpus): raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {gpus}")',
        'runner_command = [sys.executable, str(RUNNER), "--challenge", str(CHALLENGE), "--model-path", str(MODEL), "--native-config-dir", str(NATIVE_CONFIG), "--config", str(CONFIG), "--manifest", str(MANIFEST), "--output", str(OUTPUT)]',
        'print(json.dumps({"event": "TARGET_BLIND_GPU_PHASE_START", "solutions_passed_to_runner": False, "gpus": gpus}, sort_keys=True), flush=True)',
        'subprocess.run(runner_command, check=True, env={**os.environ, "TRITON_PTXAS_PATH": "/usr/local/cuda-12.5/bin/ptxas", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})',
        '# The scorer is a separate process and is invoked only after the runner has atomically frozen and hashed all candidates.',
        'freeze = json.loads((OUTPUT / "TARGET_BLIND_CANDIDATE_FREEZE.json").read_text(encoding="utf-8"))',
        'if freeze.get("status") != "TARGET_BLIND_CANDIDATES_FROZEN" or freeze.get("solutions_accessed") is not False: raise RuntimeError("target-blind freeze gate failed")',
        'SOLUTIONS = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")',
        'if not SOLUTIONS.is_file(): raise RuntimeError("post-freeze Eval60 solutions file is missing")',
        'scorer_command = [sys.executable, str(SCORER), "--output", str(OUTPUT), "--solutions", str(SOLUTIONS), "--retention", str(RETENTION)]',
        'subprocess.run(scorer_command, check=True)',
        'for name in ("RETENTION30_OUTPUTS.json", "RETENTION30_COHORT_SHA256.txt", "HISTORICAL_GREEDY_SOURCE_CLASSES.csv"):',
        '    shutil.copy2(DATASET / name, OUTPUT / name)',
        f'required_outputs = {required_outputs!r}',
        'missing_outputs = [name for name in required_outputs if not (OUTPUT / name).is_file()]',
        'if missing_outputs: raise RuntimeError(f"retention outputs missing: {missing_outputs}")',
        'decision = json.loads((OUTPUT / "DECISION.json").read_text(encoding="utf-8"))',
        'print(json.dumps({"event": "RETENTION30_NOTEBOOK_COMPLETE", "decision": decision, "output": str(OUTPUT)}, sort_keys=True), flush=True)',
    ])


def notebook_document(source: str) -> dict[str, Any]:
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in source.splitlines()]}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or (args.review_output is not None and args.review_output.exists()):
        raise FileExistsError("refusing to overwrite package or review snapshot")
    commit = args.source_commit or git_output("rev-parse", "HEAD")
    git_output("cat-file", "-e", f"{commit}^{{commit}}")
    branch = args.source_branch or git_output("branch", "--show-current")
    dataset = args.output / "dataset"
    source = dataset / "ARC2"
    archive_source(source, commit)
    config = read_json(EXPERIMENT / "retention_config.json")
    config["authoritative_source_commit"] = commit
    write_json(dataset / "retention_config_resolved.json", config)
    for name in ("TARGET_BLIND_RUN_MANIFEST.json", "RETENTION30_OUTPUTS.json", "RETENTION30_COHORT_SHA256.txt", "HISTORICAL_GREEDY_SOURCE_CLASSES.csv", "PROVENANCE.json"):
        shutil.copy2(EXPERIMENT / name, dataset / name)
    write_json(dataset / "SOURCE_IDENTITY.json", {
        "experiment_id": EXPERIMENT_ID,
        "source_commit": commit,
        "source_branch": branch,
        "source_git_status": "PACKAGED_CLEAN_FROM_GIT_ARCHIVE",
        "source_paths": list(SOURCE_PATHS),
        "telemetry_infra_only": True,
        "scientific_config_changed": False,
        "cohort_changed": False,
        "telemetry_unmatched_policy": "UNCLASSIFIED",
    })
    write_json(dataset / "dataset-metadata.json", {"id": f"{args.owner}/{args.dataset_slug}", "title": "ARC2 Eval60 Dual-TTT DFS1024 Retention30", "subtitle": "Private target-blind Retention30 experiment source", "description": "Contains a frozen output-ID manifest but no solution grids or historical candidate answers.", "licenses": [{"name": "other"}]})
    forbidden = [
        path
        for path in dataset.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl", ".csv", ".zip", ".parquet"}
        and any(token in path.name.lower() for token in ("solution", "answer", "gold", "prediction"))
    ]
    if forbidden:
        raise RuntimeError(f"private source package contains forbidden target artifact: {forbidden}")
    kernel = args.output / "kernel"
    kernel.mkdir(parents=True)
    notebook_name = f"{args.kernel_slug}.ipynb"
    write_json(kernel / notebook_name, notebook_document(notebook_source(args.owner, args.dataset_slug)))
    metadata = {
        "id": f"{args.owner}/{args.kernel_slug}", "title": "ARC2 Eval60 Dual-TTT DFS1024 Retention30",
        "code_file": notebook_name, "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu", "target-blind", "retention"],
        "dataset_sources": [f"{args.owner}/{args.dataset_slug}"], "kernel_sources": [REFERENCE_KERNEL],
        "competition_sources": [COMPETITION], "model_sources": [MODEL_SOURCE], "docker_image": IMAGE, "machine_shape": "NvidiaL4",
    }
    write_json(kernel / "kernel-metadata.json", metadata)
    source_manifest = {"experiment_id": EXPERIMENT_ID, "source_commit": commit, "contains_solutions": False, "files": files_manifest(dataset)}
    write_json(args.output / "SOURCE_PACKAGE_MANIFEST.json", source_manifest)
    package = {
        "status": "READY_FOR_GITHUB_REVIEW_NOT_PUBLISHED_NOT_RUN", "experiment_id": EXPERIMENT_ID,
        "source_commit": commit, "source_branch": branch, "dataset_slug": f"{args.owner}/{args.dataset_slug}",
        "kernel_slug": f"{args.owner}/{args.kernel_slug}", "notebook_sha256": sha256_file(kernel / notebook_name),
        "config_sha256": sha256_file(dataset / "retention_config_resolved.json"),
        "cohort_sha256": (dataset / "RETENTION30_COHORT_SHA256.txt").read_text(encoding="utf-8").strip(),
        "telemetry_infra_only": True, "scientific_config_changed": False, "cohort_changed": False,
        "kaggle_dataset_pushed": False, "kaggle_notebook_pushed": False, "kaggle_run_started": False,
        "competition_submissions": 0,
    }
    write_json(args.output / "PACKAGE_MANIFEST.json", package)
    if args.review_output is not None:
        args.review_output.mkdir(parents=True)
        for source_file in (kernel / notebook_name, kernel / "kernel-metadata.json", dataset / "retention_config_resolved.json", dataset / "TARGET_BLIND_RUN_MANIFEST.json", dataset / "RETENTION30_OUTPUTS.json", dataset / "RETENTION30_COHORT_SHA256.txt", dataset / "HISTORICAL_GREEDY_SOURCE_CLASSES.csv", dataset / "SOURCE_IDENTITY.json", args.output / "SOURCE_PACKAGE_MANIFEST.json", args.output / "PACKAGE_MANIFEST.json"):
            shutil.copy2(source_file, args.review_output / source_file.name)
        write_json(args.review_output / "REVIEW_STATUS.json", {
            "status": "READY_FOR_RETENTION30_KAGGLE_RUN",
            "telemetry_infra_only": True,
            "scientific_config_changed": False,
            "cohort_changed": False,
            "gpu_runs_started": 0,
            "kaggle_datasets_pushed": 0,
            "kaggle_notebooks_pushed": 0,
            "competition_submissions": 0,
        })
    return package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--dataset-slug", default="arc2-eval60-dual-ttt-dfs1024-retention30-v1-source")
    parser.add_argument("--kernel-slug", default="arc2-eval60-dual-ttt-dfs1024-retention30-v1")
    parser.add_argument("--source-commit")
    parser.add_argument("--source-branch")
    args = parser.parse_args()
    print(json.dumps(build(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
