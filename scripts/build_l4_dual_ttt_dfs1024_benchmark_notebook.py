#!/usr/bin/env python3
"""Build, but never publish or run, the 4xL4 dual-TTT DFS1024 benchmark."""
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
EXPERIMENT = ROOT / "experiments" / "l4_dual_ttt_dfs1024_bench_v1"
BENCHMARK_ID = "L4_DUAL_TTT_DFS1024_NOTEBOOK_BENCH_V1"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"
MODEL_SOURCE = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
REFERENCE_KERNEL = "sorokin/pip-install-unsloth-flash-patch"
COMPETITION = "arc-prize-2026-arc-agi-2"
SOURCE_PATHS = (
    "src",
    "configs/nvarc_native_846d0198",
    "scripts/run_l4_dual_ttt_dfs1024_benchmark.py",
    "scripts/run_adaptive_ttt_loo_transfer12.py",
    "scripts/run_adaptive_ttt_step1.py",
    "scripts/run_eval3_reference_ttt.py",
    "scripts/turbodfs_d1_common.py",
    "scripts/turbodfs_v4_common.py",
    "scripts/run_eval60_adaptive_inference_joint_v2.py",
    "experiments/l4_dual_ttt_dfs1024_bench_v1/BENCHMARK_TASK_IDS.json",
    "experiments/l4_dual_ttt_dfs1024_bench_v1/BENCHMARK_COHORT_SHA256.txt",
    "experiments/l4_dual_ttt_dfs1024_bench_v1/benchmark_config.json",
    "experiments/l4_dual_ttt_dfs1024_bench_v1/FAILED_V1_INFRA_PROVENANCE.json",
    "experiments/l4_dual_ttt_dfs1024_bench_v1/README.md",
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
            subprocess.run(
                ["git", "archive", "--format=tar", commit, "--", *SOURCE_PATHS],
                cwd=ROOT,
                stdout=handle,
                check=True,
            )
        with tarfile.open(archive) as handle:
            handle.extractall(destination, filter="data")
    finally:
        archive.unlink(missing_ok=True)


def notebook_source(owner: str, dataset_slug: str) -> str:
    dataset_id = f"{owner}/{dataset_slug}"
    return "\n".join(
        [
            "import json, os, subprocess, sys, time",
            "from pathlib import Path",
            "",
            "NOTEBOOK_START_MONOTONIC = time.perf_counter()",
            'print("L4_DUAL_TTT_DFS1024_NOTEBOOK_BENCH_V1", flush=True)',
            'print("RERUN_ENABLED = FALSE", flush=True)',
            f'DATASET = Path("/kaggle/input/datasets/{dataset_id}")',
            'SOURCE = DATASET / "ARC2"',
            'CHALLENGE = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")',
            'MODEL = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")',
            'RUNNER = SOURCE / "scripts/run_l4_dual_ttt_dfs1024_benchmark.py"',
            'NATIVE_CONFIG = SOURCE / "configs/nvarc_native_846d0198"',
            'CONFIG = DATASET / "benchmark_config_resolved.json"',
            'COHORT = DATASET / "BENCHMARK_TASK_IDS.json"',
            'IDENTITY = DATASET / "SOURCE_IDENTITY.json"',
            'OUTPUT = Path("/kaggle/working/analysis/l4_dual_ttt_dfs1024_bench_v1")',
            'required = [SOURCE, CHALLENGE, MODEL, RUNNER, NATIVE_CONFIG, CONFIG, COHORT, IDENTITY]',
            'missing = [str(path) for path in required if not path.exists()]',
            'if missing: raise RuntimeError(f"benchmark inputs missing: {missing}")',
            'if OUTPUT.exists() and any(OUTPUT.iterdir()): raise RuntimeError("RERUN_ENABLED=FALSE: benchmark output already exists")',
            'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {"1", "true", "yes"}: raise RuntimeError("Internet must be disabled")',
            'gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()',
            'if len(gpus) != 4 or any("NVIDIA L4" not in row for row in gpus): raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {gpus}")',
            'identity = json.loads(IDENTITY.read_text(encoding="utf-8"))',
            'env_init_s = time.perf_counter() - NOTEBOOK_START_MONOTONIC',
            'command = [sys.executable, str(RUNNER), "--challenge", str(CHALLENGE), "--model-path", str(MODEL), "--native-config-dir", str(NATIVE_CONFIG), "--config", str(CONFIG), "--cohort", str(COHORT), "--output", str(OUTPUT), "--source-branch", identity["source_branch"], "--source-git-status", identity["source_git_status"], "--notebook-start-monotonic", str(NOTEBOOK_START_MONOTONIC), "--env-init-seconds", str(env_init_s)]',
            'print(json.dumps({"event": "BENCHMARK_NOTEBOOK_PREFLIGHT_PASS", "dataset": identity, "gpus": gpus, "internet": False, "rerun": False}, sort_keys=True), flush=True)',
            'subprocess.run(command, check=True, env={**os.environ, "TRITON_PTXAS_PATH": "/usr/local/cuda-12.5/bin/ptxas", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})',
            'required_outputs = ["benchmark_provenance.json", "BENCHMARK_TASK_IDS.json", "BENCHMARK_COHORT_SHA256.txt", "task_timing.csv", "cell_dfs_metrics.csv", "gpu_telemetry_1s.csv", "gpu_summary.csv", "system_telemetry_1s.csv", "worker_summary.csv", "throughput_summary.json", "runtime_projection.json", "failures.jsonl", "L4_DUAL_TTT_DFS1024_BENCH_REPORT.md", "DECISION.json"]',
            'missing_outputs = [name for name in required_outputs if not (OUTPUT / name).is_file()]',
            'if missing_outputs: raise RuntimeError(f"benchmark outputs missing: {missing_outputs}")',
            'decision = json.loads((OUTPUT / "DECISION.json").read_text(encoding="utf-8"))',
            'if decision.get("status") != "PASS" or decision.get("gold_accessed") is not False or decision.get("rerun_enabled") is not False: raise RuntimeError(f"benchmark decision failed: {decision}")',
            'print(json.dumps({"event": "BENCHMARK_NOTEBOOK_COMPLETE", "output": str(OUTPUT), "status": decision["status"], "rerun": False}, sort_keys=True), flush=True)',
        ]
    )


def notebook_document(source: str) -> dict[str, Any]:
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in source.splitlines()],
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {
                "accelerator": "nvidiaL4",
                "isGpuEnabled": True,
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def kernel_metadata(owner: str, kernel_slug: str, dataset_slug: str) -> dict[str, Any]:
    notebook_name = f"{kernel_slug}.ipynb"
    return {
        "id": f"{owner}/{kernel_slug}",
        "title": "ARC2 L4 Dual TTT DFS1024 Performance Benchmark",
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu", "performance-benchmark"],
        "dataset_sources": [f"{owner}/{dataset_slug}"],
        "kernel_sources": [REFERENCE_KERNEL],
        "competition_sources": [COMPETITION],
        "model_sources": [MODEL_SOURCE],
        "docker_image": IMAGE,
        "machine_shape": "NvidiaL4",
    }


def files_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark staging: {args.output}")
    if args.review_output is not None and args.review_output.exists():
        raise FileExistsError(f"refusing to overwrite review snapshot: {args.review_output}")

    commit = args.source_commit or git_output("rev-parse", "HEAD")
    git_output("cat-file", "-e", f"{commit}^{{commit}}")
    source_branch = args.source_branch or git_output("branch", "--show-current")
    dataset = args.output / "dataset"
    source = dataset / "ARC2"
    archive_source(source, commit)

    config = read_json(EXPERIMENT / "benchmark_config.json")
    config["scientific_base_commit"] = config["authoritative_source_commit"]
    config["authoritative_source_commit"] = commit
    cohort = read_json(EXPERIMENT / "BENCHMARK_TASK_IDS.json")
    write_json(dataset / "benchmark_config_resolved.json", config)
    write_json(dataset / "BENCHMARK_TASK_IDS.json", cohort)
    (dataset / "BENCHMARK_COHORT_SHA256.txt").write_text(cohort["task_ids_canonical_sha256"] + "\n", encoding="utf-8")
    source_identity = {
        "benchmark_id": BENCHMARK_ID,
        "infra_fix_only": True,
        "previous_gpu_run_scientific_data": "NONE",
        "source_commit": commit,
        "scientific_base_commit": config["scientific_base_commit"],
        "source_branch": source_branch,
        "source_git_status": "PACKAGED_CLEAN_FROM_GIT_ARCHIVE",
        "source_paths": list(SOURCE_PATHS),
    }
    write_json(dataset / "SOURCE_IDENTITY.json", source_identity)
    write_json(
        dataset / "dataset-metadata.json",
        {
            "id": f"{args.owner}/{args.dataset_slug}",
            "title": "ARC2 L4 dual-TTT DFS1024 benchmark source",
            "subtitle": "Target-blind performance benchmark source with frozen exposed-dev cohort",
            "description": "Private performance-only package. Contains no ARC solution grids or predictions.",
            "licenses": [{"name": "other"}],
        },
    )

    forbidden = [
        path
        for path in dataset.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl", ".csv"}
        and any(token in path.name.lower() for token in ("solution", "answer", "gold", "prediction"))
    ]
    if forbidden:
        raise RuntimeError(f"target-blind source package contains forbidden answer artifacts: {forbidden}")

    kernel = args.output / "kernel"
    kernel.mkdir(parents=True)
    notebook_name = f"{args.kernel_slug}.ipynb"
    notebook = notebook_document(notebook_source(args.owner, args.dataset_slug))
    metadata = kernel_metadata(args.owner, args.kernel_slug, args.dataset_slug)
    write_json(kernel / notebook_name, notebook)
    write_json(kernel / "kernel-metadata.json", metadata)

    source_files = files_manifest(dataset)
    source_package_manifest = {
        "benchmark_id": BENCHMARK_ID,
        "source_commit": commit,
        "scientific_base_commit": config["scientific_base_commit"],
        "file_count": len(source_files),
        "files": source_files,
        "contains_solution_artifacts": False,
    }
    write_json(args.output / "SOURCE_PACKAGE_MANIFEST.json", source_package_manifest)
    package_manifest = {
        "status": "PACKAGE_BUILT_NOT_PUSHED_NOT_LAUNCHED",
        "benchmark_id": BENCHMARK_ID,
        "source_commit": commit,
        "source_branch": source_branch,
        "dataset_slug": f"{args.owner}/{args.dataset_slug}",
        "kernel_slug": f"{args.owner}/{args.kernel_slug}",
        "notebook_sha256": sha256_file(kernel / notebook_name),
        "kernel_metadata_sha256": sha256_file(kernel / "kernel-metadata.json"),
        "resolved_config_sha256": sha256_file(dataset / "benchmark_config_resolved.json"),
        "cohort_sha256": cohort["task_ids_canonical_sha256"],
        "files": files_manifest(args.output),
    }
    write_json(args.output / "PACKAGE_MANIFEST.json", package_manifest)

    if args.review_output is not None:
        review = args.review_output
        review.mkdir(parents=True)
        shutil.copy2(kernel / notebook_name, review / notebook_name)
        shutil.copy2(kernel / "kernel-metadata.json", review / "kernel-metadata.json")
        shutil.copy2(dataset / "benchmark_config_resolved.json", review / "benchmark_config_resolved.json")
        shutil.copy2(dataset / "BENCHMARK_TASK_IDS.json", review / "BENCHMARK_TASK_IDS.json")
        shutil.copy2(dataset / "SOURCE_IDENTITY.json", review / "SOURCE_IDENTITY.json")
        shutil.copy2(source / "experiments/l4_dual_ttt_dfs1024_bench_v1/FAILED_V1_INFRA_PROVENANCE.json", review / "FAILED_V1_INFRA_PROVENANCE.json")
        shutil.copy2(args.output / "SOURCE_PACKAGE_MANIFEST.json", review / "SOURCE_PACKAGE_MANIFEST.json")
        shutil.copy2(args.output / "PACKAGE_MANIFEST.json", review / "PACKAGE_MANIFEST.json")
        write_json(
            review / "REVIEW_STATUS.json",
            {
                "status": "READY_FOR_GITHUB_REVIEW_NOT_PUBLISHED_NOT_RUN",
                "gpu_runs_started": 0,
                "kaggle_datasets_pushed": 0,
                "kaggle_notebooks_pushed": 0,
                "competition_submissions": 0,
            },
        )
    return package_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--dataset-slug", default="arc2-l4-dual-ttt-dfs1024-bench-v1-source")
    parser.add_argument("--kernel-slug", default="arc2-l4-dual-ttt-dfs1024-bench-v1")
    parser.add_argument("--source-commit")
    parser.add_argument("--source-branch")
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
