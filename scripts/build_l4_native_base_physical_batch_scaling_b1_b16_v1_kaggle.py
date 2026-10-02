#!/usr/bin/env python3
"""Build, but never upload or launch, the private L4 batch-scaling review package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts.run_l4_native_base_physical_batch_scaling_b1_b16_v1 import (  # noqa: E402
    AUTHORITATIVE_SOURCE_COMMIT,
    EXPERIMENT,
    experiment_contract,
)


# This is the sole permitted source-core delta from the frozen native base
# benchmark: it adapts the public Transformers v5 cache-layer representation
# to the pre-existing cache pack/split protocol.  It is a byte-pinned runtime
# compatibility shim, not a decoder, view, or timing change.
RUNTIME_DYNAMICCACHE_COMPAT_SHA256 = "4e86b52ba0c65e080c4a24d0df99900dacacf0b0ba2c23af4c8c1649f95509e1"


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _assert_authoritative_core_unchanged() -> None:
    core = (
        "scripts/run_clean_hf_parallel_dfs_b4_compute_v1.py",
        "scripts/run_clean_hf_parallel_dfs_batch_scaling_v2.py",
        "scripts/run_clean_hf_b16_transient_memory_opt_v1.py",
    )
    changed = subprocess.run(["git", "diff", "--quiet", AUTHORITATIVE_SOURCE_COMMIT, "--", *core], cwd=ROOT)
    if changed.returncode != 0:
        raise RuntimeError("frozen authoritative core differs from requested source commit")
    actual = _sha256(ROOT / "src" / "inference" / "nvarc_turbodfs_dynamic_ready.py")
    if actual != RUNTIME_DYNAMICCACHE_COMPAT_SHA256:
        raise RuntimeError("DynamicCache compatibility source differs from the pinned runtime-only patch")


def _archive_curated_source(destination: Path, *, source_ref: str) -> None:
    """Archive only runtime code/config, excluding historical artifacts and target files."""
    paths = (
        "src/arc",
        "src/inference/__init__.py",
        "src/inference/arc_native_io.py",
        "src/inference/nvarc_native.py",
        "src/inference/nvarc_native_augmentation.py",
        "src/inference/nvarc_turbodfs_d1.py",
        "src/inference/nvarc_turbodfs_reference.py",
        "src/inference/nvarc_turbodfs_dynamic_ready.py",
        "configs/nvarc_native_846d0198",
        "scripts/run_clean_hf_parallel_dfs_b4_compute_v1.py",
        "scripts/run_clean_hf_parallel_dfs_batch_scaling_v2.py",
        "scripts/run_clean_hf_b16_transient_memory_opt_v1.py",
    )
    archive = destination.parent / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", source_ref, *paths], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as handle:
        handle.extractall(destination, filter="data")
    archive.unlink()
    shutil.copy2(ROOT / "scripts" / "run_l4_native_base_physical_batch_scaling_b1_b16_v1.py",
                 destination / "scripts" / "run_l4_native_base_physical_batch_scaling_b1_b16_v1.py")
    # Runtime source legitimately contains generic normalisation code whose
    # module name includes ``solution``.  Only data-like target artifacts are
    # excluded from this target-blind package.
    forbidden = [path for path in destination.rglob("*")
                 if path.is_file() and path.suffix.lower() in {".json", ".csv", ".parquet"}
                 and "solution" in path.name.lower()]
    if forbidden:
        raise RuntimeError(f"curated target-blind source contains forbidden file names: {forbidden}")


def _assert_staged_runner_importable(source: Path) -> None:
    """Exercise the package import closure, not the developer worktree's one."""
    source = source.resolve()
    runner = source / "scripts" / "run_l4_native_base_physical_batch_scaling_b1_b16_v1.py"
    if not runner.is_file():
        raise RuntimeError("curated source is missing the benchmark runner")
    environment = {**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"}
    probe = subprocess.run(
        [sys.executable, "-c", "import runpy, sys; sys.path.insert(0, sys.argv[1]); runpy.run_path(sys.argv[2])", str(source), str(runner)],
        cwd=source, env=environment, text=True, capture_output=True,
    )
    if probe.returncode:
        raise RuntimeError(f"curated runner import closure failed:\n{probe.stderr}")
    bytecode = list(source.rglob("__pycache__"))
    if bytecode:
        raise RuntimeError(f"curated source unexpectedly contains bytecode directories: {bytecode}")


def _notebook_source(dataset_slug: str, harness_commit: str, *, engineering_smoke: bool = False) -> str:
    mode = "ENGINEERING_SMOKE_ONLY" if engineering_smoke else "FORMAL_SCIENTIFIC_BENCHMARK"
    execution_widths = [1, 16] if engineering_smoke else [1, 2, 4, 8, 12, 16]
    required_output = (
        [
            "CONTRACT.json", "SOURCE_IDENTITY.json", "RUNTIME_ENVIRONMENT.json", "DYNAMICCACHE_PREFLIGHT.json",
            "ROOT_TEMPLATE_PARITY.json", "ROOT_TEMPLATE_IMMUTABILITY.json", "ROOT_TEMPLATE_MEMORY_AUDIT.json",
            "SETUP_COST_AUDIT.json", "CHECKPOINT.json", "PROGRESS.jsonl", "B1_WORKERS.json", "B16_WORKERS.json",
            "B1_PARTIAL.json", "B16_PARTIAL.json", "ENGINEERING_SMOKE.json",
        ] if engineering_smoke else [
            "CONTRACT.json", "SOURCE_IDENTITY.json", "RUNTIME_ENVIRONMENT.json", "L4_HARDWARE.json",
            "DYNAMICCACHE_PREFLIGHT.json", "ROOT_TEMPLATE_PARITY.json", "ROOT_TEMPLATE_IMMUTABILITY.json",
            "ROOT_TEMPLATE_MEMORY_AUDIT.json", "SETUP_COST_AUDIT.json", "CHECKPOINT.json", "PROGRESS.jsonl",
            "L4_BATCH_SCALING_RAW.csv", "L4_BATCH_SCALING_PER_GPU.csv", "L4_BATCH_SCALING_AGGREGATE.csv",
            "L4_BATCH_SCALING_BOOTSTRAP.csv", "B12_MEMORY_WATERFALL.csv", "B16_MEMORY_WATERFALL.csv",
            "THERMAL_POWER_SUMMARY.csv", "RTX3090_REFERENCE.json", "RTX3090_VS_L4_SCALING_SHAPE.csv",
            "DECISION.json", "REPORT.md", "HASHES.json",
        ]
    )
    return "\n".join([
        "import json, os, subprocess, sys, time",
        "from pathlib import Path",
        "dataset = Path('/kaggle/input/datasets/jimmy5566/') / " + repr(dataset_slug),
        # The Kaggle dataset mount exposes the CLI-uploaded ARC2 archive as
        # this immutable source directory.  Use the mounted source directly.
        "source = dataset / 'ARC2'",
        "out = Path('/kaggle/working/analysis/l4_native_base_batch_scaling_b1_b16_v1')",
        "if os.environ.get('KAGGLE_KERNEL_INTERNET_ENABLED', '').strip().lower() in {'1','true','yes'}: raise RuntimeError('Internet must be disabled')",
        "gpus = subprocess.check_output(['nvidia-smi', '-L'], text=True).splitlines()",
        "if len(gpus) != 4 or any('NVIDIA L4' not in row for row in gpus): raise RuntimeError(f'requires exactly four NVIDIA L4 GPUs: {gpus}')",
        "model = Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1')",
        "challenge = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json')",
        "native = source / 'configs' / 'nvarc_native_846d0198'",
        "runner = source / 'scripts' / 'run_l4_native_base_physical_batch_scaling_b1_b16_v1.py'",
        "required = {'source': source, 'runner': runner, 'model': model, 'challenge': challenge, 'native': native, 'contract': dataset / 'L4_BENCHMARK_CONTRACT.json'}",
        "missing = [name for name, path in required.items() if not path.exists()]",
        "if missing: raise RuntimeError(f'attached benchmark package incomplete: {missing}')",
        "contract = json.loads((dataset / 'L4_BENCHMARK_CONTRACT.json').read_text())",
        "if contract['experiment'] != 'L4_NATIVE_BASE_PHYSICAL_BATCH_SCALING_B1_B16_V1' or contract['authoritative_source_commit'] != '1eb8e7f60a3ca682438bb326ab3ea65ec286ed6f': raise RuntimeError('benchmark contract mismatch')",
        "env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false', 'PYTHONUNBUFFERED':'1'}",
        "common = [sys.executable, str(runner), '--output', str(out), '--source-commit', contract['authoritative_source_commit'], '--harness-commit', " + repr(harness_commit) + ", '--model-path', str(model), '--challenge', str(challenge), '--native-config-dir', str(native), '--benchmark-model-mode', 'BASE_MODEL_ONLY']" + (" + ['--engineering-smoke']" if engineering_smoke else ""),
        "print(json.dumps({'event':'NOTEBOOK_START','mode':" + repr(mode) + ",'source':str(source),'runner':str(runner),'output':str(out),'hardware_count':len(gpus)}, sort_keys=True), flush=True)",
        "print(json.dumps({'event':'L4_BATCH_SCALING_START','experiment':contract['experiment'],'target_blind':True,'gold_loaded':False,'hardware':gpus,'widths':" + repr(execution_widths) + "}, sort_keys=True), flush=True)",
        "for phase in ('preflight','runtime_preflight','controller'):",
        "    command = [sys.executable, '-u'] + common[1:2] + ['--phase', phase] + common[2:]",
        "    print(json.dumps({'event':'NOTEBOOK_PHASE_START','phase':phase,'command':command}, sort_keys=True), flush=True)",
        "    completed = subprocess.run(command, env=env)",
        "    if completed.returncode:",
        "        listing = sorted(path.name for path in out.iterdir()) if out.exists() else []",
        "        print(json.dumps({'event':'NOTEBOOK_PHASE_ERROR','phase':phase,'returncode':completed.returncode,'output_listing':listing}, sort_keys=True), flush=True)",
        "        raise RuntimeError(f'benchmark child failed: phase={phase} returncode={completed.returncode}')",
        "    print(json.dumps({'event':'NOTEBOOK_PHASE_DONE','phase':phase}, sort_keys=True), flush=True)",
        "required_output = " + repr(required_output),
        "missing_output = [name for name in required_output if not (out / name).is_file()]",
        "if missing_output: raise RuntimeError(f'benchmark output incomplete: {missing_output}')",
        "print(json.dumps({'event':'L4_BATCH_SCALING_COMPLETE','artifacts':str(out),'submission_created':False}, sort_keys=True), flush=True)",
    ])


def _assert_notebook_contract(notebook: dict[str, Any]) -> None:
    source = "".join(notebook["cells"][0]["source"])
    forbidden = ("submission.json", "KAGGLE_IS_COMPETITION_RERUN", "FAST_COMMIT", "evaluation_solutions", "arc-agi_evaluation_solutions")
    if any(token in source for token in forbidden):
        raise RuntimeError("review notebook violates target-blind/no-submission contract")
    required = ("nvidia-smi", "L4", "controller", "runtime_preflight", "BASE_MODEL_ONLY")
    if any(token not in source for token in required):
        raise RuntimeError("review notebook is missing pinned environment or L4 controller preflight")


def build(*, output: Path, owner: str, dataset_slug: str, kernel_slug: str,
          engineering_smoke: bool = False) -> dict[str, Any]:
    output = output.resolve()
    if any(os.environ.get(name) for name in ("GH_TOKEN", "GITHUB_TOKEN")):
        raise RuntimeError("GitHub token environment variables must be unset before benchmark packaging")
    _assert_authoritative_core_unchanged()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {output}")
    harness_commit = _git("rev-parse", "HEAD")
    if subprocess.run(["git", "merge-base", "--is-ancestor", AUTHORITATIVE_SOURCE_COMMIT, harness_commit], cwd=ROOT).returncode:
        raise RuntimeError("harness is not based on the authoritative source commit")
    dataset, kernel = output / "dataset", output / "kernel"
    source = dataset / "ARC2"
    source.mkdir(parents=True)
    _archive_curated_source(source, source_ref=harness_commit)
    _assert_staged_runner_importable(source)
    contract = experiment_contract(source_commit=AUTHORITATIVE_SOURCE_COMMIT)
    execution_widths = [1, 16] if engineering_smoke else list(contract["widths"])
    warmup_forwards = 1 if engineering_smoke else int(contract["measurement"]["warmup_forwards"])
    measurement_forwards = 2 if engineering_smoke else int(contract["measurement"]["measurement_forwards"])
    contract.update({
        "engineering_smoke_only": engineering_smoke,
        "scientific_benchmark": not engineering_smoke,
        "execution_widths": execution_widths,
        "warmup_forwards": warmup_forwards,
        "measurement_forwards": measurement_forwards,
    })
    _write(dataset / "L4_BENCHMARK_CONTRACT.json", contract)
    (dataset / "README.md").write_text(
        ("Private engineering-smoke package. No scientific aggregate, targets, candidate scoring, or competition output.\n"
         if engineering_smoke else
         "Private target-blind hardware characterization package. No targets, candidate scoring, or competition output.\n"),
        encoding="utf-8", newline="\n",
    )
    _write(dataset / "dataset-metadata.json", {
        "id": f"{owner}/{dataset_slug}",
        "title": ("ARC2 L4 native base B1-B16 engineering smoke source" if engineering_smoke
                  else "ARC2 L4 native base physical batch scaling source"),
        "subtitle": ("Target-blind B1/B16 orchestration and capacity smoke" if engineering_smoke
                     else "Target-blind stock-Transformers B1-B16 hardware characterization"),
        "licenses": [{"name": "other"}],
    })
    kernel.mkdir(parents=True)
    notebook_name = f"{kernel_slug}.ipynb"
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": [line + "\n" for line in _notebook_source(
                       dataset_slug, harness_commit, engineering_smoke=engineering_smoke
                   ).splitlines()]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"},
                     "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False,
                                "language": "python", "sourceType": "notebook"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    _assert_notebook_contract(notebook)
    _write(kernel / notebook_name, notebook)
    _write(kernel / "kernel-metadata.json", {
        "id": f"{owner}/{kernel_slug}",
        "title": ("ARC2 L4 Native B1/B16 Engineering Smoke V1" if engineering_smoke
                  else "ARC2 L4 Native Batch B1-B16 V1"),
        "code_file": notebook_name, "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": (["gpu", "benchmark", "engineering-smoke"] if engineering_smoke else ["gpu", "benchmark"]),
        "dataset_sources": [f"{owner}/{dataset_slug}"], "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"], "machine_shape": "NvidiaL4",
    })
    files = {str(path.relative_to(output)).replace("\\", "/"): _sha256(path)
             for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {
        "status": "PACKAGE_BUILT_NOT_PUSHED_NOT_LAUNCHED", "experiment": EXPERIMENT,
        "authoritative_source_commit": AUTHORITATIVE_SOURCE_COMMIT, "harness_commit": harness_commit,
        "runtime_dynamiccache_compatibility_sha256": RUNTIME_DYNAMICCACHE_COMPAT_SHA256,
        "dataset_slug": f"{owner}/{dataset_slug}", "kernel_slug": f"{owner}/{kernel_slug}",
        "engineering_smoke_only": engineering_smoke,
        "scientific_benchmark": not engineering_smoke,
        "execution_widths": execution_widths,
        "warmup_forwards": warmup_forwards,
        "measurement_forwards": measurement_forwards,
        "github_push": "BLOCKED_PENDING_TOKEN_ROTATION", "kaggle_run_started": False, "files": files,
    }
    _write(output / "PACKAGE_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--dataset-slug", default="arc2-l4-native-base-batch-scaling-b1-b16-v1-source")
    parser.add_argument("--kernel-slug")
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    kernel_slug = args.kernel_slug or (
        "arc2-l4-native-base-b1-b16-engineering-smoke-v1" if args.engineering_smoke
        else "arc2-l4-native-base-physical-batch-scaling-b1-b16-v1"
    )
    print(json.dumps(build(
        output=args.output, owner=args.owner, dataset_slug=args.dataset_slug, kernel_slug=kernel_slug,
        engineering_smoke=args.engineering_smoke,
    ), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
