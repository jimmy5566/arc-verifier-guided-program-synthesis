#!/usr/bin/env python3
"""Build, but never upload or launch, the isolated one-L4 batch sweep."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_l4_single_gpu_physical_batch_scaling_v1 as runner  # noqa: E402


EXPERIMENT = runner.EXPERIMENT
CONTRACT_FILE = "L4_SINGLE_GPU_BATCH_CONTRACT.json"
SOURCE_ARCHIVE = "ARC2-source.tar"
MODEL_SOURCE = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"


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


def _curated_paths() -> tuple[str, ...]:
    """The minimal import closure for the already-audited forward core."""
    return (
        "src/arc", "src/inference/__init__.py", "src/inference/arc_native_io.py",
        "src/inference/nvarc_native.py", "src/inference/nvarc_native_augmentation.py",
        "src/inference/nvarc_turbodfs_d1.py", "src/inference/nvarc_turbodfs_reference.py",
        "src/inference/nvarc_turbodfs_dynamic_ready.py",
        "configs/nvarc_native_846d0198",
    )


def _archive_source(destination: Path, *, source_ref: str) -> None:
    temporary = destination.parent / "source.tar"
    with temporary.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", source_ref, *_curated_paths()], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(temporary) as archive:
        archive.extractall(destination, filter="data")
    temporary.unlink()
    # Harness files deliberately come from this reviewed checkout.  Their
    # bytes are included in the source-tree manifest below.
    for name in (
        "run_l4_native_base_physical_batch_scaling_b1_b16_v1.py",
        "run_l4_single_gpu_physical_batch_scaling_v1.py",
        "l4_process_group_cleanup.py",
    ):
        target = destination / "scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)
    _write(destination / "HARNESS_REVIEW_IDENTITY.json", {
        "experiment": EXPERIMENT, "harness_commit": source_ref,
        "review_payload_schema": 1,
    })
    # Source modules may legitimately contain the word "solution" in a
    # filename.  Only mounted answer *artifacts* are prohibited.
    forbidden = [path for path in destination.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".csv", ".parquet"} and "solution" in path.name.lower()]
    if forbidden:
        raise RuntimeError(f"TARGET_ARTIFACT_IN_SOURCE_PAYLOAD:{forbidden}")


def _bundle_source(source: Path) -> Path:
    archive = source.parent / SOURCE_ARCHIVE
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as bundle:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            info = bundle.gettarinfo(str(path), arcname=(Path("ARC2") / path.relative_to(source)).as_posix())
            info.uid = info.gid = 0; info.uname = info.gname = ""; info.mtime = 0
            with path.open("rb") as handle:
                bundle.addfile(info, handle)
    if not archive.is_file() or not archive.stat().st_size:
        raise RuntimeError("SOURCE_ARCHIVE_NOT_CREATED")
    return archive


def _source_tree_hashes(source: Path) -> dict[str, str]:
    return {path.relative_to(source).as_posix(): _sha256(path) for path in sorted(source.rglob("*")) if path.is_file()}


def _assert_staged_runner_importable(source: Path) -> None:
    candidate = source / "scripts" / "run_l4_single_gpu_physical_batch_scaling_v1.py"
    environment = {**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"}
    probe = subprocess.run(
        [sys.executable, "-c", "import runpy, sys; sys.path.insert(0, sys.argv[1]); runpy.run_path(sys.argv[2])", str(source), str(candidate)],
        cwd=source, env=environment, text=True, capture_output=True,
    )
    if probe.returncode:
        raise RuntimeError(f"STAGED_RUNNER_IMPORT_CLOSURE_FAILED:\n{probe.stderr}")
    if list(source.rglob("__pycache__")):
        raise RuntimeError("STAGED_SOURCE_CONTAINS_BYTECODE")


def experiment_contract(*, source_commit: str) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "authoritative_source_commit": source_commit,
        "hardware": {"selected_cuda_visible_device": "0", "selected_torch_device": "cuda:0", "required_gpu_name": "NVIDIA L4", "gpu_count_used_for_timing": 1},
        "model": {"source": MODEL_SOURCE, "dtype": "BF16", "mode": "BASE_MODEL_ONLY", "model_load_count": 1},
        "root": {"task_id": "d59b0160", "output_index": 0, "depth": 24, "budget": 128, "view": "identity", "prefill_count": 1, "cpu_resident": True},
        "widths": list(runner.WIDTHS),
        "measurement": {"warmup_forwards": runner.WARMUPS, "measurement_forwards": runner.MEASUREMENTS, "repeat_order": [8, 4], "repeat_warmups": 1, "repeat_measurements": 6},
        "limits_seconds": {"global": runner.GLOBAL_LIMIT, "model_ready": runner.MODEL_READY_LIMIT, "no_progress": runner.NO_PROGRESS_LIMIT, "per_width": runner.WIDTH_LIMIT, "cleanup_grace": 10},
        "ttt_used": False, "peft_used": False, "gold_loaded": False, "submission_created": False,
        "real_8view_bucket_discovery_used": False, "cross_task_scheduler_used": False, "four_gpu_aggregation_used": False,
        "scientific_config_changed": False,
    }


def _notebook_source(harness_commit: str) -> str:
    """One outer bounded Popen; the child is a single-GPU, single-process run."""
    return "\n".join([
        "import hashlib, json, os, subprocess, sys, time",
        "from pathlib import Path",
        f"GLOBAL_LIMIT = {runner.GLOBAL_LIMIT}",
        "STARTED = time.monotonic()",
        "input_root = Path('/kaggle/input')",
        f"contract_paths = sorted(input_root.rglob({CONTRACT_FILE!r}))",
        "matches = []",
        "for path in contract_paths:",
        "    try: payload = json.loads(path.read_text())",
        "    except Exception: continue",
        f"    if payload.get('experiment') == {EXPERIMENT!r} and payload.get('authoritative_source_commit') == {harness_commit!r}: matches.append((path, payload))",
        "if len(matches) != 1: raise RuntimeError(f'expected exactly one mounted single-GPU source contract, found {[str(path) for path, _ in matches]}')",
        "contract_path, contract = matches[0]",
        "dataset = contract_path.parent",
        "out = Path('/kaggle/working/analysis/l4_single_gpu_physical_batch_scaling_v1')",
        "out.mkdir(parents=True, exist_ok=True)",
        "def atomic_json(path, payload):",
        "    temporary = path.with_suffix(path.suffix + '.tmp')",
        "    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')",
        "    os.replace(temporary, path)",
        "def smi():",
        "    try: return {'stdout': subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT, timeout=10)}",
        "    except Exception as exc: return {'error': repr(exc)}",
        "def last_progress():",
        "    path = out / 'PROGRESS.jsonl'",
        "    try:",
        "        lines = path.read_text().splitlines()",
        "        return json.loads(lines[-1]) if lines else None",
        "    except Exception as exc: return {'progress_read_error': repr(exc)}",
        "if os.environ.get('KAGGLE_KERNEL_INTERNET_ENABLED','').strip().lower() in {'1','true','yes'}: raise RuntimeError('INTERNET_MUST_BE_DISABLED')",
        "bundle = contract.get('source_bundle', {})",
        "if bundle.get('upload_file') != 'ARC2-source.tar' or bundle.get('mount_root') != 'ARC2-source' or bundle.get('root') != 'ARC2': raise RuntimeError('SOURCE_BUNDLE_CONTRACT_INVALID')",
        "source = dataset / bundle['mount_root'] / bundle['root']",
        "for relative, expected in contract.get('source_tree_files', {}).items():",
        "    rel = Path(relative)",
        "    if rel.is_absolute() or '..' in rel.parts: raise RuntimeError('UNSAFE_SOURCE_MANIFEST_PATH')",
        "    actual = source / rel",
        "    if not actual.is_file() or hashlib.sha256(actual.read_bytes()).hexdigest() != expected: raise RuntimeError(f'SOURCE_HASH_MISMATCH:{relative}')",
        "sys.path.insert(0, str(source))",
        "model = Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1')",
        "challenge = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json')",
        "native = source / 'configs' / 'nvarc_native_846d0198'",
        "runner = source / 'scripts' / 'run_l4_single_gpu_physical_batch_scaling_v1.py'",
        "missing = [str(path) for path in (model, model / 'config.json', challenge, native, runner) if not path.exists()]",
        "if missing: raise RuntimeError(f'BENCHMARK_INPUT_MISSING:{missing}')",
        "from scripts.l4_process_group_cleanup import cleanup_process_group, assert_no_process_group_survivors, pg_rows",
        "env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'0', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false', 'PYTHONUNBUFFERED':'1'}",
        "command = [sys.executable, '-u', str(runner), '--output', str(out), '--model-path', str(model), '--challenge', str(challenge), '--native-config-dir', str(native), '--benchmark-model-mode', 'BASE_MODEL_ONLY']",
        "print(json.dumps({'event':'NOTEBOOK_START','experiment':contract['experiment'],'command':command,'nvidia_smi':smi(),'cuda_visible_devices':env['CUDA_VISIBLE_DEVICES'],'submission_created':False}, sort_keys=True), flush=True)",
        "process = subprocess.Popen(command, env=env, start_new_session=True)",
        "pgid = os.getpgid(process.pid)",
        "if pgid != process.pid: raise RuntimeError(f'PROCESS_GROUP_SETUP_FAILED: leader={process.pid} pgid={pgid}')",
        "try:",
        "    returncode = process.wait(timeout=max(1.0, GLOBAL_LIMIT - (time.monotonic() - STARTED)))",
        "except subprocess.TimeoutExpired:",
        "    cleanup = cleanup_process_group(pgid, grace_seconds=10.0, kill_grace_seconds=10.0)",
        "    atomic_json(out / 'TIME_GATE_FAILURE.json', {'experiment':contract['experiment'],'reason':'GLOBAL_NOTEBOOK_LIMIT','last_progress_event':last_progress(),'nvidia_smi_snapshot':smi(), **cleanup})",
        "    if cleanup['remaining_surviving_pids']: raise RuntimeError(f'PROCESS_GROUP_SURVIVORS_AFTER_TIMEOUT:{cleanup}')",
        "    raise RuntimeError('GLOBAL_NOTEBOOK_TIME_GATE_EXCEEDED')",
        "survivors = pg_rows(pgid)",
        "if survivors:",
        "    cleanup = cleanup_process_group(pgid, grace_seconds=10.0, kill_grace_seconds=10.0)",
        "    atomic_json(out / 'NOTEBOOK_PROCESS_CLEANUP_AUDIT.json', {'returncode':returncode, 'reason':'PROCESS_GROUP_SURVIVOR', **cleanup})",
        "    raise RuntimeError(f'PROCESS_GROUP_SURVIVOR_AFTER_SUCCESS:{cleanup}')",
        "if returncode: raise RuntimeError(f'BENCHMARK_CHILD_FAILED:returncode={returncode}; last_progress={last_progress()}')",
        "assert_no_process_group_survivors(pgid)",
        "required = ['CONTRACT.json','SOURCE_IDENTITY.json','HARDWARE.json','RUNTIME_ENVIRONMENT.json','TIME_GATE_CONFIG.json','ROOT_TEMPLATE_AUDIT.json','PROGRESS.jsonl','L4_SINGLE_GPU_BATCH_RAW.csv','L4_SINGLE_GPU_BATCH_SUMMARY.csv','L4_SINGLE_GPU_BATCH_RATIOS.csv','L4_SINGLE_GPU_B4_B8_STABILITY.csv','B1_PARTIAL.json','B2_PARTIAL.json','B4_PARTIAL.json','B8_PARTIAL.json','B12_PARTIAL.json','B16_PARTIAL.json','DECISION.json','REPORT.md','PROCESS_CLEANUP_AUDIT.json','HASHES.json']",
        "missing_output = [name for name in required if not (out / name).is_file()]",
        "if missing_output: raise RuntimeError(f'BENCHMARK_OUTPUT_INCOMPLETE:{missing_output}')",
        "atomic_json(out / 'NOTEBOOK_PROCESS_CLEANUP_AUDIT.json', {'returncode':returncode,'pgid':pgid,'remaining_surviving_pids':[],'cleanup_complete':True})",
        "print(json.dumps({'event':'L4_SINGLE_GPU_BATCH_COMPLETE','artifacts':str(out),'submission_created':False}, sort_keys=True), flush=True)",
    ])


def _assert_notebook_contract(notebook: dict[str, Any]) -> None:
    source = "".join(notebook["cells"][0]["source"])
    compile(source, "generated_l4_single_gpu_batch.ipynb", "exec")
    forbidden = ("submission.json", "evaluation_solutions", "KAGGLE_IS_COMPETITION_RERUN", "FAST_COMMIT", "for gpu_id in range(4)")
    if any(token in source for token in forbidden):
        raise RuntimeError("NOTEBOOK_VIOLATES_SINGLE_GPU_TARGET_BLIND_CONTRACT")
    required = ("CUDA_VISIBLE_DEVICES':'0", "subprocess.Popen", "start_new_session=True", "cleanup_process_group", "GLOBAL_LIMIT = 6900", "BASE_MODEL_ONLY")
    if any(token not in source for token in required):
        raise RuntimeError("NOTEBOOK_SINGLE_GPU_TIME_GATE_CONTRACT_INCOMPLETE")


def build(*, output: Path, owner: str, dataset_slug: str, kernel_slug: str) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"REFUSING_TO_OVERWRITE_REVIEW_PACKAGE:{output}")
    if any(os.environ.get(name) for name in ("GH_TOKEN", "GITHUB_TOKEN")):
        raise RuntimeError("GITHUB_TOKEN_MUST_BE_UNSET_FOR_KAGGLE_PACKAGE")
    harness_commit = _git("rev-parse", "HEAD")
    dataset, kernel = output / "dataset", output / "kernel"
    source = dataset / "ARC2"; source.mkdir(parents=True)
    _archive_source(source, source_ref=harness_commit)
    _assert_staged_runner_importable(source)
    archive = _bundle_source(source)
    contract = experiment_contract(source_commit=harness_commit)
    contract["source_bundle"] = {"upload_file": archive.name, "mount_root": archive.stem, "root": "ARC2", "sha256": _sha256(archive)}
    contract["source_tree_files"] = _source_tree_hashes(source)
    _write(dataset / CONTRACT_FILE, contract)
    (dataset / "README.md").write_text("Private single-L4 physical-batch benchmark source. No targets, TTT, candidates, selector, or submission.\n", encoding="utf-8")
    _write(dataset / "dataset-metadata.json", {"id": f"{owner}/{dataset_slug}", "title": "ARC2 single L4 physical batch source", "licenses": [{"name": "other"}]})
    kernel.mkdir(parents=True)
    notebook_name = f"{kernel_slug}.ipynb"
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in _notebook_source(harness_commit).splitlines()]}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "NvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}
    _assert_notebook_contract(notebook)
    _write(kernel / notebook_name, notebook)
    _write(kernel / "kernel-metadata.json", {"id": f"{owner}/{kernel_slug}", "title": "ARC2 Single L4 Physical Batch Scaling V1", "code_file": notebook_name, "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu", "benchmark", "single-l4"], "dataset_sources": [f"{owner}/{dataset_slug}"], "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": [MODEL_SOURCE], "machine_shape": "NvidiaL4"})
    if notebook["metadata"]["kaggle"]["accelerator"] != "NvidiaL4" or json.loads((kernel / "kernel-metadata.json").read_text())["machine_shape"] != "NvidiaL4":
        raise RuntimeError("NVIDIAL4_METADATA_MISMATCH")
    files = {str(path.relative_to(output)).replace("\\", "/"): _sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"status": "PACKAGE_BUILT_NOT_PUSHED_NOT_LAUNCHED", "experiment": EXPERIMENT, "source_commit": harness_commit, "dataset_slug": f"{owner}/{dataset_slug}", "kernel_slug": f"{owner}/{kernel_slug}", "kaggle_run_started": False, "files": files}
    _write(output / "PACKAGE_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--dataset-slug", default="arc2-l4-single-gpu-batch-scaling-v1-source")
    parser.add_argument("--kernel-slug", default="arc2-l4-single-gpu-batch-scaling-v1")
    args = parser.parse_args()
    print(json.dumps(build(output=args.output, owner=args.owner, dataset_slug=args.dataset_slug, kernel_slug=args.kernel_slug), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
