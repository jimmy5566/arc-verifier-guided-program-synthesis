#!/usr/bin/env python3
"""Build the private, time-gated B4/B8 root-bucket benchmark package only.

The builder deliberately has no Kaggle API call.  The generated notebook uses
an outer process group and bounded Popen waits so a failed controller cannot
retain GPUs through orphan torch-loader children.
"""
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

from scripts.run_l4_8view_root_bucket_b4_b8_scaling_v1 import (  # noqa: E402
    AUTHORITATIVE_SOURCE_COMMIT,
    EXPERIMENT,
    GLOBAL_NOTEBOOK_BENCHMARK_LIMIT_SECONDS,
    MODEL_READY_LIMIT_SECONDS,
    NO_PROGRESS_LIMIT_SECONDS,
    WIDTH_HARD_LIMIT_SECONDS,
    experiment_contract,
)


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
        raise RuntimeError("frozen authoritative physical-forward core differs from its pinned commit")
    path = ROOT / "src" / "inference" / "nvarc_turbodfs_dynamic_ready.py"
    if _sha256(path) != RUNTIME_DYNAMICCACHE_COMPAT_SHA256:
        raise RuntimeError("DynamicCache runtime compatibility source differs from the pinned patch")


def _curated_paths() -> tuple[str, ...]:
    return (
        "src/arc", "src/inference/__init__.py", "src/inference/arc_native_io.py",
        "src/inference/nvarc_native.py", "src/inference/nvarc_native_augmentation.py",
        "src/inference/nvarc_turbodfs_d1.py", "src/inference/nvarc_turbodfs_reference.py",
        "src/inference/nvarc_turbodfs_dynamic_ready.py", "src/inference/d1_release_contract.py",
        "src/inference/selector_d1.py",
        "configs/nvarc_native_846d0198",
        "scripts/run_clean_hf_parallel_dfs_b4_compute_v1.py",
        "scripts/run_clean_hf_parallel_dfs_batch_scaling_v2.py",
        "scripts/run_clean_hf_b16_transient_memory_opt_v1.py",
    )


def _archive_source(destination: Path, *, source_ref: str) -> None:
    archive = destination.parent / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", source_ref, *_curated_paths()], cwd=ROOT,
                       stdout=handle, check=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter="data")
    archive.unlink()
    # Harness-only orchestration is intentionally copied from the reviewed
    # checkout; it is separately SHA-manifested in the dataset contract.
    for name in (
        "run_l4_native_base_physical_batch_scaling_b1_b16_v1.py",
        "run_l4_8view_root_bucket_b4_b8_scaling_v1.py",
        "l4_process_group_cleanup.py",
    ):
        target = destination / "scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / name, target)
    # Bind each newly built review payload to the exact reviewed harness
    # revision.  This keeps a source archive from being mistaken for a prior
    # review package merely because the curated scientific core is unchanged.
    _write(destination / "HARNESS_REVIEW_IDENTITY.json", {
        "harness_commit": source_ref,
        "experiment": EXPERIMENT,
        "review_payload_schema": 1,
    })
    forbidden = [path for path in destination.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".csv", ".parquet"}
                 and "solution" in path.name.lower()]
    if forbidden:
        raise RuntimeError(f"curated target-blind source contains forbidden artifacts: {forbidden}")


def _bundle_source(source: Path) -> Path:
    archive = source.parent / "ARC2-source.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as bundle:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            info = bundle.gettarinfo(str(path), arcname=(Path("ARC2") / path.relative_to(source)).as_posix())
            info.uid = info.gid = 0; info.uname = info.gname = ""; info.mtime = 0
            with path.open("rb") as handle:
                bundle.addfile(info, handle)
    if not archive.is_file() or not archive.stat().st_size:
        raise RuntimeError("curated source archive was not created")
    return archive


def _source_tree_hashes(source: Path) -> dict[str, str]:
    return {path.relative_to(source).as_posix(): _sha256(path) for path in sorted(source.rglob("*")) if path.is_file()}


def _assert_staged_runner_importable(source: Path) -> None:
    runner = source / "scripts" / "run_l4_8view_root_bucket_b4_b8_scaling_v1.py"
    environment = {**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"}
    probe = subprocess.run([sys.executable, "-c", "import runpy, sys; sys.path.insert(0, sys.argv[1]); runpy.run_path(sys.argv[2])",
                            str(source), str(runner)], cwd=source, env=environment, text=True, capture_output=True)
    if probe.returncode:
        raise RuntimeError(f"curated runner import closure failed:\n{probe.stderr}")
    if list(source.rglob("__pycache__")):
        raise RuntimeError("curated source unexpectedly contains bytecode")


def _notebook_source(harness_commit: str) -> str:
    """One cell whose outer Popen loop owns each child process group."""
    return "\n".join([
        "import hashlib, json, os, signal, subprocess, sys, time",
        "from pathlib import Path",
        f"GLOBAL_LIMIT = {GLOBAL_NOTEBOOK_BENCHMARK_LIMIT_SECONDS}",
        f"MODEL_READY_LIMIT = {MODEL_READY_LIMIT_SECONDS}",
        f"NO_PROGRESS_LIMIT = {NO_PROGRESS_LIMIT_SECONDS}",
        f"WIDTH_HARD_LIMIT = {WIDTH_HARD_LIMIT_SECONDS}",
        "STARTED = time.monotonic()",
        "input_root = Path('/kaggle/input')",
        "contract_paths = sorted(input_root.rglob('L4_8VIEW_ROOT_BUCKET_B4_B8_CONTRACT.json'))",
        "matches = []",
        "for path in contract_paths:",
        "    try: candidate = json.loads(path.read_text())",
        "    except Exception: continue",
        f"    if candidate.get('experiment') == {EXPERIMENT!r} and candidate.get('authoritative_source_commit') == {AUTHORITATIVE_SOURCE_COMMIT!r}: matches.append((path, candidate))",
        "if len(matches) != 1: raise RuntimeError(f'expected one exact B4/B8 source contract, found {[str(path) for path, _ in matches]}')",
        "contract_path, contract = matches[0]",
        "dataset = contract_path.parent",
        "out = Path('/kaggle/working/analysis/l4_8view_root_bucket_b4_b8_scaling_v1')",
        "out.mkdir(parents=True, exist_ok=True)",
        "def atomic_json(path, payload):",
        "    tmp = path.with_suffix(path.suffix + '.tmp')",
        "    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')",
        "    os.replace(tmp, path)",
        "def smi():",
        "    try: return {'stdout': subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT, timeout=10)}",
        "    except Exception as exc: return {'error': repr(exc)}",
        "def last_progress():",
        "    path = out / 'PROGRESS.jsonl'",
        "    try:",
        "        lines = path.read_text().splitlines()",
        "        return json.loads(lines[-1]) if lines else None",
        "    except Exception as exc: return {'progress_read_error': repr(exc)}",
        "sys.path.insert(0, str(source)) if 'source' in globals() else None",
        "def kill_group(pgid, phase, hard_limit, *, reason='TIME_GATE_FAILURE'):",
        "    from scripts.l4_process_group_cleanup import cleanup_process_group",
        "    snapshot = smi()",
        "    cleanup = cleanup_process_group(pgid)",
        "    payload = {'experiment': contract['experiment'], 'phase': phase, 'width_bucket': None, 'elapsed_seconds': time.monotonic()-STARTED, 'hard_limit_seconds': hard_limit, 'last_progress_event': last_progress(), 'worker_pids': [], 'worker_status': [], 'nvidia_smi_snapshot': snapshot, 'reason': reason, **cleanup}",
        "    atomic_json(out / 'TIME_GATE_FAILURE.json', payload)",
        "    print(json.dumps({'event':reason,'phase':phase, **payload}, sort_keys=True), flush=True)",
        "    return payload",
        "if os.environ.get('KAGGLE_KERNEL_INTERNET_ENABLED','').strip().lower() in {'1','true','yes'}: raise RuntimeError('Internet must be disabled')",
        "gpus = subprocess.check_output(['nvidia-smi','-L'], text=True).splitlines()",
        "if len(gpus) != 4 or any('NVIDIA L4' not in row for row in gpus): raise RuntimeError(f'requires exactly four NVIDIA L4 GPUs: {gpus}')",
        "bundle = contract.get('source_bundle', {})",
        "source = dataset / str(bundle.get('mount_root','')) / str(bundle.get('root',''))",
        "files = contract.get('source_tree_files')",
        "if bundle.get('upload_file') != 'ARC2-source.tar' or bundle.get('mount_root') != 'ARC2-source' or bundle.get('root') != 'ARC2' or not isinstance(files, dict): raise RuntimeError('attached source contract malformed')",
        "for relative, expected in files.items():",
        "    rel = Path(relative)",
        "    if rel.is_absolute() or '..' in rel.parts: raise RuntimeError('unsafe source manifest path')",
        "    path = source / rel",
        "    if not path.is_file(): raise RuntimeError(f'attached source file missing: {relative}')",
        "    digest = hashlib.sha256(path.read_bytes()).hexdigest()",
        "    if digest != expected: raise RuntimeError(f'attached source hash mismatch: {relative}')",
        "sys.path.insert(0, str(source))",
        "model = Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1')",
        "challenge = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json')",
        "native = source / 'configs' / 'nvarc_native_846d0198'",
        "runner = source / 'scripts' / 'run_l4_8view_root_bucket_b4_b8_scaling_v1.py'",
        "missing = [str(path) for path in (model, challenge, native, runner) if not path.exists()]",
        "if missing: raise RuntimeError(f'benchmark inputs missing: {missing}')",
        "env = {**os.environ, 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}",
        "common = [sys.executable, '-u', str(runner), '--output', str(out), '--source-commit', contract['authoritative_source_commit'], '--harness-commit', " + repr(harness_commit) + ", '--model-path', str(model), '--challenge', str(challenge), '--native-config-dir', str(native), '--benchmark-model-mode', 'BASE_MODEL_ONLY', '--global-limit-seconds', str(GLOBAL_LIMIT), '--model-ready-limit-seconds', str(MODEL_READY_LIMIT), '--no-progress-limit-seconds', str(NO_PROGRESS_LIMIT), '--width-hard-limit-seconds', str(WIDTH_HARD_LIMIT)]",
        "atomic_json(out / 'TIME_GATE_CONFIG.json', {'experiment':contract['experiment'],'global_notebook_benchmark_limit_seconds':GLOBAL_LIMIT,'model_ready_limit_seconds':MODEL_READY_LIMIT,'no_progress_limit_seconds':NO_PROGRESS_LIMIT,'width_hard_limit_seconds':WIDTH_HARD_LIMIT,'outer_process_group_kill':True})",
        "print(json.dumps({'event':'NOTEBOOK_START','experiment':contract['experiment'],'source':str(source),'runner':str(runner),'hardware':gpus,'global_limit_seconds':GLOBAL_LIMIT}, sort_keys=True), flush=True)",
        "for phase in ('preflight','controller'):",
        "    remaining = GLOBAL_LIMIT - (time.monotonic() - STARTED)",
        "    if remaining <= 0: raise RuntimeError('GLOBAL_TIME_GATE_BEFORE_PHASE')",
        "    command = [*common, '--phase', phase]",
        "    print(json.dumps({'event':'NOTEBOOK_PHASE_START','phase':phase,'command':command}, sort_keys=True), flush=True)",
        "    process = subprocess.Popen(command, env=env, start_new_session=True)",
        "    pgid = os.getpgid(process.pid)",
        "    if pgid != process.pid: raise RuntimeError(f'PROCESS_GROUP_SETUP_FAILED: leader={process.pid} pgid={pgid}')",
        "    try: returncode = process.wait(timeout=remaining)",
        "    except subprocess.TimeoutExpired:",
        "        cleanup = kill_group(pgid, phase, int(remaining))",
        "        if cleanup['remaining_surviving_pids']: raise RuntimeError(f'PROCESS_GROUP_SURVIVORS_AFTER_TIMEOUT:{cleanup}')",
        "        raise RuntimeError(f'TIME_GATE_EXCEEDED:{phase}')",
        "    if returncode:",
        "        from scripts.l4_process_group_cleanup import pg_rows",
        "        survivors = pg_rows(pgid)",
        "        if survivors:",
        "            cleanup = kill_group(pgid, phase, int(remaining), reason='PROCESS_GROUP_SURVIVOR_AFTER_FAILURE')",
        "            if cleanup['remaining_surviving_pids']: raise RuntimeError(f'PROCESS_GROUP_SURVIVORS_AFTER_FAILURE:{cleanup}')",
        "        listing = sorted(path.name for path in out.iterdir()) if out.exists() else []",
        "        print(json.dumps({'event':'NOTEBOOK_PHASE_ERROR','phase':phase,'returncode':returncode,'output_listing':listing}, sort_keys=True), flush=True)",
        "        raise RuntimeError(f'benchmark child failed: phase={phase} returncode={returncode}')",
        "    from scripts.l4_process_group_cleanup import assert_no_process_group_survivors",
        "    try:",
        "        assert_no_process_group_survivors(pgid)",
        "    except RuntimeError:",
        "        cleanup = kill_group(pgid, phase, int(remaining), reason='PROCESS_GROUP_SURVIVOR_AFTER_SUCCESS')",
        "        raise RuntimeError(f'PROCESS_GROUP_SURVIVOR_AFTER_SUCCESS:{cleanup}')",
        "    print(json.dumps({'event':'NOTEBOOK_PHASE_DONE','phase':phase}, sort_keys=True), flush=True)",
        "required = ['CONTRACT.json','SOURCE_IDENTITY.json','TIME_GATE_CONFIG.json','EIGHT_VIEW_ROOT_BUCKETS.json','PROCESS_CLEANUP_AUDIT.json','L4_8VIEW_ROOT_BUCKET_RAW.csv','L4_8VIEW_ROOT_BUCKET_PER_GPU.csv','L4_8VIEW_ROOT_BUCKET_AGGREGATE.csv','L4_8VIEW_ROOT_BUCKET_BOOTSTRAP.csv','DECISION.json','HASHES.json']",
        "missing_output = [name for name in required if not (out / name).is_file()]",
        "if missing_output: raise RuntimeError(f'benchmark output incomplete: {missing_output}')",
        "print(json.dumps({'event':'L4_8VIEW_ROOT_BUCKET_COMPLETE','artifacts':str(out),'submission_created':False}, sort_keys=True), flush=True)",
    ])


def _assert_notebook_contract(notebook: dict[str, Any]) -> None:
    source = "".join(notebook["cells"][0]["source"])
    forbidden = ("submission.json", "KAGGLE_IS_COMPETITION_RERUN", "FAST_COMMIT", "evaluation_solutions")
    if any(token in source for token in forbidden):
        raise RuntimeError("review notebook violates target-blind/no-submission contract")
    required = ("subprocess.Popen", "start_new_session=True", "cleanup_process_group",
                "assert_no_process_group_survivors", "TIME_GATE_FAILURE.json", "GLOBAL_LIMIT = 1800", "MODEL_READY_LIMIT = 300",
                "NO_PROGRESS_LIMIT = 180", "WIDTH_HARD_LIMIT = 600", "BASE_MODEL_ONLY", "PROCESS_GROUP_SURVIVOR_AFTER_SUCCESS")
    if any(token not in source for token in required):
        raise RuntimeError("time-gated notebook contract incomplete")


def build(*, output: Path, owner: str, dataset_slug: str, kernel_slug: str) -> dict[str, Any]:
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
    source = dataset / "ARC2"; source.mkdir(parents=True)
    _archive_source(source, source_ref=harness_commit)
    _assert_staged_runner_importable(source)
    archive = _bundle_source(source)
    contract = experiment_contract(source_commit=AUTHORITATIVE_SOURCE_COMMIT)
    contract["source_bundle"] = {"upload_file": archive.name, "mount_root": archive.stem, "root": "ARC2", "sha256": _sha256(archive)}
    contract["source_tree_files"] = _source_tree_hashes(source)
    _write(dataset / "L4_8VIEW_ROOT_BUCKET_B4_B8_CONTRACT.json", contract)
    (dataset / "README.md").write_text("Private target-blind B4/B8 hardware benchmark source. No targets, TTT, candidates, selectors, or submission.\n", encoding="utf-8")
    _write(dataset / "dataset-metadata.json", {"id": f"{owner}/{dataset_slug}", "title": "ARC2 L4 8View root bucket B4 B8 source", "licenses": [{"name": "other"}]})
    kernel.mkdir(parents=True)
    notebook_name = f"{kernel_slug}.ipynb"
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                             "source": [line + "\n" for line in _notebook_source(harness_commit).splitlines()]}],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12"},
                             "kaggle": {"accelerator": "NvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False,
                                        "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}
    _assert_notebook_contract(notebook)
    if notebook["metadata"]["kaggle"]["accelerator"] != "NvidiaL4":
        raise RuntimeError("notebook accelerator must be exactly NvidiaL4")
    _write(kernel / notebook_name, notebook)
    _write(kernel / "kernel-metadata.json", {"id": f"{owner}/{kernel_slug}", "title": "ARC2 L4 8View Root Bucket B4 B8 Scaling V1", "code_file": notebook_name,
                                               "language": "python", "kernel_type": "notebook", "is_private": True,
                                               "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                                               "keywords": ["gpu", "benchmark", "time-gated"], "dataset_sources": [f"{owner}/{dataset_slug}"],
                                               "competition_sources": ["arc-prize-2026-arc-agi-2"],
                                               "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"], "machine_shape": "NvidiaL4"})
    if json.loads((kernel / "kernel-metadata.json").read_text(encoding="utf-8"))["machine_shape"] != "NvidiaL4":
        raise RuntimeError("kernel machine_shape must be exactly NvidiaL4")
    files = {str(path.relative_to(output)).replace("\\", "/"): _sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"status": "PACKAGE_BUILT_NOT_PUSHED_NOT_LAUNCHED", "experiment": EXPERIMENT,
                "authoritative_source_commit": AUTHORITATIVE_SOURCE_COMMIT, "harness_commit": harness_commit,
                "runtime_dynamiccache_compatibility_sha256": RUNTIME_DYNAMICCACHE_COMPAT_SHA256,
                "dataset_slug": f"{owner}/{dataset_slug}", "kernel_slug": f"{owner}/{kernel_slug}",
                "kaggle_run_started": False, "files": files}
    _write(output / "PACKAGE_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--dataset-slug", default="arc2-l4-8view-root-b4-b8-scaling-v1-source")
    parser.add_argument("--kernel-slug", default="arc2-l4-8view-root-b4-b8-scaling-v1")
    args = parser.parse_args()
    print(json.dumps(build(output=args.output, owner=args.owner, dataset_slug=args.dataset_slug,
                           kernel_slug=args.kernel_slug), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
