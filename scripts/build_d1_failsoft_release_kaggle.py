#!/usr/bin/env python3
"""Build the isolated D1 fail-soft V2 candidate without publishing it."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("solution", "credential", "kaggle.json", ".safetensors", ".bin", ".pt", ".pth")
ALLOWED = (
    "scripts/run_d1_failsoft_4gpu.py", "scripts/build_d1_failsoft_submission.py",
    "scripts/run_d1_release_4gpu.py", "scripts/run_eval3_reference_ttt.py",
    "src/arc/__init__.py", "src/arc/io.py", "src/arc/task.py",
    "src/inference/__init__.py", "src/inference/arc_native_io.py",
    "src/inference/d1_release_contract.py", "src/inference/d1_failsoft_runtime.py",
    "src/inference/dynamic_task_scheduler.py", "src/inference/kaggle_l4_parallel_runner.py",
    "src/inference/nvarc_native.py", "src/inference/nvarc_native_augmentation.py",
    "src/inference/nvarc_native_candidates.py", "src/inference/selector_d1.py",
    "configs/nvarc_native_846d0198",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def archive_source(destination: Path, commit: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", commit, *ALLOWED], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(destination) as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        names = [member.name for member in members]
    if not names:
        raise RuntimeError("fail-soft source archive is empty")
    unsafe = [name for name in names if any(token in name.lower() for token in FORBIDDEN)]
    if unsafe:
        raise RuntimeError(f"forbidden fail-soft source members: {unsafe}")
    required = {
        "scripts/run_d1_failsoft_4gpu.py",
        "scripts/build_d1_failsoft_submission.py",
        "src/inference/d1_failsoft_runtime.py",
        "src/inference/selector_d1.py",
    }
    if not required.issubset(names):
        raise RuntimeError(f"fail-soft source archive lacks files: {sorted(required - set(names))}")
    return destination


def notebook(source_sha256: str) -> dict[str, Any]:
    lines = [
        "import hashlib, importlib.metadata as md, json, os, shutil, subprocess, sys",
        "from pathlib import Path",
        "def atomic_json(path,payload):",
        '    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp")',
        '    tmp.write_text(json.dumps(payload,sort_keys=True)+"\\n",encoding="utf-8"); os.replace(tmp,path)',
        "def parse_phase(raw):",
        '    value=(raw or "").strip().lower()',
        '    if value in {"","0","false","no"}: return "FAST_SAVE"',
        '    if value in {"1","true","yes"}: return "FULL_RERUN"',
        '    raise RuntimeError(f"unexpected rerun flag: {raw!r}")',
        "def fast_save(challenge,submission):",
        '    mounted=json.loads(challenge.read_text()); payload={task_id:[{"attempt_1":[[0]],"attempt_2":[[0]]} for _ in task["test"]] for task_id,task in mounted.items()}',
        "    atomic_json(submission,payload)",
        '    atomic_json(Path("/kaggle/working/artifacts/d1_failsoft/FAST_SAVE_PROVENANCE.json"),{"phase":"FAST_SAVE_ONLY","model_loaded":False,"submission_kind":"SAVE_ONLY_PLACEHOLDER","task_count":len(payload)})',
        '    print(json.dumps({"event":"D1_FAILSOFT_FAST_SAVE_COMPLETE","task_count":len(payload)},sort_keys=True),flush=True)',
        "def full_rerun(challenge,submission):",
        "    submission.unlink(missing_ok=True)",
        '    Path("/kaggle/working/artifacts/d1_failsoft/FAST_SAVE_PROVENANCE.json").unlink(missing_ok=True)',
        '    mounted=Path("/kaggle/input/datasets/jimmy5566/arc2-d1-failsoft-v2-source")',
        '    source_root=mounted/"ARC2"; manifest=json.loads((mounted/"SOURCE_MANIFEST.json").read_text())',
        f'    if manifest["archive_sha256"]!="{source_sha256}" or not source_root.is_dir(): raise RuntimeError("explicit fail-soft source identity mismatch")',
        '    actual_files={Path(base,name).relative_to(source_root).as_posix() for base,_,files in os.walk(source_root) for name in files}',
        '    if actual_files!=set(manifest["file_sha256"]): raise RuntimeError("fail-soft mounted source file set mismatch")',
        '    for name,expected in manifest["file_sha256"].items():',
        '        if hashlib.sha256((source_root/name).read_bytes()).hexdigest()!=expected: raise RuntimeError(f"fail-soft source hash mismatch: {name}")',
        '    root=Path("/kaggle/working/ARC2")',
        "    if root.exists(): shutil.rmtree(root)",
        "    shutil.copytree(source_root,root)",
        '    config=mounted/"d1_failsoft_release_config.json"',
        '    if hashlib.sha256(config.read_bytes()).hexdigest()!=manifest["config_sha256"]: raise RuntimeError("fail-soft config hash mismatch")',
        "    cfg=json.loads(config.read_text())",
        '    os.environ.update({"TRITON_PTXAS_PATH":cfg["environment"]["ptxas_path"],"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})',
        '    if not sys.version.startswith(cfg["environment"]["python_prefix"]): raise RuntimeError("Python mismatch")',
        '    for package in ("unsloth","unsloth-zoo","transformers","torch","torchao","peft","trl","triton"):',
        '        if md.version(package)!=cfg["environment"][package]: raise RuntimeError(f"dependency mismatch: {package}")',
        '    ptxas=Path(cfg["environment"]["ptxas_path"]); model=Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")',
        '    if not ptxas.is_file() or subprocess.run([str(ptxas),"--version"],capture_output=True).returncode: raise RuntimeError("verified ptxas unavailable")',
        '    out=Path("/kaggle/working/artifacts/d1_failsoft"); out.mkdir(parents=True,exist_ok=True)',
        '    candidates=out/"candidates_frozen.json"; selection=out/"d1_selection_frozen.json"; provenance=out/"PRODUCTION_PROVENANCE.json"',
        '    run=[sys.executable,str(root/"scripts/run_d1_failsoft_4gpu.py"),"--challenge",str(challenge),"--release-config",str(config),"--model-path",str(model),"--native-config-dir",str(root/"configs/nvarc_native_846d0198"),"--checkpoint-dir",str(out/"checkpoints"),"--output",str(candidates),"--resume"]',
        '    if subprocess.run(run,env=os.environ).returncode: raise RuntimeError("D1_FAILSOFT_WORKERS_FAILED")',
        '    final=[sys.executable,str(root/"scripts/build_d1_failsoft_submission.py"),"--challenge",str(challenge),"--release-config",str(config),"--records",str(candidates),"--selection-output",str(selection),"--provenance-output",str(provenance),"--output",str(submission)]',
        '    if subprocess.run(final,env={**os.environ,"CUDA_VISIBLE_DEVICES":""}).returncode: raise RuntimeError("D1_FAILSOFT_FINALIZATION_FAILED")',
        '    payload=json.loads(submission.read_text()); mounted_challenge=json.loads(challenge.read_text())',
        '    if set(payload)!=set(mounted_challenge) or any(len(payload[k])!=len(mounted_challenge[k]["test"]) for k in mounted_challenge): raise RuntimeError("final submission mapping mismatch")',
        '    print(json.dumps({"event":"D1_FAILSOFT_RELEASE_COMPLETE","task_count":len(payload),"solutions_opened":False},sort_keys=True),flush=True)',
        "def main():",
        '    challenge=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")',
        '    if not challenge.is_file(): raise RuntimeError("mounted competition challenge missing")',
        '    submission=Path("/kaggle/working/submission.json")',
        '    phase=parse_phase(os.getenv("KAGGLE_IS_COMPETITION_RERUN", ""))',
        '    fast_save(challenge,submission) if phase=="FAST_SAVE" else full_rerun(challenge,submission)',
        "main()",
    ]
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in lines]}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--source-commit", default="HEAD")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite fail-soft candidate")
    commit = subprocess.check_output(["git", "rev-parse", args.source_commit], cwd=ROOT, text=True).strip()
    config = json.loads(args.release_config.read_text(encoding="utf-8"))
    payload = args.output / "dataset"
    archive = archive_source(payload / "ARC2.tar", commit)
    write(payload / "d1_failsoft_release_config.json", config)
    with tarfile.open(archive) as source:
        hashes = {member.name: hashlib.sha256(source.extractfile(member).read()).hexdigest() for member in source.getmembers() if member.isfile()}
    manifest = {
        "release_candidate": "d1-failsoft-v2-candidate",
        "source_commit": commit,
        "archive": "ARC2.tar",
        "archive_sha256": sha256(archive),
        "file_sha256": hashes,
        "config": "d1_failsoft_release_config.json",
        "config_sha256": sha256(payload / "d1_failsoft_release_config.json"),
        "scientific_algorithm_changed": False,
        "runtime_resilience_changed": True,
        "published": False,
    }
    write(payload / "SOURCE_MANIFEST.json", manifest)
    write(payload / "dataset-metadata.json", {"id": "jimmy5566/arc2-d1-failsoft-v2-source", "title": "ARC2 D1 fail-soft V2 candidate source", "licenses": [{"name": "CC0-1.0"}], "isPrivate": True})
    kernel = args.output / "notebook"; kernel.mkdir(parents=True)
    write(kernel / "arc2-d1-failsoft-v2-candidate.ipynb", notebook(sha256(archive)))
    write(kernel / "kernel-metadata.json", {
        "id": "jimmy5566/arc2-d1-failsoft-v2-candidate",
        "title": "ARC2 D1 fail-soft V2 candidate",
        "code_file": "arc2-d1-failsoft-v2-candidate.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "dataset_sources": ["jimmy5566/arc2-d1-failsoft-v2-source"],
        "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
        "machine_shape": "NvidiaL4",
    })
    write(args.output / "CANDIDATE_MANIFEST.json", {
        "release_id": "d1-failsoft-v2-candidate",
        "source_commit": commit,
        "scientific_algorithm_changed": False,
        "runtime_resilience_changed": True,
        "kaggle_published": False,
        "competition_submitted": False,
        "source_archive_sha256": sha256(archive),
        "config_sha256": manifest["config_sha256"],
    })
    print(json.dumps({"event": "D1_FAILSOFT_CANDIDATE_STAGED", "source_commit": commit, "archive_sha256": sha256(archive), "published": False}, sort_keys=True))


if __name__ == "__main__":
    main()

