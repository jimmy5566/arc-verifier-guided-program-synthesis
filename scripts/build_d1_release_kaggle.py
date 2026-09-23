#!/usr/bin/env python3
"""Build the isolated fixed-4+4+D1 Kaggle release payload.

This builder is intentionally transport-only: it makes source/artifact paths
and hashes explicit and refuses unsafe archive members.  It never performs a
model load or a Kaggle submission.
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
FORBIDDEN = ("solution", "credential", "kaggle.json", ".safetensors", ".bin", ".pt", ".pth")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def archive_source(destination: Path, commit: str) -> Path:
    allowed = (
        "scripts/run_d1_release_4gpu.py", "scripts/build_d1_release_submission.py", "scripts/run_eval3_reference_ttt.py",
        "src/arc/__init__.py", "src/arc/io.py", "src/arc/task.py",
        "src/inference/__init__.py", "src/inference/arc_native_io.py", "src/inference/d1_release_contract.py",
        "src/inference/dynamic_task_scheduler.py", "src/inference/kaggle_l4_parallel_runner.py",
        "src/inference/nvarc_native.py", "src/inference/nvarc_native_augmentation.py",
        "src/inference/nvarc_native_candidates.py", "src/inference/selector_d1.py",
        "configs/nvarc_native_846d0198",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", commit, *allowed], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(destination) as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
    if not names:
        raise RuntimeError("source archive is empty")
    unsafe = [name for name in names if any(part in name.lower() for part in FORBIDDEN)]
    if unsafe:
        raise RuntimeError("forbidden production source archive members: " + ", ".join(unsafe))
    required = {"scripts/run_d1_release_4gpu.py", "scripts/build_d1_release_submission.py", "src/inference/d1_release_contract.py", "src/inference/selector_d1.py"}
    if not required.issubset(names):
        raise RuntimeError("source archive lacks D1 release files: " + repr(sorted(required - set(names))))
    return destination


def notebook(source_archive: str, source_sha256: str, config_name: str, *, smoke_task_ids: tuple[str, ...] = ()) -> dict[str, Any]:
    lines = [
        "import hashlib, json, os, shutil, subprocess, sys",
        "from pathlib import Path",
        'work=Path("/kaggle/working"); source_archive=Path("' + source_archive + '")',
        'expected_source_sha256="' + source_sha256 + '"',
        'source_root=source_archive.parent/"ARC2"; source_manifest=json.loads((source_archive.parent/"SOURCE_MANIFEST.json").read_text())',
        'if source_manifest["archive_sha256"]!=expected_source_sha256 or not source_root.is_dir(): raise RuntimeError("explicit D1 source identity/mount mismatch")',
        'expected_files=source_manifest["file_sha256"]; mounted_files={Path(base,name).relative_to(source_root).as_posix() for base,_,files in os.walk(source_root) for name in files}',
        'if mounted_files!=set(expected_files): raise RuntimeError("D1 mounted source file set mismatch")',
        'for name,expected in expected_files.items():',
        '    if hashlib.sha256((source_root/name).read_bytes()).hexdigest()!=expected: raise RuntimeError(f"D1 mounted source hash mismatch: {name}")',
        'root=work/"ARC2"; shutil.copytree(source_root,root)',
        'challenge=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")',
        'config=Path("/kaggle/input/datasets/jimmy5566/arc2-d1-release-source/' + config_name + '")',
        'if not config.is_file(): raise RuntimeError("explicit D1 release config missing")',
        'if hashlib.sha256(config.read_bytes()).hexdigest()!=source_manifest["config_sha256"]: raise RuntimeError("D1 release config hash mismatch")',
        'cfg=json.loads(config.read_text())',
        'if cfg["environment"].get("bootstrap_mode")!="pinned_kaggle_image_offline": raise RuntimeError("unverified offline bootstrap mode")',
        'os.environ.update({"TRITON_PTXAS_PATH":cfg["environment"]["ptxas_path"],"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})',
        'print(json.dumps({"event":"D1_OFFLINE_REFERENCE_ENV_BOOTSTRAPPED","mode":"pinned_kaggle_image_offline","ptxas":os.environ["TRITON_PTXAS_PATH"]},sort_keys=True),flush=True)',
        'import importlib.metadata as md',
        'if not sys.version.startswith(cfg["environment"]["python_prefix"]): raise RuntimeError(f"Python mismatch: {sys.version}")',
        'for package in ("unsloth","unsloth-zoo","transformers","torch","torchao","peft","trl","triton"):',
        '    if md.version(package)!=cfg["environment"][package]: raise RuntimeError(f"frozen dependency mismatch: {package}")',
        'ptxas=Path(cfg["environment"]["ptxas_path"]); model=Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")',
        'if not ptxas.is_file() or subprocess.run([str(ptxas),"--version"],capture_output=True).returncode: raise RuntimeError("verified ptxas unavailable")',
        'if not model.is_dir(): raise RuntimeError("explicit model mount missing")',
        'out=work/"artifacts"/"d1_release"; out.mkdir(parents=True,exist_ok=True); candidates=out/"candidates_frozen.json"; selection=out/"d1_selection_frozen.json"; provenance=out/"PRODUCTION_PROVENANCE.json"; submission=work/"submission.json"',
        'print(json.dumps({"event":"D1_RELEASE_SINGLE_INFERENCE_PATH","rerun_flag_observed":os.getenv("KAGGLE_IS_COMPETITION_RERUN", ""),"challenge":str(challenge),"source_sha256":expected_source_sha256,"model":str(model),"portfolio":cfg["generation"]["portfolio"]},sort_keys=True),flush=True)',
        'run=[sys.executable,str(root/"scripts"/"run_d1_release_4gpu.py"),"--challenge",str(challenge),"--release-config",str(config),"--model-path",str(model),"--native-config-dir",str(root/"configs"/"nvarc_native_846d0198"),"--checkpoint-dir",str(out/"checkpoints"),"--output",str(candidates),"--resume"]',
        'if subprocess.run(run,env={**os.environ,"TRITON_PTXAS_PATH":str(ptxas),"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}).returncode: raise RuntimeError("D1_REAL_WORKERS_FAILED")',
        'finalize=[sys.executable,str(root/"scripts"/"build_d1_release_submission.py"),"--challenge",str(challenge),"--release-config",str(config),"--records",str(candidates),"--selection-output",str(selection),"--provenance-output",str(provenance),"--output",str(submission)]',
        'if subprocess.run(finalize,env={**os.environ,"CUDA_VISIBLE_DEVICES":""}).returncode: raise RuntimeError("D1_PER_OUTPUT_FINALIZATION_FAILED")',
        'if not submission.is_file(): raise RuntimeError("D1 submission missing")',
        'payload=json.loads(submission.read_text()); mounted=json.loads(challenge.read_text())',
        'if set(payload)!=set(mounted) or any(len(payload[k])!=len(mounted[k]["test"]) for k in mounted): raise RuntimeError("runtime challenge/submission mapping mismatch")',
        'print(json.dumps({"event":"D1_RELEASE_COMPLETE","task_count":len(payload),"test_output_count":sum(len(v) for v in payload.values()),"submission_sha256":hashlib.sha256(submission.read_bytes()).hexdigest(),"solutions_opened":False},sort_keys=True),flush=True)',
    ]
    if smoke_task_ids:
        if len(smoke_task_ids) != 2 or len(set(smoke_task_ids)) != 2:
            raise ValueError("the D1 live harness requires exactly two predeclared tasks")
        lines.insert(lines.index('config=Path("/kaggle/input/datasets/jimmy5566/arc2-d1-release-source/' + config_name + '")'),
            'mounted_eval=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"); eval_tasks=json.loads(mounted_eval.read_text()); smoke_ids=' + repr(smoke_task_ids) + '; assert all(task_id in eval_tasks for task_id in smoke_ids); challenge=work/"d1_smoke_challenges.json"; challenge.write_text(json.dumps({task_id:eval_tasks[task_id] for task_id in smoke_ids},sort_keys=True))')
        lines.insert(lines.index('import importlib.metadata as md'),
            'cfg["runtime"]["hard_deadline_seconds"]=900; cfg["runtime"]["finalization_margin_seconds"]=120; config=work/"d1_smoke_config.json"; config.write_text(json.dumps(cfg,sort_keys=True))')
        lines = [line.replace('submission=work/"submission.json"', 'submission=out/"smoke_submission.json"') for line in lines]
        lines[-1] = lines[-1].replace('"D1_RELEASE_COMPLETE"', '"D1_SMOKE_COMPLETE"')
    else:
        lines.insert(lines.index('config=Path("/kaggle/input/datasets/jimmy5566/arc2-d1-release-source/' + config_name + '")'), 'if not challenge.is_file(): raise RuntimeError("mounted competition challenge missing")')
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in lines]}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--source-input-path", default="/kaggle/input/datasets/jimmy5566/arc2-d1-release-source/ARC2.tar")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite release staging")
    config = json.loads(args.release_config.read_text(encoding="utf-8"))
    required = {"model_identity", "ttt24_recipe", "ttt48_recipe", "generation", "scoring"}
    if not required.issubset(config): raise ValueError("D1 release config missing required identity fields")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = args.output / "dataset"; archive = archive_source(payload / "ARC2.tar", commit)
    write(payload / "d1_release_config.json", config)
    with tarfile.open(archive) as source:
        file_hashes = {member.name: hashlib.sha256(source.extractfile(member).read()).hexdigest() for member in source.getmembers() if member.isfile()}
    write(payload / "SOURCE_MANIFEST.json", {"source_commit": commit, "archive": "ARC2.tar", "archive_sha256": sha256(archive), "file_sha256": file_hashes, "config": "d1_release_config.json", "config_sha256": sha256(payload / "d1_release_config.json"), "archive_inspected": True})
    write(payload / "dataset-metadata.json", {"id": "jimmy5566/arc2-d1-release-source", "title": "ARC2 D1 release source", "licenses": [{"name": "CC0-1.0"}], "isPrivate": True})
    kernel = args.output / "kernel"; kernel.mkdir(parents=True)
    write(kernel / "arc2-d1-fixed4plus4-production.ipynb", notebook(args.source_input_path, sha256(archive), "d1_release_config.json"))
    write(kernel / "kernel-metadata.json", {"id": "jimmy5566/arc2-fixed4plus4-d1-release", "title": "ARC2 fixed 4+4 D1 release", "code_file": "arc2-d1-fixed4plus4-production.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False, "dataset_sources": ["jimmy5566/arc2-d1-release-source"], "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"], "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"], "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868", "machine_shape": "NvidiaL4"})
    print(json.dumps({"event": "D1_RELEASE_STAGING_READY", "source_commit": commit, "archive_sha256": sha256(archive), "archive_members_inspected": True}, sort_keys=True))


if __name__ == "__main__": main()
