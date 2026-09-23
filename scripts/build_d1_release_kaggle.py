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
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", commit], cwd=ROOT, stdout=handle, check=True)
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


def notebook(source_archive: str, source_sha256: str, config_name: str) -> dict[str, Any]:
    lines = [
        "import hashlib, json, os, shutil, subprocess, sys, tarfile",
        "from pathlib import Path",
        'work=Path("/kaggle/working"); source_archive=Path("' + source_archive + '")',
        'expected_source_sha256="' + source_sha256 + '"',
        'if not source_archive.is_file(): raise RuntimeError("explicit D1 source archive is missing")',
        'if hashlib.sha256(source_archive.read_bytes()).hexdigest()!=expected_source_sha256: raise RuntimeError("D1 source archive hash mismatch")',
        'root=work/"ARC2"; root.mkdir(exist_ok=False)',
        'with tarfile.open(source_archive) as archive: archive.extractall(root, filter="data")',
        'challenge=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")',
        'if not challenge.is_file(): raise RuntimeError("mounted competition challenge missing")',
        'config=Path("/kaggle/input/arc2-d1-release-source/' + config_name + '")',
        'if not config.is_file(): raise RuntimeError("explicit D1 release config missing")',
        'print(json.dumps({"event":"D1_RELEASE_SINGLE_INFERENCE_PATH","rerun_flag_observed":os.getenv("KAGGLE_IS_COMPETITION_RERUN", ""),"challenge":str(challenge),"source_sha256":expected_source_sha256},sort_keys=True),flush=True)',
        '# No fast-commit or dummy branch exists.  The release image supplies the verified CUDA worker bootstrap.',
        'raise RuntimeError("D1 live worker bootstrap must be bound in the verified release image before Kaggle execution")',
    ]
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in lines]}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--source-input-path", default="/kaggle/input/arc2-d1-release-source/ARC2.tar")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite release staging")
    config = json.loads(args.release_config.read_text(encoding="utf-8"))
    required = {"model_identity", "ttt24_recipe", "ttt48_recipe", "generation", "scoring"}
    if not required.issubset(config): raise ValueError("D1 release config missing required identity fields")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = args.output / "dataset"; archive = archive_source(payload / "ARC2.tar", commit)
    write(payload / "d1_release_config.json", config)
    write(payload / "SOURCE_MANIFEST.json", {"source_commit": commit, "archive": "ARC2.tar", "archive_sha256": sha256(archive), "config": "d1_release_config.json", "config_sha256": sha256(payload / "d1_release_config.json"), "archive_inspected": True})
    kernel = args.output / "kernel"; kernel.mkdir(parents=True)
    write(kernel / "arc2-d1-fixed4plus4-production.ipynb", notebook(args.source_input_path, sha256(archive), "d1_release_config.json"))
    write(kernel / "kernel-metadata.json", {"title": "ARC2 fixed 4+4 D1 release (review only)", "code_file": "arc2-d1-fixed4plus4-production.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False})
    print(json.dumps({"event": "D1_RELEASE_STAGING_READY", "source_commit": commit, "archive_sha256": sha256(archive), "archive_members_inspected": True}, sort_keys=True))


if __name__ == "__main__": main()
