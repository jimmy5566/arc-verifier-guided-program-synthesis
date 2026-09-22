"""Build a no-model Kaggle input-mount diagnostic notebook.

It is intentionally separate from inference packages and prints only mounted
input names plus shallow file inventory.  It never loads CUDA or a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--kernel-slug", required=True)
    parser.add_argument("--owner", default="jimmy5566")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite preflight package: {args.output}")
    args.output.mkdir(parents=True)
    code = '''import json
import os
from pathlib import Path

inputs = Path("/kaggle/input")
inventory = {}
for root, directories, files in os.walk(inputs):
    relative = Path(root).relative_to(inputs)
    if len(relative.parts) > 3:
        directories[:] = []
        continue
    inventory[str(relative)] = {"directories": sorted(directories)[:40], "files": sorted(files)[:40]}
print(json.dumps({"event": "KAGGLE_INPUT_MOUNT_PREFLIGHT", "cuda_or_model_loaded": False, "inputs": inventory}, sort_keys=True), flush=True)
'''
    notebook_name = f"{args.kernel_slug}.ipynb"
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}, "kaggle": {"isGpuEnabled": False, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    (args.output / notebook_name).write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": f"{args.owner}/{args.kernel_slug}", "title": "ARC2 Kaggle Input Mount Preflight",
        "code_file": notebook_name, "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [f"{args.owner}/{args.dataset_slug}"],
        "competition_sources": ["arc-prize-2026-arc-agi-2"],
    }
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
