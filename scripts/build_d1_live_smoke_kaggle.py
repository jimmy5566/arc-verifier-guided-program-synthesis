#!/usr/bin/env python3
"""Package a separate two-task live harness around the unchanged D1 route."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_d1_release_kaggle import notebook, sha256, write


TASK_IDS = ("58490d8a", "f931b4a8")  # frozen single-output + multi-output Eval development tasks
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    source = args.source_dataset
    manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    if sha256(source / "ARC2.tar") != manifest["archive_sha256"] or sha256(source / "d1_release_config.json") != manifest["config_sha256"]:
        raise ValueError("two-task harness source/config hash mismatch")
    kernel = args.output / "kernel"
    write(kernel / "arc2-d1-two-task-live-check.ipynb", notebook("/kaggle/input/datasets/jimmy5566/arc2-d1-release-source/ARC2.tar", manifest["archive_sha256"], "d1_release_config.json", smoke_task_ids=TASK_IDS))
    write(kernel / "kernel-metadata.json", {"id": "jimmy5566/arc2-d1-two-task-live-check", "title": "ARC2 D1 two-task live check", "code_file": "arc2-d1-two-task-live-check.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False, "dataset_sources": ["jimmy5566/arc2-d1-release-source"], "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"], "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"], "docker_image": IMAGE, "machine_shape": "NvidiaL4"})
    print(json.dumps({"event": "D1_TWO_TASK_LIVE_HARNESS_READY", "task_ids": TASK_IDS, "source_archive_sha256": manifest["archive_sha256"], "hard_budget_seconds": 900}, sort_keys=True))


if __name__ == "__main__": main()
