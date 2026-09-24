#!/usr/bin/env python3
"""Stage the thin experiment notebook locally; no Kaggle API is called."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def notebook(source_root: str, config_path: str) -> dict[str, object]:
    lines = [
        "import json, os, subprocess, sys",
        "from pathlib import Path",
        'raw=os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower()',
        'if raw in {"1","true","yes"}: raise RuntimeError("EXPERIMENT_WORKBENCH_REJECTS_COMPETITION_RERUN")',
        'if raw not in {"","0","false","no"}: raise RuntimeError(f"unexpected rerun flag: {raw!r}")',
        f'root=Path({source_root!r}); config=Path({config_path!r})',
        'runner=root/"scripts"/"run_experiment_workbench.py"',
        'if not runner.is_file() or not config.is_file(): raise RuntimeError("explicit workbench source/config missing")',
        'validate=[sys.executable,str(runner),"validate","--config",str(config)]',
        'subprocess.run(validate,check=True)',
        'prepare=[sys.executable,str(runner),"prepare","--config",str(config)]',
        'subprocess.run(prepare,check=True)',
        'resolved=json.loads(config.read_text()); run_dir=Path(resolved["execution"]["artifact_root"])/resolved["experiment"]["run_id"]',
        'subprocess.run([sys.executable,str(runner),"run","--run-dir",str(run_dir)],check=True)',
        'if Path("/kaggle/working/submission.json").exists(): raise RuntimeError("experiment workbench must not create official submission.json")',
    ]
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in lines]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--source-root", required=True); parser.add_argument("--config-path", required=True); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook(args.source_root, args.config_path), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "EXPERIMENT_WORKBENCH_STAGED_LOCALLY", "output": str(args.output), "published": False}, sort_keys=True))


if __name__ == "__main__": main()
