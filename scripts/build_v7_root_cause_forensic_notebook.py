"""Build the isolated Kaggle V7 forensic notebook without altering V7 source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cell(harness: str) -> str:
    lines = [
        "import json", "import shutil", "import subprocess", "import sys", "import zipfile", "from pathlib import Path", "",
        'INPUT = Path("/kaggle/input")', 'WORKING = Path("/kaggle/working")', 'SOURCE = WORKING / "v7_exact_source" / "ARC2"',
        'archive = next(INPUT.rglob("ARC2.zip"), None)', 'source_script = next(INPUT.rglob("run_qwen4b_native_augmentation_search.py"), None)',
        'if archive is not None:', '    with zipfile.ZipFile(archive) as value: value.extractall(SOURCE.parent)',
        'elif source_script is not None:', '    shutil.copytree(source_script.parents[1], SOURCE)',
        'else: raise RuntimeError("Missing exact V38/V7 hardened source input")', "",
        "def one(pattern):", "    matches = sorted(INPUT.rglob(pattern))", "    if len(matches) != 1: raise RuntimeError(f'expected exactly one {pattern}, found {matches}')", "    return matches[0]", "",
        'challenge = one("arc-agi_training_challenges.json")', 'solutions = one("arc-agi_training_solutions.json")',
        'golden_candidates = one("NEW_candidates_frozen.json")', 'golden_selection = one("NEW_B_selection_frozen.json")',
        'v7_notebook = one("arc2-arc-prize-2026-dynamic-b-production.ipynb")',
        'model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")',
        'native_config = SOURCE / "configs" / "nvarc_native_846d0198"', 'config = SOURCE / "configs" / "QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"',
        'for path in (SOURCE, challenge, solutions, golden_candidates, golden_selection, v7_notebook, model, native_config, config):',
        "    if not path.exists(): raise RuntimeError(f'required forensic input missing: {path}')", "",
        'harness = WORKING / "run_v7_root_cause_forensic.py"', f"harness.write_text({harness!r}, encoding='utf-8')",
        'output = WORKING / "artifacts" / "v7_root_cause_forensic"',
        "command = [sys.executable, str(harness), '--production-root', str(SOURCE), '--challenge-path', str(challenge), '--solutions-path', str(solutions), '--golden-candidates', str(golden_candidates), '--golden-selection', str(golden_selection), '--model-path', str(model), '--native-config-dir', str(native_config), '--config', str(config), '--notebook-path', str(v7_notebook), '--output-dir', str(output)]",
        "print(json.dumps({'event': 'V7_FORENSIC_START', 'command': command, 'source': str(SOURCE), 'golden': str(golden_candidates), 'internet': False}, sort_keys=True), flush=True)",
        "completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)",
        "(WORKING / 'v7_forensic.log').write_text(completed.stdout, encoding='utf-8')", "print(completed.stdout, flush=True)",
        "if completed.returncode: raise RuntimeError(f'V7 forensic failed with code {completed.returncode}; see /kaggle/working/v7_forensic.log')",
        "print(json.dumps({'event': 'V7_FORENSIC_COMPLETE', 'report': str(output / 'ROOT_CAUSE_REPORT.json')}, sort_keys=True), flush=True)",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    harness = args.harness.read_text(encoding="utf-8")
    cell = _cell(harness)
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
