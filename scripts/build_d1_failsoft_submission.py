#!/usr/bin/env python3
"""CPU-only shared fail-soft D1 finalizer.

The release notebook writes the returned predictions to the official path.
The experiment workbench calls the same function but freezes them only inside
its run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from inference.d1_failsoft_runtime import finalize_failsoft
from inference.d1_release_contract import atomic_json


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("challenge", "release_config", "records", "selection_output", "provenance_output", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() for path in (args.selection_output, args.provenance_output, args.output)):
        raise FileExistsError("refusing to overwrite fail-soft release output")
    selection, predictions, provenance = finalize_failsoft(read(args.challenge), read(args.release_config), read(args.records))
    atomic_json(args.selection_output, selection)
    atomic_json(args.output, predictions)
    provenance.update({
        "selection_sha256": hashlib.sha256(args.selection_output.read_bytes()).hexdigest(),
        "submission_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    })
    atomic_json(args.provenance_output, provenance)
    print(json.dumps({"event": "D1_FAILSOFT_COMPLETE_COVERAGE_PASS", "task_count": provenance["task_count"], "test_output_count": provenance["test_output_count"], "counts": provenance["task_source_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()

