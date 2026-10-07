#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from base_eval60_capability_alignment_v1.pipeline import run


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Build the CPU-only Base × Eval60 capability alignment.")
    value.add_argument("--taxonomy", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--freeze-path", type=Path)
    value.add_argument("--input-root", type=Path)
    value.add_argument("--base-predictions", type=Path)
    value.add_argument("--oracle-csv", type=Path)
    value.add_argument("--observed-base-count", type=int)
    value.add_argument("--observed-foundation-count", type=int)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    result = run(
        taxonomy=args.taxonomy,
        out=args.output,
        freeze_path=args.freeze_path,
        input_root=args.input_root,
        base_predictions=args.base_predictions,
        oracle_csv=args.oracle_csv,
        observed_base_count=args.observed_base_count,
        observed_foundation_count=args.observed_foundation_count,
    )
    print(json.dumps(result, sort_keys=True))
