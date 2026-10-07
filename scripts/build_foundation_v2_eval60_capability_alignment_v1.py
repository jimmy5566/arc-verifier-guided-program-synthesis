#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from foundation_v2_eval60_alignment_v1.pipeline import run


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Build CPU-only Base / Foundation-V2 / Eval60 capability matrices.")
    value.add_argument("--base-dir", type=Path, required=True)
    value.add_argument("--foundation-dir", type=Path, required=True)
    value.add_argument("--taxonomy-dir", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    result = run(base_dir=args.base_dir, foundation_dir=args.foundation_dir, taxonomy_dir=args.taxonomy_dir, out=args.output)
    print(json.dumps(result, sort_keys=True))
