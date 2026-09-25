#!/usr/bin/env python3
"""Materialize the 3090 execution-only view of a frozen D1 release config."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from inference.d1_release_contract import atomic_json
from scripts.run_d1_release_4gpu import validate_live_config
from scripts.run_eval60_d1_ampere import validate_ampere_release_config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(*, frozen_config: Path, output: Path, ptxas_path: Path, environment_id: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime config: {output}")
    if not ptxas_path.is_file():
        raise FileNotFoundError(f"ptxas is unavailable: {ptxas_path}")
    source = json.loads(frozen_config.read_text(encoding="utf-8"))
    # This proves the input is the historical frozen release before changing
    # its execution-only PTXAS path.
    validate_live_config(source)
    runtime = json.loads(json.dumps(source))
    runtime["environment"]["ptxas_path"] = str(ptxas_path)
    runtime.setdefault("execution_environment", {})
    runtime["execution_environment"].update(
        {
            "environment_id": environment_id,
            "expected_gpu_name": "RTX 3090",
            "expected_compute_capability": [8, 6],
            "source_frozen_config_sha256": sha256(frozen_config),
            "scope": "execution_environment_only",
        }
    )
    validate_ampere_release_config(runtime)
    atomic_json(output, runtime)
    return {
        "event": "D1_AMPERE_RUNTIME_CONFIG_MATERIALIZED",
        "source_frozen_config_sha256": sha256(frozen_config),
        "output_config_sha256": sha256(output),
        "ptxas_path": str(ptxas_path),
        "environment_id": environment_id,
        "scientific_fields_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-release-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ptxas-path", type=Path, default=Path("/usr/local/cuda/bin/ptxas"))
    parser.add_argument("--environment-id", default="3090-ampere-env-v1")
    args = parser.parse_args()
    print(json.dumps(materialize(frozen_config=args.frozen_release_config, output=args.output, ptxas_path=args.ptxas_path, environment_id=args.environment_id), sort_keys=True))


if __name__ == "__main__":
    main()
