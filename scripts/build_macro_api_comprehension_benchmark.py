"""Freeze the deterministic Macro API comprehension benchmark and inventory."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from llm.macro_api_benchmark import benchmark_definition, registry_inventory


ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    parser.add_argument("--inventory", type=Path, default=ROOT / "experiments" / "results" / "MACRO_API_INVENTORY.json")
    parser.add_argument("--definition", type=Path, default=ROOT / "experiments" / "results" / "MACRO_API_COMPREHENSION_BENCHMARK_DEFINITION.json")
    args = parser.parse_args()
    definition = benchmark_definition() | {"creation_timestamp_utc": datetime.now(timezone.utc).isoformat()}
    write(args.config, definition)
    write(args.inventory, registry_inventory())
    write(args.definition, {key: value for key, value in definition.items() if key != "cases"})
    print(json.dumps({"status": "FROZEN", "benchmark_hash": definition["benchmark_hash"], "case_count": definition["case_count"]}))


if __name__ == "__main__":
    main()
