"""Reuse solution-blind records from completed compatible frozen split runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-config", required=True, type=Path)
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--source", action="append", default=[], metavar="TASK_CONFIG:CHECKPOINT")
    args = parser.parse_args()
    full = read_json(args.full_config)
    frozen_hash = hashlib.sha256(args.frozen_config.read_bytes()).hexdigest()
    sources: list[tuple[dict, dict, Path]] = []
    for source in args.source:
        config_name, checkpoint_name = source.split(":", 1)
        config_path, checkpoint_path = Path(config_name), Path(checkpoint_name)
        config, checkpoint = read_json(config_path), read_json(checkpoint_path)
        if not checkpoint.get("complete") or set(checkpoint.get("records", ())) != set(config["task_ids"]):
            raise ValueError(f"source is not fully frozen: {checkpoint_path}")
        known_hash = checkpoint.get("frozen_config_sha256")
        if known_hash is not None and known_hash != frozen_hash:
            raise ValueError(f"source has incompatible frozen config hash: {checkpoint_path}")
        if Path(checkpoint.get("frozen_config", "")).resolve() != args.frozen_config.resolve():
            raise ValueError(f"source references a different frozen config path: {checkpoint_path}")
        sources.append((config, checkpoint, checkpoint_path))
    args.target.parent.mkdir(parents=True, exist_ok=True)
    target = read_json(args.target) if args.target.exists() else {
        "condition": full["config_id"],
        "task_config": str(args.full_config).replace("\\", "/"),
        "frozen_config": str(args.frozen_config).replace("\\", "/"),
        "protocol": "training challenges only; no solution file loaded by this script",
        "records": {}, "runtime_seconds": 0.0, "reused_from_conditions": [],
    }
    if target.get("frozen_config_sha256") not in (None, frozen_hash):
        raise ValueError("target has incompatible frozen config hash")
    target["frozen_config_sha256"] = frozen_hash
    allowed = set(full["task_ids"])
    inherited_runtime = 0.0
    for config, checkpoint, path in sources:
        if not set(config["task_ids"]).issubset(allowed):
            raise ValueError(f"source tasks are not a subset of full config: {path}")
        for task_id, record in checkpoint["records"].items():
            old = target["records"].get(task_id)
            if old is not None and old != record:
                raise ValueError(f"conflicting reused record for {task_id}")
            target["records"][task_id] = record
        provenance = {"condition": config["config_id"], "checkpoint": str(path).replace("\\", "/"), "task_count": len(checkpoint["records"])}
        if provenance not in target["reused_from_conditions"]:
            target["reused_from_conditions"].append(provenance)
            inherited_runtime += float(checkpoint.get("runtime_seconds", 0.0))
    target["runtime_seconds"] += inherited_runtime
    target["task_count"] = len(full["task_ids"])
    target["complete"] = len(target["records"]) == target["task_count"]
    args.target.write_text(json.dumps(target, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"target": str(args.target), "reused_records": len(target["records"]), "remaining": target["task_count"] - len(target["records"]), "complete": target["complete"]}, indent=2))


if __name__ == "__main__":
    main()
