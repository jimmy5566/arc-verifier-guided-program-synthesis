"""Freeze audited complete RuleSpec witnesses for downstream-purity execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen purity manifest")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    cases = [
        {"task_id": task_id, "rule_spec": record["matching_complete_rulespec"]}
        for task_id, record in audit["records"].items()
        if record["status"] == "COVERED" and record["matching_complete_rulespec"] is not None
    ]
    manifest = {
        "experiment_id": "ARC2_V3_REPEAT_SEMANTICS_EXPANSION",
        "status": "GOLD_COMPLETE_RULESPEC_MANIFEST_FROZEN",
        "task_ids_hash": audit["task_ids_hash"],
        "protocol": "This manifest freezes only complete RuleSpec witnesses that the frozen30 train-only audit already proved exact. The downstream purity runner reads it without evidence extraction, candidate generation or parameter inference.",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
