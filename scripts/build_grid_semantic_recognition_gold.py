"""Build private ARCSEMANTICIRV1 gold from separately reviewed annotations.

The annotation input is ignored and must contain no grids; it is intentionally
not shipped to Kaggle.  This builder is the only path that joins task IDs to
semantic gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--annotations", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(annotations, dict) or len(annotations) != 30:
        raise ValueError("reviewed oracle annotations must contain exactly the frozen 30 task IDs")
    sys_path = str(ROOT / "src")
    import sys
    sys.path.insert(0, sys_path)
    from recognition.arc_semantic_ir import canonical, validate
    normalized = {}
    for task_id, value in annotations.items():
        valid, reason = validate(value)
        if not valid: raise ValueError(f"{task_id}: invalid oracle semantic IR: {reason}")
        normalized[task_id] = canonical(value)
    payload = {"schema_id": "ARCSEMANTICIRV1", "task_count": len(normalized), "task_ids_hash": hashlib.sha256(json.dumps(sorted(normalized), separators=(",", ":")).encode()).hexdigest(), "gold": normalized}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"task_count": len(normalized), "task_ids_hash": payload["task_ids_hash"], "contains_grids": False}))


if __name__ == "__main__": main()
