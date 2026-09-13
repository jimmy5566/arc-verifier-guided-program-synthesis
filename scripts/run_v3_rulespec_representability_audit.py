"""Audit frozen30 semantic *expressibility*, not deterministic solving.

The input is a private, train-only forensic classification.  This script maps
generic semantic requirements to the public Capability Library; it never opens
ARC grids, test outputs, solutions, or invokes candidate generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from v3.schema.capability_library import SemanticTag, supported_semantic_tags


_REQUIREMENTS: dict[str, frozenset[SemanticTag | str]] = {
    "COVERED": frozenset(),
    "MOTIF_EXTRACTION_OR_TRANSFORM_UNRESOLVED": frozenset({SemanticTag.MOTIF, SemanticTag.TRANSFORM}),
    "PROGRESSIVE_SPACING_OR_STATE_UPDATE": frozenset({SemanticTag.PROGRESSIVE_SPACING, SemanticTag.STATE_UPDATE}),
    "COLOR_OR_SHAPE_SEQUENCE": frozenset({SemanticTag.SEQUENCE}),
    "COMPOSITION_STATE_UPDATE_SEMANTICS": frozenset({SemanticTag.COMPOSITION, SemanticTag.STATE_UPDATE}),
    "CONDITIONAL_ROLE_SEMANTICS_INSUFFICIENT": frozenset({SemanticTag.CONDITIONAL}),
    "REFERENCE_ROLE_CROP_RELATION_INSUFFICIENT": frozenset({SemanticTag.REFERENCE_ROLE_CROP}),
    "DIRECTION_AMBIGUITY": frozenset({SemanticTag.DERIVED_DIRECTION}),
    # This label intentionally remains unsupported.  It asks for a genuinely
    # absent geometric-expansion rule, not a task-specific patch.
    "MISSING_GENERIC_OPERATION_SEMANTICS": frozenset({"GEOMETRIC_EXPANSION"}),
}


def _requirements(category: str) -> frozenset[SemanticTag | str]:
    try:
        return _REQUIREMENTS[category]
    except KeyError as exc:
        raise ValueError(f"unknown frozen semantic category: {category}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forensics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite representability audit")
    forensic = json.loads(args.forensics.read_text(encoding="utf-8"))
    records = forensic.get("records")
    if not isinstance(records, dict) or len(records) != 30:
        raise ValueError("requires frozen30 construction forensics")
    available = supported_semantic_tags()
    audit_records: dict[str, dict[str, object]] = {}
    missing_taxonomy: Counter[str] = Counter()
    for task_id, record in records.items():
        if not isinstance(record, dict) or not isinstance(record.get("semantic_category"), str):
            raise ValueError("invalid frozen forensic record")
        category = record["semantic_category"]
        required = _requirements(category)
        missing = sorted(item.value if isinstance(item, SemanticTag) else item for item in required if item not in available)
        audit_records[task_id] = {
            "semantic_category": category,
            "required_capabilities": sorted(item.value if isinstance(item, SemanticTag) else item for item in required),
            "representable": not missing,
            "missing_capabilities": missing,
        }
        missing_taxonomy.update(missing)
    represented = sum(bool(item["representable"]) for item in audit_records.values())
    artifact = {
        "experiment_id": "ARC2_V3_CAPABILITY_AND_UPSTREAM_PUSH",
        "status": "RULESPEC_REPRESENTABILITY_AUDIT_COMPLETE",
        "task_ids_hash": forensic.get("task_ids_hash"),
        "protocol": "Maps private train-only forensic semantic categories to generic public Capability Library tags. It performs no parameter inference, deterministic candidate search, model call, test access, solution access, Kaggle, or E2E execution.",
        "capability_library_tags": sorted(item.value for item in available),
        "representability": {"represented": represented, "total": len(audit_records)},
        "missing_capability_taxonomy": dict(sorted(missing_taxonomy.items())),
        "records": audit_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"representability": artifact["representability"], "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
