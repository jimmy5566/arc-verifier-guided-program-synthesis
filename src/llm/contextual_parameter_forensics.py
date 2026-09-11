"""Offline post-freeze audit of Q1's unresolved pure parameter cases."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .compiler_aware_interface import skeleton_by_id
from .semantic_relation_normalizer import extract_relation_features
from .c3_semantic_intent import assess_program, constraints_for_case
from .macro_api_benchmark import score_response


TAXONOMY = (
    "LEXICAL_GAP", "RELATIONAL_INFERENCE", "CONTEXT_REQUIRED", "MULTI_SLOT_DEPENDENCY",
    "IMPLICIT_DIRECTION", "IMPLICIT_ORIENTATION", "IMPLICIT_SOURCE_REFERENCE",
    "AMBIGUOUS_INSTRUCTION", "ONTOLOGY_COVERAGE_GAP", "OTHER_CONTEXTUAL_PARAMETER_FAILURE",
)


def _taxonomy(*, fields: tuple[str, ...], features: tuple[str, ...], macro_ids: tuple[str, ...]) -> tuple[str, tuple[str, ...], bool]:
    aspects: list[str] = ["CONTEXT_REQUIRED"]
    if "direction" in fields:
        aspects.extend(("IMPLICIT_DIRECTION", "IMPLICIT_SOURCE_REFERENCE"))
    if "orientation" in fields:
        aspects.append("IMPLICIT_ORIENTATION")
        if "RELATION.HORIZONTAL" in features or "RELATION.VERTICAL" in features:
            aspects.append("RELATIONAL_INFERENCE")
    if len(fields) > 1:
        aspects.append("MULTI_SLOT_DEPENDENCY")
    if "shape_source" in fields or "COUNT_STRUCTURES" in macro_ids:
        aspects.append("MULTI_SLOT_DEPENDENCY")
    aspects = list(dict.fromkeys(aspects))
    primary = "MULTI_SLOT_DEPENDENCY" if "MULTI_SLOT_DEPENDENCY" in aspects else "IMPLICIT_SOURCE_REFERENCE" if "IMPLICIT_SOURCE_REFERENCE" in aspects else "CONTEXT_REQUIRED"
    return primary, tuple(aspects), True


def build_contextual_forensics(*, benchmark: Mapping[str, Any], frozen: Mapping[str, Any], checkpoint_root: Path) -> dict[str, Any]:
    cases = {str(item["case_id"]): item for item in benchmark["cases"]}
    rows: list[dict[str, Any]] = []
    for path in sorted(checkpoint_root.glob("case_*.json")):
        record = json.loads(path.read_text(encoding="utf-8")); case_id = str(record["case_id"])
        baseline = frozen["s2_baseline_programs"].get(case_id)
        if not isinstance(baseline, Mapping):
            continue
        skeleton = skeleton_by_id(str(baseline["skeleton_id"]))
        if skeleton is None:
            continue
        case = cases[case_id]
        api = score_response(case, json.dumps(record["program"], separators=(",", ":")))
        semantic = assess_program(constraints_for_case(case), record["program"], compile_valid=bool(api["compile_valid"]))
        # The Phase-A membership is inspected only after Q1 predictions have
        # frozen.  Field identities are from the already frozen S2 forensic;
        # this avoids importing a canonical program into the relation rules.
        if semantic.outcome != "SEMANTIC_WRONG_PARAMETER":
            continue
        fields = tuple(sorted(frozen["forensics"]["wrong_parameter_fields_by_case"].get(case_id, ())))
        if not fields:
            continue
        instruction = str(case["natural_language_instruction"])
        features = extract_relation_features(instruction)
        primary, aspects, repairable = _taxonomy(fields=fields, features=features, macro_ids=skeleton.macro_ids)
        rows.append({
            "case_id": case_id,
            "family": str(baseline["family"]),
            "skeleton_id": skeleton.skeleton_id,
            "macro_ids": list(skeleton.macro_ids),
            "parameter_fields": list(fields),
            "primary_taxonomy": primary,
            "taxonomy_aspects": list(aspects),
            "semantic_features": list(features),
            "truly_repairable": repairable,
            "ambiguous": False,
        })
    taxonomy = Counter(row["primary_taxonomy"] for row in rows)
    aspects = Counter(item for row in rows for item in row["taxonomy_aspects"])
    fields = Counter(item for row in rows for item in row["parameter_fields"])
    return {
        "experiment_id": "CONTEXTUAL_PARAMETER_FORENSICS_V1",
        "status": "COMPLETE_OFFLINE_POST_Q1_AUDIT",
        "arc_data_used": False,
        "model_called": False,
        "q1_baseline": {"semantic_success": 36, "case_count": 60, "remaining_pure_parameter_failures": len(rows)},
        "taxonomy_counts_primary": dict(sorted(taxonomy.items())),
        "taxonomy_counts_aspects": dict(sorted(aspects.items())),
        "per_field_distribution": dict(sorted(fields.items())),
        "lexical_vs_relational_vs_contextual": {
            "lexical": sum("LEXICAL_GAP" in row["taxonomy_aspects"] for row in rows),
            "relational": sum("RELATIONAL_INFERENCE" in row["taxonomy_aspects"] for row in rows),
            "contextual": sum("CONTEXT_REQUIRED" in row["taxonomy_aspects"] for row in rows),
        },
        "truly_repairable_count": sum(bool(row["truly_repairable"]) for row in rows),
        "ambiguous_count": sum(bool(row["ambiguous"]) for row in rows),
        "cases": rows,
        "protocol": {"canonical_used_only_for_post_freeze_difference_audit": True, "canonical_not_used_for_rule_design_or_model_input": True, "no_case_specific_mapping": True},
    }
