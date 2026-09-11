"""Deterministic, no-inference forensics for frozen S2 parameter failures."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .compiler_aware_interface import skeleton_by_id


def _steps(program: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    hypotheses = program.get("hypotheses") if isinstance(program, Mapping) else None
    if not isinstance(hypotheses, list) or len(hypotheses) != 1 or not isinstance(hypotheses[0], Mapping):
        raise ValueError("expected exactly one macro hypothesis")
    steps = hypotheses[0].get("steps")
    if not isinstance(steps, list):
        raise ValueError("expected macro steps")
    return steps


def _subtype(parameter: str, expected: object, actual: object) -> str:
    if actual is None:
        return "MISSING_SEMANTIC_SLOT"
    if expected is None:
        return "EXTRA_SEMANTIC_SLOT"
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            return "WRONG_SYMBOLIC_SOURCE"
        expected_value = next(iter(expected.values())); actual_value = next(iter(actual.values()))
        if expected_value != actual_value:
            return {
                "axis": "WRONG_AXIS", "direction": "WRONG_DIRECTION", "orientation": "WRONG_ORIENTATION",
                "selector": "WRONG_SELECTOR", "color_source": "WRONG_COLOR_SEMANTICS",
                "count_kind": "WRONG_COUNT_SEMANTICS", "translation": "WRONG_TRANSLATION_VALUE",
            }.get(parameter, "WRONG_ENUM_VALUE")
    return "OTHER_PARAMETER_SEMANTIC_ERROR"


def analyze_s2_parameter_failures(*, benchmark: Mapping[str, Any], checkpoint_root: Path) -> dict[str, Any]:
    cases = {str(case["case_id"]): case for case in benchmark["cases"]}
    rows: list[dict[str, Any]] = []
    files = sorted(checkpoint_root.glob("case_*.json"))
    if len(files) != 60 or len(cases) != 60:
        raise ValueError("requires all 60 frozen S2 checkpoints and cases")
    fields: Counter[str] = Counter(); subtypes: Counter[str] = Counter(); single = multi = pure = structural = static_slot_confounds = 0
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("semantic", {}).get("outcome") != "SEMANTIC_WRONG_PARAMETER":
            continue
        case_id = str(record["case_id"]); case = cases.get(case_id)
        if case is None:
            raise ValueError(f"checkpoint case not in benchmark: {case_id}")
        actual = _steps(json.loads(str(record["materialized_program"])))
        expected = _steps(case["canonical_program"])
        raw_structured = json.loads(str(record["structured_raw_response"]))
        skeleton = skeleton_by_id(str(raw_structured["skeleton_id"]))
        if skeleton is None:
            raise ValueError(f"unknown S2 skeleton for {case_id}")
        actual_ids = tuple(str(step["macro_id"]) for step in actual)
        expected_ids = tuple(str(step["macro_id"]) for step in expected)
        family_correct = bool(record.get("selection", {}).get("family_semantic_correct"))
        skeleton_compatible = actual_ids == expected_ids == skeleton.macro_ids
        differences: list[tuple[str, str]] = []
        for index, (wanted, got) in enumerate(zip(expected, actual)):
            expected_params = wanted.get("params", {}); actual_params = got.get("params", {})
            for parameter in sorted(set(expected_params) | set(actual_params)):
                if expected_params.get(parameter) != actual_params.get(parameter):
                    subtype = _subtype(parameter, expected_params.get(parameter), actual_params.get(parameter))
                    differences.append((parameter, subtype)); fields[parameter] += 1; subtypes[subtype] += 1
        only_parameter = family_correct and skeleton_compatible and bool(differences)
        if only_parameter:
            pure += 1
        else:
            structural += 1
        # C3's static literal slots are intentionally narrower than the full
        # frozen Macro API.  This is reported as an interface confound, not a
        # skeleton-selection confound, because Macro IDs/order are compatible.
        if any(item[1] in {"MISSING_SEMANTIC_SLOT", "WRONG_SYMBOLIC_SOURCE"} for item in differences):
            static_slot_confounds += 1
        if len(differences) == 1:
            single += 1
        else:
            multi += 1
        rows.append({
            "case_id": case_id, "family_correct": family_correct, "skeleton_compatible": skeleton_compatible,
            "only_parameter_wrong": only_parameter, "parameter_fields": sorted({item[0] for item in differences}),
            "error_subtypes": sorted({item[1] for item in differences}), "parameter_difference_count": len(differences),
        })
    if len(rows) != 17:
        raise ValueError(f"expected 17 frozen S2 wrong-parameter cases, found {len(rows)}")
    return {
        "experiment_id": "PARAMETER_GROUNDING_FORENSICS_V1", "status": "COMPLETE_NO_NEW_INFERENCE",
        "protocol": {"new_model_generation": False, "arc_data_used": False, "arc_solutions_used": False, "semantic_labels_used_for_repair": False, "canonical_sent_to_model": False},
        "s2_wrong_parameter_cases": len(rows), "pure_parameter_failure_count": pure,
        "non_parameter_structural_failure_count": structural, "c3_static_slot_interface_confounds": static_slot_confounds,
        "single_parameter_error_cases": single, "multiple_parameter_error_cases": multi,
        "top_parameter_fields": dict(sorted(fields.items())), "top_error_subtypes": dict(sorted(subtypes.items())),
        "case_forensics": sorted(rows, key=lambda item: item["case_id"]),
    }


def safe_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Parameter Grounding Forensics V1", "",
        "- 仅分析冻结 S2 checkpoint；无新模型调用、ARC 数据、ARC solution 或 LLM judge。",
        f"- S2 wrong-parameter: {result['s2_wrong_parameter_cases']}；pure parameter failures: {result['pure_parameter_failure_count']}；structural confounds: {result['non_parameter_structural_failure_count']}。",
        f"- C3 static-slot interface confounds: {result['c3_static_slot_interface_confounds']}；single/multiple parameter errors: {result['single_parameter_error_cases']}/{result['multiple_parameter_error_cases']}。",
        "", "## Parameter fields", "",
        *[f"- {field}: {count}" for field, count in result["top_parameter_fields"].items()],
        "", "## Error subtypes", "",
        *[f"- {name}: {count}" for name, count in result["top_error_subtypes"].items()], "",
        "## Interpretation", "",
        "- 这里的 structural confound 仅指 family 或 Macro-id skeleton 不兼容；C3 static-slot interface mismatch 单独报告，不能被误写为 skeleton 选择错误。",
        "",
    ]
    return "\n".join(lines)
