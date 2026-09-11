from __future__ import annotations

import json

from llm.compiler_failure_forensics import classify_compiler_result, forensics_result
from llm.compiler_reachability import compile_chain, reachability_inventory, type_valid_chains


def _program(*steps: tuple[str, dict[str, object]]) -> str:
    return json.dumps({"hypotheses": [{"hypothesis_id": "synthetic", "steps": [{"macro_id": macro_id, "params": params} for macro_id, params in steps]}]})


def test_static_reachability_measures_compiler_not_only_declared_types() -> None:
    inventory = reachability_inventory()
    assert [len(type_valid_chains(depth)) for depth in (1, 2, 3)] == [6, 40, 268]
    assert [inventory["chain_inventory"][str(depth)]["compile_valid_chain_count"] for depth in (1, 2, 3)] == [2, 6, 19]
    assert inventory["compiler_supported_macro_count"] == 13
    assert compile_chain(("FIND_OBJECTS", "SELECT_OBJECT", "TRANSFORM_OBJECT")).compile_valid
    assert not compile_chain(("FIND_OBJECTS", "ALIGN_OBJECTS")).compile_valid


def test_compiler_taxonomy_keeps_parameter_and_capability_causes_distinct() -> None:
    assert classify_compiler_result("PARAMETER_INVALID", "symbolic parameter cannot be resolved") == ("UNSUPPORTED_PARAMETER_FORM", "UNRESOLVED_SYMBOLIC_SOURCE")
    assert classify_compiler_result("COMPILER_INVALID", "period compiler requires an unambiguous period resolver") == ("LOW_LEVEL_PRIMITIVE_GAP", "PERIOD_RESOLVER_MISSING")
    assert classify_compiler_result("COMPILER_INVALID", "compiler mapping for X requires a future deterministic evidence rule") == ("UNSUPPORTED_MACRO_EXPANSION", "NO_IMPLEMENTED_COMPILER_BRANCH")


def test_forensics_replays_type_valid_compile_invalid_without_exposing_raw(tmp_path) -> None:
    raw = _program(
        ("COUNT_STRUCTURES", {"count_kind": {"literal": "OBJECT_COUNT"}}),
        ("GENERATE_FROM_COUNT", {"shape_source": {"source": "SAME_AS_INPUT"}, "color_source": {"source": "FROM_UNIQUE_COLOR"}, "orientation": {"source": "HORIZONTAL"}}),
    )
    for index in range(60):
        record = {
            "case_id": f"synthetic_{index:02d}",
            "category": "TWO_STEP_TYPED_CHAIN",
            "raw_response": raw,
            "score": {
                "type_valid": True,
                "compile_valid": False,
                "failure_message": "symbolic parameter cannot be resolved",
            },
        }
        (tmp_path / f"case_synthetic_{index:02d}.json").write_text(json.dumps(record), encoding="utf-8")
    result = forensics_result(tmp_path)
    assert result["r2_counts"]["type_valid_compile_invalid"] == 60
    assert result["failure_taxonomy"] == {"UNSUPPORTED_PARAMETER_FORM": 60}
    assert result["audit"]["compiler_rejections_correct"]
    assert "raw_response" not in json.dumps(result)
