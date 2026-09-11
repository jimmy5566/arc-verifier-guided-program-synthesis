"""Machine-readable, static compiler reachability for the frozen Macro DSL.

This module is deliberately an *audit* of :class:`MacroProgramCompilerV1`.
It does not change the registry, macro semantics, type checker, compiler, or
parameter solver.  The inventory uses a no-training-data fixture and direct
literals so that "compiler reachable" has a precise meaning for the API-only
benchmark: the current compiler can materialize a primitive program without
ARC grids or solutions.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from .macro_compiler_v1 import MacroProgramCompilerV1
from .macro_dsl import MACROS, MacroHypothesis, MacroStatus, MacroStep


# These are compiler-safe witnesses, not benchmark answers.  Every value is a
# direct literal because this audit intentionally has no training examples from
# which symbolic sources could be resolved.
_STATIC_ARGUMENT_WITNESSES: dict[str, dict[str, Any]] = {
    "SELECT_OBJECT": {"selector": {"literal": "largest"}},
    "TRANSFORM_OBJECT": {"operation": {"literal": "crop"}},
    "COPY_OBJECT": {"translation": {"literal": [0, 0]}},
    "ALIGN_OBJECTS": {"alignment": {"literal": "ALIGN_LEFT"}},
    "FILL_ENCLOSED_REGION": {"color_source": {"literal": 1}},
    "EXTRACT_REGION": {"position": {"literal": [0, 0]}},
    "COMPLETE_PATTERN": {"period": {"literal": 1}, "output_shape": {"literal": [1, 1]}},
    "REPAIR_PATTERN": {"period": {"literal": 1}},
    "COMPLETE_SYMMETRY": {"axis": {"literal": "HORIZONTAL"}},
    "TRACE_PATH": {"direction": {"literal": "PATH_ENDPOINT_ORDER"}},
    "SERIALIZE_PATH": {"orientation": {"literal": "HORIZONTAL"}},
    "COUNT_STRUCTURES": {"count_kind": {"literal": "OBJECT_COUNT"}},
    "GENERATE_FROM_COUNT": {
        "shape_source": {"literal": [1, 1]},
        "color_source": {"literal": 1},
        "orientation": {"literal": "HORIZONTAL"},
    },
    "EXTRACT_SEQUENCE": {"orientation": {"literal": "HORIZONTAL"}, "position": {"literal": 0}},
    "TRANSFORM_SEQUENCE": {"operation": {"literal": "reverse"}},
    "BUILD_GRID_FROM_SEQUENCE": {"orientation": {"literal": "HORIZONTAL"}},
    "SPLIT_BY_SEPARATOR": {"orientation": {"literal": "HORIZONTAL"}, "position": {"literal": 0}},
    "CONNECT_STRUCTURES": {"position": {"literal": [0, 0]}, "color_source": {"literal": 1}},
    "REPEAT_UNTIL_BOUNDARY": {
        "direction": {"literal": "HORIZONTAL"},
        "color_source": {"literal": 1},
        "position": {"literal": [0, 0]},
    },
    "PROPAGATE_PATTERN": {"translation": {"literal": [0, 0]}},
}


@dataclass(frozen=True)
class StaticCompileOutcome:
    macro_ids: tuple[str, ...]
    compile_valid: bool
    status: str
    reason: str
    primitive_ids: tuple[str, ...]


def static_params(macro_id: str) -> dict[str, Any]:
    """Return a fresh no-data witness for one registered macro."""
    return {name: dict(value) for name, value in _STATIC_ARGUMENT_WITNESSES.get(macro_id, {}).items()}


def hypothesis_for_chain(macro_ids: Iterable[str]) -> MacroHypothesis:
    ids = tuple(macro_ids)
    return MacroHypothesis(
        hypothesis_id="compiler_reachability_witness",
        steps=tuple(MacroStep(macro_id, static_params(macro_id)) for macro_id in ids),
    )


def compile_chain(macro_ids: Iterable[str]) -> StaticCompileOutcome:
    """Compile one type-chain with no ARC train pairs and static witnesses."""
    ids = tuple(macro_ids)
    result = MacroProgramCompilerV1(allow_direct_literals=True).compile(
        hypothesis_for_chain(ids), SimpleNamespace(train=()),
    )
    primitives = () if result.program is None else tuple(step.primitive_id for step in result.program.steps)
    return StaticCompileOutcome(ids, result.status == MacroStatus.COMPILED and result.program is not None, result.status.value, result.reason, primitives)


def type_valid_chains(depth: int) -> tuple[tuple[str, ...], ...]:
    """Enumerate all registry-declared Grid -> Grid chains of exact depth."""
    if depth < 1:
        raise ValueError("depth must be positive")
    chains: list[tuple[str, ...]] = []

    def visit(concept: str, prefix: tuple[str, ...]) -> None:
        if len(prefix) == depth:
            if concept == "Grid":
                chains.append(prefix)
            return
        for macro_id in sorted(MACROS):
            macro = MACROS[macro_id]
            if macro["input_concept"] == concept:
                visit(str(macro["output_concept"]), prefix + (macro_id,))

    visit("Grid", ())
    return tuple(chains)


def compiler_valid_chains(depth: int) -> tuple[StaticCompileOutcome, ...]:
    return tuple(outcome for chain in type_valid_chains(depth) if (outcome := compile_chain(chain)).compile_valid)


def _argument_forms(macro_id: str, supported: bool) -> dict[str, list[dict[str, Any]]]:
    macro = MACROS[macro_id]
    if not supported:
        return {name: [] for name in sorted(macro["allowed_parameter_sources"])}
    params = static_params(macro_id)
    return {
        name: [{"wrapper": "literal", "static_witness": params[name]["literal"]}]
        for name in sorted(params)
    }


def reachability_inventory(max_depth: int = 3) -> dict[str, Any]:
    """Produce serializable support records and exact static-chain coverage."""
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    all_outcomes = {
        depth: tuple(compile_chain(chain) for chain in type_valid_chains(depth))
        for depth in range(1, max_depth + 1)
    }
    reachable = {
        macro_id
        for outcomes in all_outcomes.values()
        for outcome in outcomes
        if outcome.compile_valid
        for macro_id in outcome.macro_ids
    }
    predecessor_macros: dict[str, set[str]] = defaultdict(set)
    successor_macros: dict[str, set[str]] = defaultdict(set)
    expansions: dict[str, set[str]] = defaultdict(set)
    for outcomes in all_outcomes.values():
        for outcome in outcomes:
            if not outcome.compile_valid:
                continue
            for index, macro_id in enumerate(outcome.macro_ids):
                if index:
                    predecessor_macros[macro_id].add(outcome.macro_ids[index - 1])
                if index + 1 < len(outcome.macro_ids):
                    successor_macros[macro_id].add(outcome.macro_ids[index + 1])
                expansions[macro_id].update(outcome.primitive_ids)
    macro_records = []
    for macro_id in sorted(MACROS):
        macro = MACROS[macro_id]
        supported = macro_id in reachable
        macro_records.append({
            "macro_id": macro_id,
            "type_signature": f"{macro['input_concept']} -> {macro['output_concept']}",
            "type_declared": True,
            "compiler_supported": supported,
            "supported_argument_forms": _argument_forms(macro_id, supported),
            "supported_symbolic_wrappers": [],
            "supported_predecessor_types": [macro["input_concept"]] if supported else [],
            "supported_successor_types": [macro["output_concept"]] if supported else [],
            "compiler_reachable_predecessor_macros": sorted(predecessor_macros[macro_id]),
            "compiler_reachable_successor_macros": sorted(successor_macros[macro_id]),
            "valid_low_level_expansions": sorted(expansions[macro_id]),
            "compiler_preconditions": list(macro["preconditions"]),
            "static_audit_scope": "direct literal witnesses with zero ARC train pairs",
        })
    chain_inventory: dict[str, Any] = {}
    for depth, outcomes in all_outcomes.items():
        valid = [outcome for outcome in outcomes if outcome.compile_valid]
        chain_inventory[str(depth)] = {
            "type_valid_chain_count": len(outcomes),
            "compile_valid_chain_count": len(valid),
            "compiler_coverage_rate": len(valid) / len(outcomes) if outcomes else 0.0,
            "compile_valid_chains": [
                {"macro_ids": list(outcome.macro_ids), "low_level_primitives": list(outcome.primitive_ids)}
                for outcome in valid
            ],
        }
    return {
        "inventory_id": "COMPILER_REACHABILITY_INVENTORY_V1",
        "scope": "Static no-data compiler reachability; not an ARC execution inventory.",
        "registry_macro_count": len(MACROS),
        "type_declared_macro_count": len(MACROS),
        "compiler_supported_macro_count": len(reachable),
        "compiler_macro_coverage_rate": len(reachable) / len(MACROS),
        "macros": macro_records,
        "chain_inventory": chain_inventory,
    }
