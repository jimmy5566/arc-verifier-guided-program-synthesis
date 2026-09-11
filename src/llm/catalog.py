"""Generate the frozen capability catalogue exposed to LLM providers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from primitives.registry import REGISTRY, PrimitiveSpec


def _entry(spec: PrimitiveSpec) -> dict[str, Any]:
    return {
        "primitive_id": spec.primitive_id,
        "input_types": spec.input_types,
        "output_types": spec.output_types,
        "parameters": spec.parameters,
        "semantics": spec.exact_semantics,
        "preconditions": spec.preconditions,
        "failure_conditions": spec.failure_conditions,
        "does_not_do": spec.does_not_do,
        "complexity_cost": spec.complexity_cost,
    }


def build_capability_catalog(registry: dict[str, PrimitiveSpec] = REGISTRY) -> dict[str, Any]:
    entries = [_entry(registry[primitive_id]) for primitive_id in sorted(registry)]
    return {
        "catalog_version": "llm_capability_catalog_v1",
        "frozen_registry_count": len(entries),
        "canonical_id_only": True,
        "capabilities": entries,
    }


def compact_prompt_catalog(catalog: dict[str, Any], max_entries: int | None = None) -> str:
    entries = catalog["capabilities"][:max_entries] if max_entries is not None else catalog["capabilities"]
    lines = ["Canonical primitive IDs only. Do not invent functions."]
    for entry in entries:
        params = json.dumps(entry["parameters"], sort_keys=True, separators=(",", ":"))
        lines.append(
            f"{entry['primitive_id']} | in={','.join(entry['input_types'])} | out={','.join(entry['output_types'])}"
            f" | params={params} | semantics={entry['semantics']} | fails={entry['failure_conditions']}"
        )
    return "\n".join(lines)


def write_catalog(output_dir: str | Path = "artifacts", registry: dict[str, PrimitiveSpec] = REGISTRY) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    catalog = build_capability_catalog(registry)
    (directory / "llm_capability_catalog_v1.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    (directory / "llm_capability_catalog_v1_prompt.txt").write_text(compact_prompt_catalog(catalog), encoding="utf-8")
    return catalog
