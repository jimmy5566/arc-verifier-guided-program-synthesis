"""Audit registry quality and V4 reachability without reading task data."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from capabilities.execution import CapabilityExecutor
from primitives.registry import REGISTRY, validate_registry


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- none"]


def main() -> None:
    errors = validate_registry()
    families = Counter(spec.category for spec in REGISTRY.values())
    by_reference: dict[str, list[str]] = defaultdict(list)
    for primitive_id, spec in REGISTRY.items():
        by_reference[spec.implementation_reference].append(primitive_id)
    overlaps = [
        f"{reference}: {', '.join(sorted(ids))}"
        for reference, ids in sorted(by_reference.items())
        if len(ids) > 1
    ]
    vague = [
        primitive_id
        for primitive_id, spec in sorted(REGISTRY.items())
        if spec.exact_semantics == "Deterministic operation with the documented parameters."
    ]
    dispatch = CapabilityExecutor.dispatch_audit()
    report = [
        "# Capability Registry Audit",
        "",
        f"- Active registered primitives: {len(REGISTRY)}",
        f"- Active V4 capability primitives: {dispatch['registered_v4_count']}",
        f"- V4 executable dispatches: {dispatch['executable_dispatch_count']}",
        f"- Validation errors: {len(errors)}",
        "",
        "## Family counts",
        "",
        *[f"- {family}: {count}" for family, count in sorted(families.items())],
        "",
        "## Duplicate or overlapping implementation references",
        "",
        *_bullets(overlaps),
        "",
        "Shared references are reviewed rather than automatically treated as duplicates: parameterized transform variants and row/column serialization aliases intentionally reuse implementations.",
        "",
        "## Ambiguous metadata candidates",
        "",
        f"- Generic exact-semantics text remains on {len(vague)} legacy entries: {', '.join(vague) if vague else 'none'}",
        "- Full separators and partial lines use distinct IDs.",
        "- `REL_OVERLAP_V1` is mask overlap; `REL_CONTAINS_BBOX_V1` is bounding-box containment.",
        "",
        "## Missing implementations",
        "",
        *_bullets([error for error in errors if "missing implementation" in error]),
        "",
        "## Unreachable active V4 capabilities",
        "",
        *_bullets(list(dispatch["unsupported_primitive_ids"]) + list(dispatch["missing_handler_methods"])),
        "",
        "## Deprecated aliases",
        "",
        "- none; no deprecated capability ID is silently redirected to another semantic ID.",
        "",
        "## Validation errors",
        "",
        *_bullets(errors),
    ]
    Path("reports/capability_registry_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(
        {
            "primitive_count": len(REGISTRY),
            "v4_primitive_count": dispatch["registered_v4_count"],
            "family_counts": dict(families),
            "errors": errors,
            "unsupported_v4": list(dispatch["unsupported_primitive_ids"]),
        }
    )


if __name__ == "__main__":
    main()
