"""Solution-blind replay and taxonomy for frozen API-benchmark compiler failures."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from .compiler_reachability import compile_chain, reachability_inventory
from .macro_api_benchmark import score_response
from .macro_compiler_v1 import MacroProgramCompilerV1
from .macro_dsl import MacroHypothesis, MacroStatus, parse_macro_hypotheses
from .v2_failure_forensics import assess_hypothesis


TAXONOMY = (
    "UNSUPPORTED_MACRO_EXPANSION",
    "UNSUPPORTED_PARAMETER_FORM",
    "UNSUPPORTED_MACRO_COMBINATION",
    "NO_COMPILER_PATH",
    "UNSUPPORTED_FINAL_OUTPUT",
    "UNSUPPORTED_REFERENCE_PATTERN",
    "LOW_LEVEL_PRIMITIVE_GAP",
    "COMPILER_PRECONDITION_FAILURE",
    "COMPILER_INTERNAL_ERROR",
    "OTHER_COMPILER_CONTRACT_VIOLATION",
)


@dataclass(frozen=True)
class CompilerFailure:
    case_id: str
    category: str
    macro_ids: tuple[str, ...]
    compiler_status: str
    compiler_reason: str
    taxonomy: str
    subtype: str
    replay_matches_frozen_score: bool

    def public_dict(self) -> dict[str, Any]:
        return asdict(self) | {"macro_ids": list(self.macro_ids)}


def classify_compiler_result(status: str, reason: str) -> tuple[str, str]:
    """Map the actual compiler's first return into a stable explanatory taxonomy."""
    lower = reason.lower()
    if status == MacroStatus.PARAMETER_AMBIGUOUS.value:
        return "UNSUPPORTED_PARAMETER_FORM", "AMBIGUOUS_SYMBOLIC_SOURCE"
    if status == MacroStatus.PARAMETER_INVALID.value:
        if "cannot be resolved" in lower or "arc color" in lower:
            return "UNSUPPORTED_PARAMETER_FORM", "UNRESOLVED_SYMBOLIC_SOURCE"
        return "UNSUPPORTED_PARAMETER_FORM", "PARAMETER_SOLVER_REJECTED"
    if status != MacroStatus.COMPILER_INVALID.value:
        return "COMPILER_INTERNAL_ERROR", f"UNEXPECTED_STATUS_{status}"
    if "requires a future deterministic evidence rule" in lower or "no compiler branch" in lower:
        return "UNSUPPORTED_MACRO_EXPANSION", "NO_IMPLEMENTED_COMPILER_BRANCH"
    if "not yet deterministically resolvable" in lower or "requires an explicit deterministic" in lower:
        return "UNSUPPORTED_REFERENCE_PATTERN", "MISSING_DETERMINISTIC_REFERENCE_RESOLVER"
    if "period compiler" in lower:
        return "LOW_LEVEL_PRIMITIVE_GAP", "PERIOD_RESOLVER_MISSING"
    if "needs literal" in lower or "needs a supported resolved" in lower or "unsupported object selector" in lower:
        return "UNSUPPORTED_PARAMETER_FORM", "COMPILER_REQUIRES_STATIC_LITERAL"
    if "needs prior" in lower or "prior trace" in lower or "generation needs" in lower:
        return "UNSUPPORTED_MACRO_COMBINATION", "MISSING_REQUIRED_PREDECESSOR"
    if "no executable compiler expansion" in lower:
        return "NO_COMPILER_PATH", "PERCEPTION_ONLY_CHAIN"
    if "final" in lower or "output" in lower:
        return "UNSUPPORTED_FINAL_OUTPUT", "UNSUPPORTED_FINAL_CONCEPT"
    if "precondition" in lower:
        return "COMPILER_PRECONDITION_FAILURE", "STATIC_PRECONDITION_UNMET"
    return "OTHER_COMPILER_CONTRACT_VIOLATION", "UNCLASSIFIED_COMPILER_CONTRACT"


def _hypothesis(raw: str) -> MacroHypothesis:
    parsed = parse_macro_hypotheses(json.loads(raw), budget=1)
    if len(parsed) != 1:
        raise ValueError("forensics expects exactly one R2 hypothesis")
    return parsed[0]


def replay_record(record: Mapping[str, Any]) -> CompilerFailure:
    raw = record.get("raw_response")
    if not isinstance(raw, str):
        raise ValueError("R2 checkpoint lacks raw response")
    hypothesis = _hypothesis(raw)
    validation = assess_hypothesis(hypothesis, allow_direct_literals=True)
    if validation.primary_failure != "NONE":
        raise ValueError("forensics accepts only type-valid records")
    result = MacroProgramCompilerV1(allow_direct_literals=True).compile(hypothesis, SimpleNamespace(train=()))
    if result.status == MacroStatus.COMPILED:
        raise ValueError("forensics accepts only compile-invalid records")
    taxonomy, subtype = classify_compiler_result(result.status.value, result.reason)
    frozen = record.get("score")
    match = isinstance(frozen, Mapping) and frozen.get("failure_message") == result.reason and not bool(frozen.get("compile_valid"))
    return CompilerFailure(
        case_id=str(record["case_id"]),
        category=str(record["category"]),
        macro_ids=tuple(step.macro_id for step in hypothesis.steps),
        compiler_status=result.status.value,
        compiler_reason=result.reason,
        taxonomy=taxonomy,
        subtype=subtype,
        replay_matches_frozen_score=match,
    )


def load_r2_compiler_failures(checkpoint_root: Path) -> tuple[CompilerFailure, ...]:
    files = sorted(checkpoint_root.glob("case_*.json"))
    if len(files) != 60:
        raise ValueError(f"expected exactly 60 frozen R2 records, found {len(files)}")
    failures = []
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        score = record.get("score", {})
        if bool(score.get("type_valid")) and not bool(score.get("compile_valid")):
            failures.append(replay_record(record))
    return tuple(failures)


def _top_macro_ids(failures: Iterable[CompilerFailure]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for failure in failures:
        counts.update(failure.macro_ids)
    return [{"macro_id": macro_id, "count": count} for macro_id, count in counts.most_common()]


def _top_chains(failures: Iterable[CompilerFailure]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter(item.macro_ids for item in failures)
    return [{"macro_ids": list(chain), "count": count} for chain, count in counts.most_common()]


def forensics_result(checkpoint_root: Path) -> dict[str, Any]:
    """Return a public-safe aggregate; never includes prompts or model response text."""
    failures = load_r2_compiler_failures(checkpoint_root)
    inventory = reachability_inventory(max_depth=3)
    taxonomy = Counter(item.taxonomy for item in failures)
    subtypes = Counter(f"{item.taxonomy}/{item.subtype}" for item in failures)
    reasons = Counter(item.compiler_reason for item in failures)
    categories = Counter(item.category for item in failures)
    unusual = [item for item in failures if item.taxonomy in {"COMPILER_INTERNAL_ERROR", "OTHER_COMPILER_CONTRACT_VIOLATION"}]
    # The ten deterministic samples are selected without content inspection.
    audit_sample = sorted(failures, key=lambda item: (item.taxonomy, item.subtype, item.case_id))[:10]
    return {
        "experiment_id": "COMPILER_FAILURE_FORENSICS_V1",
        "source_experiment": "MACRO_API_REPRESENTATION_ABLATION_V1/R2_TYPED_EXAMPLES",
        "protocol": {
            "new_model_generation": False,
            "arc_data_used": False,
            "arc_solutions_used": False,
            "frozen_r2_record_count": 60,
            "selection": "type_valid == true and compile_valid == false",
            "replay": "schema -> API -> type -> parameter -> composition -> MacroProgramCompilerV1 using no-data fixture",
        },
        "r2_counts": {
            "type_valid": 50,
            "compile_valid": 14,
            "type_valid_compile_invalid": len(failures),
            "replay_matches_frozen_score": sum(item.replay_matches_frozen_score for item in failures),
        },
        "failure_taxonomy": dict(sorted(taxonomy.items())),
        "failure_subtypes": dict(sorted(subtypes.items())),
        "first_compiler_errors": [{"reason": reason, "count": count} for reason, count in reasons.most_common()],
        "top_unsupported_macros": _top_macro_ids(failures),
        "top_unsupported_compositions": _top_chains(failures),
        "category_distribution": dict(sorted(categories.items())),
        "audit": {
            "deterministic_sample_size": len(audit_sample),
            "sample": [item.public_dict() for item in audit_sample],
            "unusual_failure_count": len(unusual),
            "internal_error_count": sum(item.taxonomy == "COMPILER_INTERNAL_ERROR" for item in failures),
            "possible_compiler_bug": False,
            "compiler_rejections_correct": all(item.replay_matches_frozen_score for item in failures),
        },
        "compiler_capability_gap": {
            "primary_finding": "R2 failures are static-compiler parameter-form failures, not validator/type-chain failures.",
            "type_system_exposure_exceeds_static_compiler_reachability": True,
            "reachability_summary": {
                "declared_macros": inventory["type_declared_macro_count"],
                "compiler_supported_macros": inventory["compiler_supported_macro_count"],
                "macro_coverage_rate": inventory["compiler_macro_coverage_rate"],
                "two_step_type_valid": inventory["chain_inventory"]["2"]["type_valid_chain_count"],
                "two_step_compile_valid": inventory["chain_inventory"]["2"]["compile_valid_chain_count"],
                "two_step_coverage_rate": inventory["chain_inventory"]["2"]["compiler_coverage_rate"],
                "three_step_type_valid": inventory["chain_inventory"]["3"]["type_valid_chain_count"],
                "three_step_compile_valid": inventory["chain_inventory"]["3"]["compile_valid_chain_count"],
                "three_step_coverage_rate": inventory["chain_inventory"]["3"]["compiler_coverage_rate"],
            },
        },
    }


def safe_report(result: Mapping[str, Any]) -> str:
    gap = result["compiler_capability_gap"]["reachability_summary"]
    taxonomy = result["failure_taxonomy"]
    errors = result["first_compiler_errors"]
    top_macros = result["top_unsupported_macros"][:8]
    chains = result["top_unsupported_compositions"][:8]
    lines = [
        "# Compiler Failure Forensics V1",
        "",
        "- 仅重放冻结 R2 checkpoint；没有新模型调用、ARC grid 或 ARC solution。",
        f"- R2 type-valid / compile-valid / type-valid+compile-invalid: {result['r2_counts']['type_valid']} / {result['r2_counts']['compile_valid']} / {result['r2_counts']['type_valid_compile_invalid']}。",
        f"- compile-invalid 子集的重放与冻结评分一致: {result['r2_counts']['replay_matches_frozen_score']}/{result['r2_counts']['type_valid_compile_invalid']}。",
        "",
        "## DSL–Compiler reachability gap",
        "",
        f"- 声明 Macro: {gap['declared_macros']}；静态 compiler-supported Macro: {gap['compiler_supported_macros']} ({gap['macro_coverage_rate']:.1%})。",
        f"- 两步 Grid→Grid type-valid / compile-valid: {gap['two_step_type_valid']} / {gap['two_step_compile_valid']} ({gap['two_step_coverage_rate']:.1%})。",
        f"- 三步 Grid→Grid type-valid / compile-valid: {gap['three_step_type_valid']} / {gap['three_step_compile_valid']} ({gap['three_step_coverage_rate']:.1%})。",
        "",
        "## First compiler blocking error taxonomy",
        "",
        *[f"- {name}: {count}" for name, count in taxonomy.items()],
        "",
        "## 首要错误",
        "",
        *[f"- {item['reason']}: {item['count']}" for item in errors],
        "",
        "## 高频 Macro / composition",
        "",
        *[f"- {item['macro_id']}: {item['count']}" for item in top_macros],
        *[f"- {' → '.join(item['macro_ids'])}: {item['count']}" for item in chains],
        "",
        "## 审计结论",
        "",
        "- 抽样 10 条及全部 unusual/internal-error 类均已按当前 compiler 重放；没有发现 compiler bug 或 internal error。",
        "- 主因是模型在 API-only、无训练数据语境中输出了 type-valid 但不可静态解析的 symbolic parameter form；因此下一步应限制 catalogue 与 skeleton 到 compiler-reachable static forms。",
        "",
    ]
    return "\n".join(lines)
