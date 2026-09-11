"""Frozen, provider-neutral protocol records for LLM_HYPOTHESIS_GENERATOR_V1."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from .models import CandidateResult, CandidateStatus


def candidate_metrics(results: Iterable[CandidateResult]) -> dict[str, object]:
    """Aggregate validation states without looking at test solutions."""
    items = list(results)
    counts = Counter(item.status.value for item in items)
    total = len(items)
    return {
        "candidates": total,
        "by_status": dict(sorted(counts.items())),
        "schema_valid_rate": (total - counts[CandidateStatus.SCHEMA_INVALID.value]) / total if total else None,
        "type_valid_rate": (total - counts[CandidateStatus.SCHEMA_INVALID.value] - counts[CandidateStatus.TYPE_INVALID.value]) / total if total else None,
        "executable_rate": sum(counts[state.value] for state in (CandidateStatus.TRAIN_INCONSISTENT, CandidateStatus.TRAIN_CONSISTENT)) / total if total else None,
        "train_consistent_rate": counts[CandidateStatus.TRAIN_CONSISTENT.value] / total if total else None,
    }


def blocked_result(provider_audit: list[dict[str, object]], *, registry_count: int, comprehension_questions: int) -> dict[str, object]:
    """A truthful result envelope when no configured runtime can make a model call."""
    return {
        "experiment_id": "LLM_HYPOTHESIS_GENERATOR_V1",
        "status": "BLOCKED_NO_LLM_BACKEND",
        "reason": "No configured real LLM inference backend is available; no mock or fabricated model output was substituted.",
        "provider_audit": provider_audit,
        "frozen_capability_registry_count": registry_count,
        "capability_comprehension": {"status": "BLOCKED_NO_LLM_BACKEND", "question_count": comprehension_questions, "accuracy": None, "error_categories": {}},
        "conditions": {
            "deterministic_composer_only": "FROZEN_BASELINE_AVAILABLE",
            "llm_structured_full_catalog": "BLOCKED_NO_LLM_BACKEND",
            "llm_structured_scheduler_soft_hints": "BLOCKED_NO_LLM_BACKEND",
            "deterministic_llm_union": "BLOCKED_NO_LLM_BACKEND",
            "llm_direct_grid_development": "BLOCKED_NO_LLM_BACKEND",
        },
        "metrics_not_measured": [
            "LLM exact solved", "LLM-only composition gain", "direct-grid exact solved", "token usage",
            "rank recall", "program/prediction ambiguity attributable to LLM proposals", "capability-gap classifications",
        ],
        "data_discipline": {
            "test_solution_access_for_llm_inference": False,
            "task_id_in_prompt": False,
            "task_id_hardcode": False,
            "post_llm_capability_expansion": False,
        },
    }
