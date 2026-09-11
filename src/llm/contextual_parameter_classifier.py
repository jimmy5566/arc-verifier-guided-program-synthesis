"""Teacher-forced context-aware classification over legal candidate IDs only."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .contextual_parameter_ir import ContextualParameterIRV1
from .parameter_candidate_scorer import CandidateRanking, CandidateScore, ContinuationScorer
from .parameter_semantic_ontology import stable_hash


def contextual_prompt(context: ContextualParameterIRV1) -> str:
    payload = context.public() | {
        "rule": "Choose exactly one supplied candidate ID. Do not explain, generate DSL, or invent a candidate.",
        "answer_prefix": "The most context-consistent candidate ID is:",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class ContextualParameterClassifierV1:
    normalization = "mean_teacher_forced_log_probability_per_candidate_id_token"

    def __init__(self, scorer: ContinuationScorer) -> None:
        self.scorer = scorer

    def rank(self, context: ContextualParameterIRV1) -> CandidateRanking:
        ids = tuple(item.candidate_id for item in context.allowed_candidates)
        values = self.scorer.score_continuations(contextual_prompt(context), ids)
        if len(values) != len(ids):
            raise ValueError("continuation scorer returned wrong candidate count")
        ranked = tuple(sorted((CandidateScore(candidate, float(score)) for candidate, score in zip(context.allowed_candidates, values)), key=lambda item: (-item.score, item.candidate.candidate_id)))
        return CandidateRanking(ranked, ranked[0].candidate, ranked[1].candidate if len(ranked) > 1 else None, ranked[0].score - ranked[1].score if len(ranked) > 1 else float("inf"))


def classifier_config_hash() -> str:
    return stable_hash({"id": "CONTEXTUAL_PARAMETER_CLASSIFIER_V1", "normalization": ContextualParameterClassifierV1.normalization, "prompt": "typed_context_ir_candidate_id_teacher_forcing.v1", "generation_calls": 0})
