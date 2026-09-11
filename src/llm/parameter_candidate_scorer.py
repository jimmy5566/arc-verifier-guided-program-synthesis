"""Deterministic forward-likelihood scoring over legal parameter candidates.

There is no completion parsing in this module.  A scorer receives a fixed
context and scores supplied canonical candidate IDs via teacher forcing.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .compiler_aware_interface import CompilerValidSkeleton
from .parameter_grounding import ParameterSlot
from .parameter_semantic_ontology import ParameterCandidate, ParameterSemanticOntologyV1, stable_hash


class ContinuationScorer(Protocol):
    def score_continuations(self, prompt: str, continuations: tuple[str, ...]) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class CandidateScore:
    candidate: ParameterCandidate
    score: float

    def public(self) -> dict[str, object]:
        return {"candidate_id": self.candidate.candidate_id, "score": self.score}


@dataclass(frozen=True)
class CandidateRanking:
    scores: tuple[CandidateScore, ...]
    top1: ParameterCandidate | None
    top2: ParameterCandidate | None
    margin: float | None

    def public(self) -> dict[str, object]:
        return {
            "scores": [item.public() for item in self.scores],
            "top1": None if self.top1 is None else self.top1.candidate_id,
            "top2": None if self.top2 is None else self.top2.candidate_id,
            "margin": self.margin,
        }


def _payload(instruction: str, skeleton: CompilerValidSkeleton, slot: ParameterSlot, candidates: tuple[ParameterCandidate, ...]) -> dict[str, Any]:
    return {
        "instruction": instruction,
        "frozen_skeleton": {"skeleton_id": skeleton.skeleton_id, "macro_ids": list(skeleton.macro_ids)},
        "target_slot": {"slot": slot.key, "macro_id": slot.macro_id, "parameter": slot.parameter},
        "allowed_candidates": [{"candidate_id": item.candidate_id, "description": item.semantic_description} for item in candidates],
        "rule": "Choose the one legal candidate that best expresses the instruction. Family, skeleton, Macro IDs and order are frozen.",
    }


def likelihood_prompt(instruction: str, skeleton: CompilerValidSkeleton, slot: ParameterSlot, candidates: tuple[ParameterCandidate, ...]) -> str:
    payload = _payload(instruction, skeleton, slot, candidates)
    payload["answer_prefix"] = "The most consistent canonical candidate ID is:"
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def pairwise_prompt(instruction: str, skeleton: CompilerValidSkeleton, slot: ParameterSlot, left: ParameterCandidate, right: ParameterCandidate) -> str:
    payload = _payload(instruction, skeleton, slot, (left, right))
    payload["rule"] = "Compare only the two legal candidates. The more semantically consistent candidate ID is:"
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class ParameterLikelihoodScorerV1:
    normalization = "mean_teacher_forced_log_probability_per_continuation_token"

    def __init__(self, scorer: ContinuationScorer) -> None:
        self.scorer = scorer

    def rank(self, instruction: str, skeleton: CompilerValidSkeleton, slot: ParameterSlot, ontology: ParameterSemanticOntologyV1) -> CandidateRanking:
        candidates = ontology.candidates_for_slot(slot)
        if not candidates:
            return CandidateRanking((), None, None, None)
        values = self.scorer.score_continuations(likelihood_prompt(instruction, skeleton, slot, candidates), tuple(item.candidate_id for item in candidates))
        if len(values) != len(candidates):
            raise ValueError("continuation scorer returned wrong candidate count")
        ranked = tuple(sorted((CandidateScore(candidate, float(score)) for candidate, score in zip(candidates, values)), key=lambda item: (-item.score, item.candidate.candidate_id)))
        return CandidateRanking(ranked, ranked[0].candidate, ranked[1].candidate if len(ranked) > 1 else None, ranked[0].score - ranked[1].score if len(ranked) > 1 else float("inf"))


class PairwiseContrastiveScorerV1:
    """Deterministic ordered tournament; each match is a two-choice logit test."""

    normalization = "mean_teacher_forced_log_probability_per_candidate_id_token"

    def __init__(self, scorer: ContinuationScorer) -> None:
        self.scorer = scorer

    def rank(self, instruction: str, skeleton: CompilerValidSkeleton, slot: ParameterSlot, ontology: ParameterSemanticOntologyV1) -> CandidateRanking:
        candidates = ontology.candidates_for_slot(slot)
        if not candidates:
            return CandidateRanking((), None, None, None)
        champion = candidates[0]; match_margins: dict[str, float] = {candidate.candidate_id: 0.0 for candidate in candidates}
        for challenger in candidates[1:]:
            prompt = pairwise_prompt(instruction, skeleton, slot, champion, challenger)
            left, right = self.scorer.score_continuations(prompt, (champion.candidate_id, challenger.candidate_id))
            if right > left or (right == left and challenger.candidate_id < champion.candidate_id):
                match_margins[challenger.candidate_id] += float(right - left); champion = challenger
            else:
                match_margins[champion.candidate_id] += float(left - right)
        # A stable ranking is enough for consensus; only the tournament winner
        # and its final margin are used by the repair gate.
        ranked = tuple(sorted((CandidateScore(candidate, match_margins[candidate.candidate_id] + (1.0 if candidate == champion else 0.0)) for candidate in candidates), key=lambda item: (-item.score, item.candidate.candidate_id)))
        return CandidateRanking(ranked, champion, ranked[1].candidate if len(ranked) > 1 else None, match_margins[champion.candidate_id])


def scorer_config_hash() -> str:
    return stable_hash({
        "source_b": {"normalization": ParameterLikelihoodScorerV1.normalization, "prompt": "candidate_id_teacher_forcing.v1"},
        "source_c": {"normalization": PairwiseContrastiveScorerV1.normalization, "aggregation": "sorted_sequential_tournament.v1"},
        "no_generation": True,
    })
