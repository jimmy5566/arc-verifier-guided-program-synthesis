"""Provider-neutral data structures for LLM program-synthesis experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CandidateStatus(str, Enum):
    SCHEMA_INVALID = "SCHEMA_INVALID"
    TYPE_INVALID = "TYPE_INVALID"
    PRECONDITION_INVALID = "PRECONDITION_INVALID"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    TRAIN_INCONSISTENT = "TRAIN_INCONSISTENT"
    TRAIN_CONSISTENT = "TRAIN_CONSISTENT"


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 2_000
    context_window: int = 12_288
    seed: int | None = None
    hypothesis_budget: int = 10
    prompt_version: str = "llm_hypothesis_generator_v1.prompt.1"


@dataclass(frozen=True)
class ProviderAvailability:
    provider: str
    available: bool
    reason: str
    model: str | None = None
    package_available: bool = False
    credential_configured: bool = False


@dataclass(frozen=True)
class HypothesisStep:
    primitive_id: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMHypothesis:
    hypothesis_id: str
    steps: tuple[HypothesisStep, ...]
    confidence: float | None = None
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResponse:
    hypotheses: tuple[LLMHypothesis, ...]
    provider: str
    model: str
    elapsed_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_response_id: str | None = None
    # Diagnostic-only trace.  Main frozen checkpoints created before this field
    # intentionally remain immutable and therefore cannot be reconstructed.
    raw_response: str | None = None


@dataclass(frozen=True)
class CandidateResult:
    hypothesis_id: str
    status: CandidateStatus
    reason: str = ""
    program_depth: int = 0
    test_predictions: tuple[object, ...] = ()
