"""Typed finite parameter hypotheses, independent of LLM and execution."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

from v3.schema.rule_skeleton import ParameterSlot


@dataclass(frozen=True)
class ParameterHypotheses:
    candidates: Mapping[ParameterSlot, frozenset[Any]]

    def assignments(self) -> Sequence[Mapping[ParameterSlot, Any]]:
        slots = tuple(sorted(self.candidates, key=lambda item: item.value))
        if any(not self.candidates[slot] for slot in slots):
            return ()
        return tuple(dict(zip(slots, values)) for values in product(*(sorted(self.candidates[slot], key=repr) for slot in slots)))
