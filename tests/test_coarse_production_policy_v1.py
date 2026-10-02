"""CPU-only checks for the deterministic coarse fallback rules."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import run_coarse_production_policy_v1 as coarse  # noqa: E402


def test_physical_stage_reduces_only_physical_ceiling_first() -> None:
    configs = [coarse._config(4, 4), coarse._config(4, 2), coarse._config(4, 1), coarse._config(2, 2)]
    assert coarse._next_fallback(configs, 0, "BEFORE_MODEL_FORWARD") == 1


def test_admission_stage_reduces_resident_capacity() -> None:
    configs = [coarse._config(4, 4), coarse._config(4, 2), coarse._config(2, 2), coarse._config(2, 1)]
    assert coarse._next_fallback(configs, 0, "BEFORE_CELL_PREFILL") == 2


def test_ladder_is_deterministic_and_bounded() -> None:
    assert coarse._fallbacks(coarse._config(4, 4)) == [
        coarse._config(4, 2), coarse._config(4, 1), coarse._config(2, 2), coarse._config(2, 1), coarse._config(1, 1),
    ]
