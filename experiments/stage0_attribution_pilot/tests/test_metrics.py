from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import beta_estimate, clean_selector_target, reliability_variants


ROWS = [
    {"any_of_k_correct": True, "top1_correct": True},
    {"any_of_k_correct": True, "top1_correct": False},
    {"any_of_k_correct": False, "top1_correct": False},
]


def test_clean_target_excludes_generator_limited_tasks():
    assert clean_selector_target(ROWS) == 0.5


def test_shared_prior_variants_have_expected_counts():
    variants = reliability_variants(ROWS)
    assert variants["NAIVE"].estimate == pytest.approx(2 / 5)
    assert variants["ORACLE_SKIP"].estimate == pytest.approx(2 / 4)
    assert variants["ORACLE_CLEAN"] == variants["ORACLE_SKIP"]


def test_beta_estimate_rejects_invalid_counts():
    with pytest.raises(ValueError):
        beta_estimate(-1, 0)
