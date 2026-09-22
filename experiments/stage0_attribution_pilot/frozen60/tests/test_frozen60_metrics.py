from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from metrics import clean_selector_target, failure_source


def test_primary_frozen60_taxonomy_shape():
    rows = ([{"any_of_k_correct": True, "top1_correct": True}] * 16 +
            [{"any_of_k_correct": True, "top1_correct": False}] * 14 +
            [{"any_of_k_correct": False, "top1_correct": False}] * 30)
    assert len(rows) == 60
    assert sum(row["any_of_k_correct"] for row in rows) == 30
    assert sum(row["top1_correct"] for row in rows) == 16
    assert clean_selector_target(rows) == 16 / 30
    assert sum(failure_source(**row) == "generator_limited_failure" for row in rows) == 30
