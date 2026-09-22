from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import FAILURE_GENERATOR, FAILURE_SELECTION, FAILURE_SUCCESS, failure_source


def test_failure_taxonomy_is_exhaustive_and_ordered():
    assert failure_source(True, True) == FAILURE_SUCCESS
    assert failure_source(True, False) == FAILURE_SELECTION
    assert failure_source(False, False) == FAILURE_GENERATOR
    # A Top-1 hit necessarily entails a correct candidate; this defensive input
    # is classified as success rather than inventing an unsupported category.
    assert failure_source(False, True) == FAILURE_SUCCESS
