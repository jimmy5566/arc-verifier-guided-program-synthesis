import numpy as np
import pytest
from evaluation.submission import validate_submission
from verification.verifier import verify_prediction
def test_verifier_partial_metrics():
    result=verify_prediction(np.array([[1,2]]),np.array([[1,3]])); assert not result.exact_match and result.shape_match and result.pixel_accuracy==0.5
def test_submission_validator():
    payload={"a":[{"attempt_1":[[0]],"attempt_2":[[1]]}]}; validate_submission(payload,{"a"},{"a":1})
    with pytest.raises(ValueError): validate_submission({"a":[]},{"a"},{"a":1})
