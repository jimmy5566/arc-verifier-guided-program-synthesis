from inference.soar_refinement import choose_refinement_parents, deterministic_feedback, initial_prompt, refinement_prompt


TASK = {
    "train": [{"input": [[1, 0]], "output": [[0, 1]]}],
    "test": [{"input": [[2, 0]]}],
}


def _candidate(code: str, passed: int, *, exact: bool = False):
    return {
        "extracted_code": code,
        "all_train_exact": exact,
        "verification": {
            "all_train_exact": exact,
            "train_pair_count": 1,
            "train_pass_count": passed,
            "train_execution": [{"ok": bool(passed), "grid": [[0, 1]], "status": "SUCCESS"}],
        },
    }


def test_official_style_initial_prompt_exposes_train_and_test_input_only():
    prompt = initial_prompt(TASK)
    assert "## Input 1" in prompt
    assert "## Output 1" in prompt
    assert "## Test Input 1" in prompt
    assert "Test Output" not in prompt
    assert "function called `transform`" in prompt


def test_refinement_feedback_contains_train_evidence_not_target():
    candidate = _candidate("def transform(x):\n return x", 0)
    feedback = deterministic_feedback(candidate)
    prompt = refinement_prompt(TASK, candidate)
    assert "0/1 train input-output pairs" in feedback
    assert "Runtime status" not in feedback
    assert "Previous implementation:" in prompt
    assert "Test Output" not in prompt


def test_parent_selection_is_train_only_deterministic_and_deduplicated():
    low = _candidate("def transform(x):\n return x", 0)
    high = _candidate("def transform(x):\n return x[::-1]", 1, exact=True)
    duplicate_high = _candidate("def transform(x):\n return x[::-1]", 1, exact=True)
    assert choose_refinement_parents([low, duplicate_high, high], 2) == [high, low]
