"""Train-only SOAR-style ARC program sampling and refinement helpers.

The prompt layout and the repair-feedback contract are adapted from the public
SOAR repository (flowersteam/SOAR, commit 8ed0890).  This module deliberately
contains no ARC solver: it only formats examples, records executor evidence,
and deterministically chooses a program to refine.
"""
from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence


Grid = list[list[int]]


def _numpy_grid(grid: Grid) -> str:
    """Match SOAR's ``str(np.array(grid))`` representation without numpy."""
    return "[" + "\n ".join("[" + " ".join(str(cell) for cell in row) + "]" for row in grid) + "]"


def format_task(task: Mapping[str, Any], *, include_test: bool = True) -> str:
    """Format ARC examples in the public SOAR numpy-grid presentation."""
    sections: list[str] = []
    for index, pair in enumerate(task["train"], start=1):
        source, target = pair["input"], pair["output"]
        sections.extend(
            (
                f"## Input {index} (grid shape: {len(source)} by {len(source[0])}):\n{_numpy_grid(source)}",
                f"## Output {index} (grid shape: {len(target)} by {len(target[0])}):\n{_numpy_grid(target)}",
            )
        )
    if include_test:
        for index, pair in enumerate(task.get("test", ()), start=1):
            source = pair["input"]
            sections.append(f"## Test Input {index} (grid shape: {len(source)} by {len(source[0])}):\n{_numpy_grid(source)}")
    return "\n\n".join(sections)


_COLOR_INFO = (
    "The number in the input grid can be mapped to the following colors: "
    "0:Black; 1:Blue; 2:Red; 3:Green; 4:Yellow; 5:Grey; 6:Pink; "
    "7:Orange; 8:Purple; 9:Brown\n"
)

_INITIAL_TEMPLATE = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating Python code.
Your goal is to analyze input-output grid pairs. The outputs were produced by applying a transformation rule to the inputs. Implement the transformation rules as a Python function.
You should only write the implemented the transformation in code.
You must write code in triple backticks (```python and then ```). You must write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`).
You should make sure that you implement a version of the transformation that works in general (at least for all given input-output pairs and test input pairs).
{color_info}
Now, solve the following ARC-AGI task:

# Task to solve:
{task}
"""

_REFINEMENT_TEMPLATE = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by repairing Python code implementations.
Your goal is to analyze input-output grid pairs. The outputs were produced by applying a transformation rule to the inputs.
You will be given a python function `transform` that was supposed to implement the transformation rule, but it is not working correctly for all inputs.
You role is to fix this `transform` function.

Your solution should be:
- Accurate: Correctly fix the transformation for all given inputs so they give correct outputs as provided (it should also work for all test inputs)
- Comprehensive: Handles all possible input scenarios
- Well-structured: Uses clear, readable, and efficient code

{color_info}
**Now, repair the following ARC-AGI task implementation:**

# Task to solve:
{task}

Previous implementation:
{feedback}

Return only a complete replacement in a ```python fenced block.
"""


def initial_prompt(task: Mapping[str, Any]) -> str:
    """Official SOAR no-few-shot initial prompt, including test *input* only."""
    return _INITIAL_TEMPLATE.format(color_info=_COLOR_INFO, task=format_task(task, include_test=True)).strip()


def _render_execution(index: int, result: Mapping[str, Any]) -> str:
    if result.get("ok"):
        return f"## Output {index} computed by `transform` is correct."
    if result.get("grid") is not None:
        grid = result["grid"]
        return (
            f"## Output {index} computed by `transform` is incorrect.\n"
            f"The execution gave the following results (grid shape: {len(grid)} by {len(grid[0])}):\n{_numpy_grid(grid)}"
        )
    return f"## Output {index} computed by `transform` is incorrect. Runtime status: {result.get('status', 'UNKNOWN')}; error: {result.get('error', '')}"


def deterministic_feedback(candidate: Mapping[str, Any]) -> str:
    """Give the refiner only prior code plus train-execution evidence."""
    verification = candidate["verification"]
    executions = verification.get("train_execution", ())
    total = verification.get("train_pair_count", len(executions))
    passed = verification.get("train_pass_count", sum(bool(item.get("ok")) for item in executions))
    code = candidate.get("extracted_code") or "# No executable code was extracted."
    detail = "\n".join(_render_execution(index, result) for index, result in enumerate(executions, start=1))
    incorrect = [str(index) for index, result in enumerate(executions, start=1) if not result.get("ok")]
    suffix = "all train input-output pairs" if not incorrect else "output " + ", ".join(incorrect)
    return (
        f"```python\n{code}\n```\n"
        f"This implementation of transform function correctly worked on {passed}/{total} train input-output pairs.\n"
        f"Detailed results:\n{detail}\n\n"
        f"The previous code gave incorrect output for: {suffix}. Now, you need to fix the code to produce correct output for all inputs."
    )


def refinement_prompt(task: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    """Official SOAR repair template with deterministic train-only feedback."""
    return _REFINEMENT_TEMPLATE.format(
        color_info=_COLOR_INFO,
        task=format_task(task, include_test=True),
        feedback=deterministic_feedback(candidate),
    ).strip()


def candidate_quality(candidate: Mapping[str, Any]) -> tuple[int, int, str]:
    """Stable train-only parent ordering: exactness, passes, then code hash."""
    verification = candidate.get("verification", {})
    return (
        int(bool(candidate.get("all_train_exact") or verification.get("all_train_exact"))),
        int(verification.get("train_pass_count", 0)),
        sha256((candidate.get("extracted_code") or "").encode("utf-8")).hexdigest(),
    )


def choose_refinement_parents(candidates: Iterable[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
    """Select distinct executable parents using only verified train evidence."""
    unique: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        code = candidate.get("extracted_code")
        if code and code not in unique:
            unique[code] = candidate
    ranked = sorted(unique.values(), key=lambda item: (-candidate_quality(item)[0], -candidate_quality(item)[1], candidate_quality(item)[2]))
    return ranked[:count]
