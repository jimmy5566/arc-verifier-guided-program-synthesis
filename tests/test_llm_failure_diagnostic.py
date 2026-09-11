from __future__ import annotations

import numpy as np

from arc.task import ARCExample, ARCGrid, ARCTask
from llm.retrieval import retrieve_capability_ids
from scripts.analyze_llm_failure_diagnostic import program_failure_counts, status_funnel
from scripts.finalize_llm_retrieval_diagnostic import candidate_metrics


def _task() -> ARCTask:
    grid = np.array([[0, 2, 0], [2, 2, 2], [0, 2, 0]], dtype=np.int16)
    return ARCTask("no_id_in_retrieval", (ARCExample(ARCGrid(grid), ARCGrid(grid)),), (ARCExample(ARCGrid(grid)),))


def test_retrieval_is_deterministic_and_bounded_without_task_id_features():
    first = retrieve_capability_ids(_task(), 15)
    assert first == retrieve_capability_ids(_task(), 15)
    expanded = retrieve_capability_ids(_task(), 30)
    assert len(first) == 15 and len(set(first)) == 15
    assert first == expanded[:15]


def test_historical_checkpoint_funnel_and_failure_counts_do_not_invent_traces():
    checkpoint = {
        "task_count": 2,
        "records": {
            "a": {"candidate_statuses": ["SCHEMA_INVALID"], "candidate_reasons": ["provider_error:JSONDecodeError"], "prediction": None},
            "b": {"candidate_statuses": ["PRECONDITION_INVALID", "TRAIN_INCONSISTENT"], "candidate_reasons": ["unknown parameter(s): color", "prediction differs"], "prediction": None},
        },
    }
    funnel = status_funnel(checkpoint, {"exact_solved": 0})
    assert funnel["parseable_response"]["count"] == 1
    assert funnel["schema_valid_program"]["count"] == 1
    assert funnel["executable_program"]["count"] == 1
    counts = program_failure_counts(checkpoint, None)
    assert counts["program_counts"]["malformed_json"] == 1
    assert counts["program_counts"]["precondition_failure"] == 1
    assert "program depth" in counts["unavailable_historical_fields"]


def test_retrieval_diagnostic_metrics_count_traced_steps_only():
    checkpoint = {
        "records": {
            "a": {
                "candidate_results": [{"hypothesis_id": "h", "status": "TRAIN_INCONSISTENT", "program_depth": 2}],
                "parsed_hypotheses": [{"hypothesis_id": "h", "steps": [{"primitive_id": "REG_FILL_INTERIOR_V1", "params": {}}]},],
            }
        }
    }
    metrics = candidate_metrics(checkpoint)
    assert metrics["candidate_count"] == 1
    assert metrics["executable_rate"] == 1.0
    assert metrics["family_usage"]["Region"]["proposed"] == 1
