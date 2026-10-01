from scripts.audit_round_robin_scheduler import audit


VIEWS = ("anti_transpose", "flip_ud", "identity", "transpose")


def _raw():
    keys = [f"task:o0:d24:{view}" for view in VIEWS]
    traces = {
        key: [{"request_ordinal": ordinal} for ordinal in range(2)]
        for key in keys
    }
    events = []
    cursor = 0
    consumed = {key: 0 for key in keys}
    for forward, key in enumerate(keys * 2, start=1):
        events.append({
            "forward_index": forward,
            "physical_batch": 1,
            "cell_keys": [key],
            "request_ordinals": [consumed[key]],
            "ready_cell_keys": [item for item in keys if consumed[item] < 2],
            "round_robin_cursor_before": cursor,
            "round_robin_cursor_after": (cursor + 1) % len(keys),
        })
        consumed[key] += 1
        cursor = (cursor + 1) % len(keys)
    return {
        "mode": "round-robin",
        "target_blind": True,
        "gold_loaded": False,
        "task": {"task_id": "task", "output_index": 0, "depth": 24},
        "cells": [{"cell_key": key, "per_forward_trace": traces[key]} for key in keys],
        "scheduler": {
            "scheduling_policy": "round_robin",
            "frozen_logical_order": list(VIEWS),
            "events": events,
        },
    }


def test_audit_accepts_exact_b1_round_robin_trace():
    result = audit(_raw(), "test")
    assert result["status"] == "PASS"


def test_audit_rejects_monopolizing_lexical_trace_even_if_label_claims_round_robin():
    raw = _raw()
    raw["scheduler"]["events"][1]["cell_keys"] = ["task:o0:d24:anti_transpose"]
    result = audit(raw, "test")
    assert result["status"] == "ROUND_ROBIN_IMPLEMENTATION_FAIL"
    assert any(error["reason"] == "cursor_progression_or_starvation" for error in result["errors"])
