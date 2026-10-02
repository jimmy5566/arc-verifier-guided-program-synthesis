"""CPU-only classification tests for the same-contract Core gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("repro_gate", ROOT / "scripts" / "run_ttt24_aug8_r1024_reproducibility_gate_v1.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def cell(*, owner: int = 1, score: float = 0.1, token: int = 2, candidate: int = 2) -> dict:
    return {
        "cell_key": "task:o0:d24:aug8:geom=identity__color=id__order=canonical",
        "augmentation_id": "geom=identity__color=id__order=canonical",
        "nodes": [{"node_id": 1, "parent_node_id": 0, "selected_token": token, "token_position": 3,
                   "branch_depth": 1, "token_logprob": -score, "cumulative_score": score, "cumulative_regret": 0.0}],
        "events": [], "final_candidate_pool": [{"token_ids": [candidate]}], "nodes_expanded": 1,
        "completed_candidates": 1, "termination_reason": "budget", "budget_exhausted": True,
        "diagnostic_trace": {"prefill": {"prompt_token_ids": [1, 2], "prompt_length": 2,
            "full_logits_sha256": "logits", "root_cache_sha256": "cache", "root_cache_valid_length": 2,
            "root_cache_geometry": "shape", "first_ready_request": {"cache_owner_id": owner, "parent_node_id": 0,
                "token_id": 2, "position": 2, "cache_compatibility_key": "k", "valid_kv_length": 2}},
            "search_trace": [], "frontier_samples": [], "branch_probabilities": []},
    }


def raw(value: dict) -> dict:
    return {"output": {"profile": "PROFILE_S"}, "root_admission": {"root_lengths": {value["cell_key"]: 2}},
            "cells": {"cell": value}, "scheduler_events": []}


class ReproducibilityGateTests(unittest.TestCase):
    def test_process_local_owner_ids_are_ignored(self) -> None:
        row, _ = MODULE._compare_cell("task:o0", cell()["augmentation_id"], raw(cell(owner=1)), raw(cell(owner=2)))
        self.assertEqual("EXACT_SEMANTIC_AND_NUMERIC", row["classification"])

    def test_timing_and_memory_telemetry_are_ignored(self) -> None:
        a, b = raw(cell()), raw(cell())
        a["scheduler_events"] = [{"event": "FORWARD", "selected_cell_keys": ["cell"], "physical_batch": 1,
                                  "host_model_call_seconds": 1.0, "memory_allocated_bytes": 1}]
        b["scheduler_events"] = [{"event": "FORWARD", "selected_cell_keys": ["cell"], "physical_batch": 1,
                                  "host_model_call_seconds": 9.0, "memory_allocated_bytes": 99}]
        self.assertEqual(MODULE._scheduler_projection(a), MODULE._scheduler_projection(b))

    def test_score_only_drift_is_semantic_exact_numeric_drift(self) -> None:
        row, _ = MODULE._compare_cell("task:o0", cell()["augmentation_id"], raw(cell(score=0.1)), raw(cell(score=0.100001)))
        self.assertEqual("SEMANTIC_EXACT_NUMERIC_DRIFT", row["classification"])

    def test_selected_token_difference_is_logical_divergence(self) -> None:
        row, _ = MODULE._compare_cell("task:o0", cell()["augmentation_id"], raw(cell(token=2)), raw(cell(token=3)))
        self.assertEqual("LOGICAL_DIVERGENCE", row["classification"])

    def test_candidate_difference_is_logical_divergence(self) -> None:
        row, _ = MODULE._compare_cell("task:o0", cell()["augmentation_id"], raw(cell(candidate=2)), raw(cell(candidate=3)))
        self.assertEqual("LOGICAL_DIVERGENCE", row["classification"])

    def test_scheduler_batch_timing_is_ignored_but_semantic_change_is_not(self) -> None:
        a = {"scheduler_events": [{"event": "FORWARD", "selected_cell_keys": ["a", "b"], "physical_batch": 2,
                                    "request_position": 4, "host_model_call_seconds": 1.0}]}
        b = {"scheduler_events": [{"event": "FORWARD", "selected_cell_keys": ["a", "b"], "physical_batch": 2,
                                    "request_position": 4, "host_model_call_seconds": 2.0}]}
        self.assertEqual(MODULE._scheduler_projection(a), MODULE._scheduler_projection(b))
        b["scheduler_events"][0]["selected_cell_keys"] = ["b", "a"]
        self.assertNotEqual(MODULE._scheduler_projection(a), MODULE._scheduler_projection(b))

    def test_incompatible_or_over_ceiling_batch_fails_invariant(self) -> None:
        value = cell()
        value["diagnostic_trace"]["logical_advances"] = [{"cache_owner_id": 1, "cache_owner_id_after_reply": 1,
            "absolute_position": 2, "valid_kv_length": 2, "output_valid_kv_length": 3,
            "physical_batch_width": 3}]
        payload = raw(value); payload["profile_configuration"] = {"physical_batch_ceiling": 2}
        self.assertEqual("FAIL", MODULE._cache_invariants(payload)["status"])


if __name__ == "__main__":
    unittest.main()
