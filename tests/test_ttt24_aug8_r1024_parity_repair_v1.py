"""CPU-only regression tests for the Phase-1 forensic comparator."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ttt24_aug8_r1024_parity_repair_v1.py"
SPEC = importlib.util.spec_from_file_location("parity_repair", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ParityRepairComparatorTests(unittest.TestCase):
    def test_score_only_pilot_difference_is_numerical_drift(self) -> None:
        augmentation = "geom=identity__color=id__order=canonical"
        old = {"cells": {"cell": {"augmentation_id": augmentation, "nodes": [{
            "state": "expanded", "node_id": 1, "parent_node_id": 0, "token_position": 7,
            "selected_token": 2, "token_logprob": -0.1, "cumulative_score": 0.1,
            "cumulative_regret": 0.0, "branch_depth": 1,
        }]}}}
        replay = {"cells": {"cell": {"augmentation_id": augmentation, "nodes": [{
            "state": "expanded", "node_id": 1, "parent_node_id": 0, "token_position": 7,
            "selected_token": 2, "token_logprob": -0.100001, "cumulative_score": 0.100001,
            "cumulative_regret": 0.0, "branch_depth": 1,
        }]}}}
        result = MODULE._old_vs_replay(old, replay, augmentation, 128)
        self.assertEqual("DIVERGED", result["status"])
        self.assertEqual("NUMERICAL_SCORE_DRIFT_SAME_NODE_SEQUENCE", result["category"])

    def test_prefill_ignores_only_run_local_owner_identity(self) -> None:
        output_id = "test:o0"
        augmentation = "geom=identity__color=id__order=canonical"
        original_cells = MODULE.TRACE_CELLS
        MODULE.TRACE_CELLS = ((output_id, augmentation, "TEST"),)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                roots = {}
                for name, owner in (("A", 101), ("B", 202), ("C", 303)):
                    root = Path(temporary) / name
                    raw_dir = root / "RAW_OUTPUTS"
                    raw_dir.mkdir(parents=True)
                    prefill = {
                        "prompt_token_ids": [1, 2], "prompt_length": 2, "full_logits_sha256": "logits",
                        "root_cache_sha256": "cache", "root_cache_valid_length": 2, "root_cache_geometry": "shape",
                        "first_ready_request": {
                            "cache_owner_id": owner, "cache_compatibility_key": "key", "parent_node_id": 0,
                            "position": 2, "request_ordinal": 0, "token_id": 1, "valid_kv_length": 2,
                        },
                    }
                    payload = {"cells": {"cell": {"augmentation_id": augmentation, "diagnostic_trace": {"prefill": prefill}}}}
                    (raw_dir / "test_o0.json").write_text(json.dumps(payload), encoding="utf-8")
                    roots[name] = root
                rows, passed = MODULE._prefill_rows(roots)
        finally:
            MODULE.TRACE_CELLS = original_cells
        self.assertTrue(passed)
        self.assertTrue(rows[0]["prefill_equal_A_B_C"])

    def test_numeric_pilot_drift_requires_gate_redesign(self) -> None:
        rows = [
            {"comparison": "A_VS_PILOT", "status": "DIVERGED", "earliest_divergence_category": "NUMERICAL_SCORE_DRIFT_SAME_NODE_SEQUENCE", "fields": ["token_logprob"]},
            {"comparison": "B_VS_A", "status": "EXACT", "earliest_divergence_category": "NONE", "fields": []},
            {"comparison": "C_VS_B", "status": "EXACT", "earliest_divergence_category": "NONE", "fields": []},
        ]
        result = MODULE._classification({"first_rows": rows, "prefill_equal": True, "cache_audit": {"status": "PASS"}})
        self.assertEqual("MODEL_CALL_HISTORY_NUMERICAL_DIVERGENCE", result["classification"])
        self.assertEqual("PHASE1_GATE_REDESIGN_REQUIRED", result["decision"])


if __name__ == "__main__":
    unittest.main()
