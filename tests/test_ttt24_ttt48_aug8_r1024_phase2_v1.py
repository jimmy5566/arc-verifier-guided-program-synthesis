"""CPU-only tests for the Phase-2-only orchestration contract."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase2", ROOT / "scripts" / "run_ttt24_ttt48_aug8_r1024_phase2_v1.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Phase2ContractTests(unittest.TestCase):
    def test_frozen_target_ids_and_profiles_are_exact(self) -> None:
        self.assertEqual(("58490d8a:o0", "78332cb0:o1", "b5ca7ac4:o0"), MODULE.EXPECTED_OUTPUT_IDS)
        self.assertEqual("PROFILE_M", MODULE.EXPECTED_PROFILES["58490d8a:o0"])
        self.assertEqual("PROFILE_S", MODULE.EXPECTED_PROFILES["78332cb0:o1"])
        self.assertEqual("PROFILE_L_LOW", MODULE.EXPECTED_PROFILES["b5ca7ac4:o0"])

    def test_phase1_requires_ready_32_cell_authorization(self) -> None:
        good = {"status": "CORE_CLASS_READY", "core_status": "READY", "phase2": "AUTHORIZED_BUT_NOT_STARTED",
                "phase2_started": False, "gold_loaded": False, "target_blind": True,
                "reproducibility_evidence": {"strict_exact_cells": 32}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "phase1.json"
            path.write_text(json.dumps(good), encoding="utf-8")
            self.assertEqual(good, MODULE._check_phase1(path))
            good["reproducibility_evidence"]["strict_exact_cells"] = 31
            path.write_text(json.dumps(good), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "PHASE1_STRICT_EXACT_32_REQUIRED"):
                MODULE._check_phase1(path)


if __name__ == "__main__":
    unittest.main()
