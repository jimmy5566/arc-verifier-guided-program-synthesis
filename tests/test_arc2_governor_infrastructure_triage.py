from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("governor_infra_triage", ROOT / "scripts/arc2_governor.py")
assert SPEC and SPEC.loader
governor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(governor)


class GovernorInfrastructureTriageTests(unittest.TestCase):
    def fixture(self, root: Path, **overrides):
        root.mkdir(parents=True, exist_ok=True)
        path = root / "state.json"
        state = {
            "disposition": "PAUSED",
            "infra_failure_class": "DETACHED_LAUNCH_PRE_MODEL_FAILURE",
            "infra_failure_receipt": "failed-run-001.json",
            "terminal": False,
            "experiment_terminal": False,
            "remote_job": None,
            "active_remote_job": None,
            "gpu_inference_authorized": True,
            "model_loading_authorized": True,
            "scientific_training_authorized": True,
            "_governor_log_path": str(root / "governor.log"),
        }
        state.update(overrides)
        governor.atomic(path, state)
        return path, state

    def test_recoverable_infra_pause_gets_two_cpu_only_turns_not_gpu_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, state = self.fixture(Path(tmp))
            self.assertTrue(governor.route_bounded_infrastructure_pause(state, path))
            first = governor.load(path)
            self.assertEqual(first["disposition"], "CONTINUE_CONTROLLER")
            self.assertEqual(first["next_action"], "CPU_ONLY_DIAGNOSE_AND_REPAIR_INFRASTRUCTURE")
            for key in ("gpu_inference_authorized", "model_loading_authorized", "scientific_training_authorized", "candidate_generation_authorized"):
                self.assertFalse(first[key], key)
            first["disposition"] = "PAUSED"
            self.assertTrue(governor.route_bounded_infrastructure_pause(first, path))
            second = governor.load(path)
            self.assertEqual(second["next_action"], "CPU_ONLY_VALIDATE_REPAIR_OR_FREEZE_MINIMAL_BLOCKER")
            second["disposition"] = "PAUSED"
            self.assertFalse(governor.route_bounded_infrastructure_pause(second, path))
            self.assertEqual(governor.load(path)["pause_reason"], "INFRA_TRIAGE_EXHAUSTED")

    def test_bound_review_is_routed_but_routine_repair_does_not_require_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, state = self.fixture(root / "routine")
            self.assertTrue(governor.route_bounded_infrastructure_pause(state, path))
            self.assertEqual(governor.load(path)["disposition"], "CONTINUE_CONTROLLER")
            brief = root / "brief.json"
            brief.write_text(json.dumps({"scope": "REPLACEMENT_AUTHORIZATION_ONLY"}), encoding="utf-8")
            path, state = self.fixture(root / "review", review_brief=str(brief), review_reason="consumed one-shot replacement")
            self.assertTrue(governor.route_bounded_infrastructure_pause(state, path))
            self.assertEqual(governor.load(path)["disposition"], "REVIEW_REQUIRED")

    def test_explicit_pauses_budget_sealed_security_and_identity_are_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = {
                "owner": {"owner_pause": True},
                "user": {"user_pause": True},
                "director": {"last_actor": "director"},
                "safety": {"pause_reason": "SAFETY_BLOCKER"},
                "scientific": {"pause_reason": "SCIENTIFIC_DECISION_REQUIRED"},
                "budget": {"compute_budget_exhausted": True},
                "sealed": {"sealed_data_blocker": True},
                "security": {"security_blocker": True},
                "identity": {"asset_identity_blocker": True},
                "terminal": {"experiment_terminal": True},
                "remote": {"active_remote_job": {"job_id": "live"}},
            }
            for name, flags in cases.items():
                path, state = self.fixture(root / name, **flags)
                self.assertFalse(governor.route_bounded_infrastructure_pause(state, path), name)
                self.assertEqual(governor.load(path)["disposition"], "PAUSED", name)

    def test_remote_process_death_returns_cpu_only_to_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "ARC2_WORKFLOW_STATE.json"
            state = {
                "disposition": "WAIT_REMOTE",
                "remote_job": {"job_id": "run-1"},
                "gpu_inference_authorized": True,
                "model_loading_authorized": True,
                "scientific_training_authorized": True,
                "_governor_log_path": str(root / "governor.log"),
            }
            governor.consume_remote(state, state_path, "PROCESS_DEAD", "confirmed dead")
            saved = governor.load(state_path)
            self.assertEqual(saved["disposition"], "CONTINUE_CONTROLLER")
            self.assertEqual(saved["next_action"], "CPU_ONLY_DIAGNOSE_REMOTE_PROCESS_FAILURE")
            self.assertFalse(saved["gpu_inference_authorized"])
            self.assertFalse(saved["model_loading_authorized"])
            self.assertTrue(Path(saved["infra_failure_receipt"]).is_file())

    def test_remote_receipt_wakes_controller_with_receipt_processing_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "ARC2_WORKFLOW_STATE.json"
            state = {
                "disposition": "WAIT_REMOTE",
                "remote_job": {"job_id": "run-1"},
                "next_action": "STALE_LAUNCH_ACTION",
                "_governor_log_path": str(root / "governor.log"),
            }
            governor.consume_remote(state, state_path, "RECEIPT_PRESENT", "receipt present")
            saved = governor.load(state_path)
            self.assertEqual(saved["disposition"], "CONTINUE_CONTROLLER")
            self.assertEqual(saved["next_action"], "PROCESS_REMOTE_RECEIPT")
            self.assertIsNone(saved["remote_job"])

    def test_preserved_remote_death_receipt_classifies_legacy_unlabelled_pause(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, state = self.fixture(
                Path(tmp),
                infra_failure_class=None,
                remote_completion_status="PROCESS_DEAD",
                remote_failure_receipt="machine-receipt.json",
            )
            self.assertTrue(governor.route_bounded_infrastructure_pause(state, path))
            saved = governor.load(path)
            self.assertEqual(saved["disposition"], "CONTINUE_CONTROLLER")
            self.assertFalse(saved["gpu_inference_authorized"])

    def test_scientific_pause_beats_stale_remote_death_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, state = self.fixture(
                Path(tmp),
                infra_failure_class=None,
                pause_reason="STAGE_SCOPED_STOP_REQUIRES_NEW_REVIEW_BRIEF",
                remote_completion_status="PROCESS_DEAD",
                remote_failure_receipt="old-machine-receipt.json",
            )
            self.assertFalse(governor.route_bounded_infrastructure_pause(state, path))
            self.assertEqual(governor.load(path)["disposition"], "PAUSED")


if __name__ == "__main__":
    unittest.main()
