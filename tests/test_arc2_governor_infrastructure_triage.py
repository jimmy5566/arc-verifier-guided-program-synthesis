from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("governor_infra_triage", ROOT / "scripts" / "arc2_governor.py")
assert SPEC and SPEC.loader
governor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(governor)


class GovernorInfrastructureTriageTests(unittest.TestCase):
    def fixture(self, root: Path, **overrides):
        root.mkdir(parents=True, exist_ok=True)
        path = root / "state.json"
        state = {
            "disposition": "PAUSED",
            "infra_failure_class": "DETACHED_LAUNCH_EXITED_NO_TERMINAL_RECEIPT",
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
            self.assertEqual(first["next_action"], "CPU_ONLY_DIAGNOSE_EXISTING_INFRA_FAILURE")
            for key in ("gpu_inference_authorized", "model_loading_authorized", "scientific_training_authorized", "candidate_generation_authorized"):
                self.assertFalse(first[key], key)
            self.assertEqual(first["infra_cpu_requeues"], 1)
            first["disposition"] = "PAUSED"
            self.assertTrue(governor.route_bounded_infrastructure_pause(first, path))
            second = governor.load(path)
            self.assertEqual(second["next_action"], "CPU_ONLY_FREEZE_DIRECTOR_INFRA_RETRY_BRIEF_OR_EXPLICIT_BLOCKER")
            self.assertEqual(second["infra_cpu_requeues"], 2)
            second["disposition"] = "PAUSED"
            self.assertFalse(governor.route_bounded_infrastructure_pause(second, path))
            last = governor.load(path)
            self.assertEqual(last["disposition"], "PAUSED")
            self.assertEqual(last["pause_reason"], "INFRA_TRIAGE_EXHAUSTED_DIRECTOR_BRIEF_MISSING")
            self.assertFalse(last["gpu_inference_authorized"])

    def test_existing_frozen_review_brief_reaches_director_without_prompting_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brief = root / "director-brief.json"
            brief.write_text(json.dumps({"scientific_scope": "UNCHANGED", "failure_class": "INFRA_LAUNCH"}), encoding="utf8")
            path, state = self.fixture(root, review_brief=str(brief), review_reason="One-shot same-science retry review")
            self.assertTrue(governor.route_bounded_infrastructure_pause(state, path))
            saved = governor.load(path)
            self.assertEqual(saved["disposition"], "REVIEW_REQUIRED")
            self.assertEqual(saved["review_brief"], str(brief))
            self.assertEqual(saved["infra_cpu_requeues"], 0)

    def test_owner_director_or_scientific_pause_is_not_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, flags in (
                ("owner", {"owner_pause": True}),
                ("director", {"last_actor": "director"}),
                ("safety", {"pause_reason": "SAFETY_BLOCKER"}),
                ("terminal", {"experiment_terminal": True}),
                ("active-remote", {"active_remote_job": {"job_id": "123"}}),
                ("scientific", {"infra_failure_class": None}),
            ):
                path, state = self.fixture(root / name, **flags)
                self.assertFalse(governor.route_bounded_infrastructure_pause(state, path), name)
                self.assertEqual(governor.load(path)["disposition"], "PAUSED", name)

    def test_new_incident_resets_only_the_bounded_cpu_turn_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, state = self.fixture(Path(tmp), infra_recovery_incident="failed-run-000.json", infra_cpu_requeues=2)
            self.assertTrue(governor.route_bounded_infrastructure_pause(state, path))
            self.assertEqual(governor.load(path)["infra_cpu_requeues"], 1)
            self.assertEqual(governor.load(path)["infra_recovery_incident"], "failed-run-001.json")


if __name__ == "__main__":
    unittest.main()
