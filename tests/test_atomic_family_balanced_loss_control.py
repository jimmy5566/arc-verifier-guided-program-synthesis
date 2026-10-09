from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/forward_capability_repair_v2/atomic_prerequisite_family_balanced_loss_control_v1"
SCHEDULE = ROOT / "experiments/capability_repair_baseline_v1/forward_capability_repair_v2/atomic_prerequisite_repair_successor_proposal_v1/concrete_protocol_v1/ATOMIC_PREREQUISITE_TRAINING_SCHEDULE_V1.json"
LEDGER = BASE / "ATOMIC_PREREQUISITE_FAMILY_BALANCED_LOSS_COEFFICIENT_LEDGER_V1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("balanced", ROOT / "scripts/prepare_atomic_family_balanced_loss_control_v1.py")
balanced = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(balanced)

worker_spec = importlib.util.spec_from_file_location("worker", ROOT / "scripts/run_same_color_preservation_stage_a_v1.py")
worker = importlib.util.module_from_spec(worker_spec)
assert worker_spec.loader is not None
worker_spec.loader.exec_module(worker)


class FamilyBalancedLossControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
        self.ledger = json.loads(LEDGER.read_text(encoding="utf-8"))

    def test_ledger_equalizes_weak_families_without_global_scale_change(self) -> None:
        check = self.ledger["validation"]
        self.assertTrue(check["all_step_scales_one"])
        self.assertTrue(check["all_coefficients_positive"])
        self.assertTrue(check["weak_totals_equal"])
        self.assertTrue(check["weak_role_total_preserved"])
        self.assertTrue(check["nonweak_totals_preserved"])
        totals = self.ledger["family_totals"]
        weak = [totals[name]["balanced_coefficient_total"] for name in balanced.WEAK]
        self.assertLess(max(weak) - min(weak), 1e-10)
        self.assertAlmostEqual(sum(step["scalar_loss_scale"] for step in self.ledger["per_optimizer_step"]), 100.0, places=10)

    def test_ledger_binds_exact_schedule_and_episode_order(self) -> None:
        self.assertEqual(self.ledger["schedule_sha256"], sha(SCHEDULE))
        episodes = self.schedule["episodes"]
        observed = [entry["episode_id"] for step in self.ledger["per_optimizer_step"] for entry in step["entries"]]
        self.assertEqual(observed, [row["episode_id"] for row in episodes])

    def test_worker_accepts_only_the_bound_balanced_ledger(self) -> None:
        work = [{"episode_id": row["episode_id"], "supervised": row["supervised_token_count"]} for row in self.schedule["episodes"]]
        config = {
            "gradient_accumulation": 4,
            "loss_aggregation_policy": "FAMILY_BALANCED_GLOBAL_IPF_V1",
            "loss_coefficient_ledger_path": str(LEDGER.relative_to(ROOT)).replace("\\", "/"),
            "loss_coefficient_ledger_sha256": sha(LEDGER),
            "schedule_sha256": sha(SCHEDULE),
        }
        values = worker.load_loss_coefficients(config, work)
        self.assertEqual(len(values), 100)
        self.assertTrue(all(abs(sum(step) - 1.0) < 1e-10 for step in values))
        config["loss_coefficient_ledger_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "LOSS_COEFFICIENT_LEDGER_IDENTITY_FAIL"):
            worker.load_loss_coefficients(config, work)

    def test_pre_model_interface_rejects_output_root_mismatch_before_loading(self) -> None:
        config = json.loads((BASE / "ATOMIC_PREREQUISITE_FAMILY_BALANCED_LOSS_CONTROL_V1_REMOTE_RUN_CONFIG_DRAFT.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(RuntimeError, "OUTPUT_ROOT_CONFIG_MISMATCH"):
            worker.validate_pre_model_config(config, "/workspace/arc2/wrong-output-root")

    def test_pre_model_interface_requires_complete_config(self) -> None:
        config = json.loads((BASE / "ATOMIC_PREREQUISITE_FAMILY_BALANCED_LOSS_CONTROL_V1_REMOTE_RUN_CONFIG_DRAFT.json").read_text(encoding="utf-8"))
        config.pop("train_sha256")
        with self.assertRaisesRegex(RuntimeError, "REQUIRED_CONFIG_KEY_MISSING:train_sha256"):
            worker.validate_pre_model_config(config, config["output_root"])

    def _authenticated_review_paths(self, directory: Path) -> tuple[dict, Path, Path, Path, Path]:
        config = json.loads((BASE / "ATOMIC_PREREQUISITE_FAMILY_BALANCED_LOSS_CONTROL_V1_REMOTE_RUN_CONFIG_DRAFT.json").read_text(encoding="utf-8"))
        config_path = directory / "config.json"
        protocol_path = directory / "protocol.json"
        brief_path = directory / "brief.json"
        review_path = directory / "review.json"
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        protocol_path.write_text('{"protocol": "test"}', encoding="utf-8")
        brief_path.write_text('{"brief": "test"}', encoding="utf-8")
        binding = worker.launch_identity(config, config_path, protocol_path, brief_path)
        review_path.write_text(json.dumps({"decision": "CONTINUE_CONTROLLER", "launch_binding": binding}, sort_keys=True), encoding="utf-8")
        return config, config_path, protocol_path, brief_path, review_path

    def test_exact_governor_review_is_required_and_accepted_without_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            config, config_path, protocol_path, brief_path, review_path = self._authenticated_review_paths(Path(root))
            result = worker.authenticate_governor_review(config, config_path, protocol_path, brief_path, review_path)
            self.assertEqual(result["status"], "GOVERNOR_REVIEW_AUTHENTICATED")
            self.assertFalse((Path(root) / "output").exists())

    def test_noncontinue_malformed_and_identity_mismatch_reviews_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            config, config_path, protocol_path, brief_path, review_path = self._authenticated_review_paths(Path(root))
            with self.assertRaisesRegex(RuntimeError, "GOVERNOR_REVIEW_MISSING"):
                worker.authenticate_governor_review(config, config_path, protocol_path, brief_path, Path(root) / "missing.json")
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review["decision"] = "REQUIRE_CHANGES"
            review_path.write_text(json.dumps(review), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "GOVERNOR_REVIEW_DECISION_NOT_CONTINUE"):
                worker.authenticate_governor_review(config, config_path, protocol_path, brief_path, review_path)
            review_path.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "GOVERNOR_REVIEW_MALFORMED"):
                worker.authenticate_governor_review(config, config_path, protocol_path, brief_path, review_path)
            _, _, _, _, review_path = self._authenticated_review_paths(Path(root))
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review["launch_binding"]["source_commit"] = "0" * 40
            review_path.write_text(json.dumps(review), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "GOVERNOR_REVIEW_BINDING_MISMATCH:source_commit"):
                worker.authenticate_governor_review(config, config_path, protocol_path, brief_path, review_path)


if __name__ == "__main__":
    unittest.main()
