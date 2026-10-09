from __future__ import annotations

import hashlib
import importlib.util
import json
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


if __name__ == "__main__":
    unittest.main()
