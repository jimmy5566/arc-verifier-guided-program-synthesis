from __future__ import annotations
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_foundation_v2_reconstruction_v1.py"
spec = importlib.util.spec_from_file_location("reconstruction_launcher", WRAPPER)
launcher = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(launcher)


class ReconstructionLaunchGuardTest(unittest.TestCase):
    def test_hold_rejects_before_child_process(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw); binding = d / "binding.json"; preflight = d / "preflight.json"; gate = d / "gate.json"; marker = d / "child-ran"
            binding.write_text(json.dumps({"protocol_id": "FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1"}), encoding="utf-8")
            preflight.write_text(json.dumps({"status": "PASS", "launch_binding_sha256": "x", "binding_path": str(binding)}), encoding="utf-8")
            gate.write_text(json.dumps({"GPU_GATE_READY": True, "AUTO_SCIENTIFIC_EXECUTION_AUTHORIZED": True}), encoding="utf-8")
            run = subprocess.run([sys.executable, str(WRAPPER), "--attempt", "GUARD_TEST", "--max-seconds", "1", "--scientific-gate", str(gate), "--preflight", str(preflight), "--ledger", str(d / "ledger.jsonl"), "--snapshot", str(d / "snapshot.json"), "--interval", str(d / "interval.json"), "--accounting-state", str(d / "accounting.json"), "--", sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"], capture_output=True, text=True, check=False)
            self.assertNotEqual(0, run.returncode)
            self.assertIn("SCIENTIFIC_STATE_CONFLICT_HOLD_ACTIVE", run.stderr)
            self.assertFalse(marker.exists())

    def test_unqualified_or_scope_mismatched_directive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw); directive = d / "directive.json"
            directive.write_text(json.dumps({"decision": "REQUIRE_INFRA_REPAIR", "scientific_training_authorized": False, "scope": "ALL"}), encoding="utf-8")
            gate = {"authorization": {"directive_path": str(directive), "directive_sha256": launcher.sha256(directive), "protocol_id": "FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1", "launch_nonce": "n", "consumed": False}}
            binding = {"protocol_id": "FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1"}
            with self.assertRaisesRegex(RuntimeError, "DIRECTOR_AUTHORIZATION_NOT_QUALIFYING"):
                launcher.validate_authorization(gate, binding, "ROUND_001")
            directive.write_text(json.dumps({"decision": "CONTINUE", "scientific_training_authorized": True, "protocol_id": "FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1", "scope": "WRONG"}), encoding="utf-8")
            gate["authorization"]["directive_sha256"] = launcher.sha256(directive)
            with self.assertRaisesRegex(RuntimeError, "DIRECTOR_AUTHORIZATION_SCOPE_MISMATCH"):
                launcher.validate_authorization(gate, binding, "ROUND_001")

    def test_worker_failure_propagates_and_zero_work_is_not_charged(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw); preflight = d / "preflight.json"; ledger = d / "ledger.jsonl"
            preflight.write_text(json.dumps({"status": "PASS", "launch_binding_sha256": "test"}), encoding="utf-8")
            run = subprocess.run([sys.executable, str(WRAPPER), "--attempt", "EXIT_PROPAGATION", "--max-seconds", "1", "--dummy", "--preflight", str(preflight), "--ledger", str(ledger), "--snapshot", str(d / "snapshot.json"), "--interval", str(d / "interval.json"), "--accounting-state", str(d / "accounting.json"), "--", sys.executable, "-c", "raise SystemExit(7)"], capture_output=True, text=True, check=False)
            self.assertEqual(1, run.returncode)
            records = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(0.0, records[-1]["gpu_optimizer_seconds"])
            self.assertEqual(7, records[-1]["evidence"]["exit_code"])

    def test_worker_accounting_state_charges_only_declared_optimizer_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            p = Path(raw) / "state.json"
            p.write_text(json.dumps({"completed_optimizer_training_seconds": 12.5, "active_optimizer_step_started_monotonic_ns": 1_000_000_000}), encoding="utf-8")
            self.assertEqual(14.5, launcher.worker_gpu_seconds(p, 3_000_000_000))
            self.assertEqual(0.0, launcher.worker_gpu_seconds(Path(raw) / "missing.json", 3_000_000_000))

if __name__ == "__main__":
    unittest.main()
