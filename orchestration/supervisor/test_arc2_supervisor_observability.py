from __future__ import annotations

import base64
import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from orchestration.supervisor import arc2_supervisor as supervisor_module


class SupervisorObservabilityTest(unittest.TestCase):
    def _receipt(self, root: Path, round_id: str) -> None:
        path = root / f"ROUND_{round_id}" / f"ROUND_{round_id}_TERMINAL_RECEIPT.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"round_id": round_id, "status": "SUCCESS"}), encoding="utf-8")

    def test_five_minute_interval_and_timestamped_heartbeat(self) -> None:
        self.assertEqual(300, supervisor_module.LOW_FREQUENCY_RECONCILIATION_SECONDS)
        root = Path(__file__).resolve().parents[2]
        protocol = json.loads((root / "orchestration" / "supervisor" / "SUPERVISOR_PROTOCOL.json").read_text(encoding="utf-8"))
        self.assertEqual(300, protocol["low_frequency_reconciliation_seconds"])
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            cycle = base / "DIRECTIVE_CYCLE_STATE.json"
            cycle.write_text(json.dumps({"directives": {"current": {"round_id": "ROUND_009", "state": "WAITING_REMOTE_JOB"}}}), encoding="utf-8")
            service = supervisor_module.Supervisor(base / "state.json", base / "notifications")
            snapshot = supervisor_module.heartbeat_snapshot(service, cycle, "LOW_FREQUENCY_RECONCILIATION_OK", "2026-10-08T18:35:00+10:30")
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                supervisor_module.print_heartbeat(snapshot, datetime(2026, 10, 8, 18, 35, tzinfo=timezone.utc))
            line = stream.getvalue().strip()
            self.assertTrue(line.startswith("[2026-10-08 18:35:00] HEARTBEAT "))
            self.assertIn("round_id=ROUND_009", line)
            self.assertIn("optimizer_step=null", line)
            self.assertEqual(service.state, supervisor_module.default_state())

    def test_terminal_wakeup_once_heartbeat_never_wakes_and_restart_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            receipts, state, notes, acknowledgements = base / "receipts", base / "state.json", base / "notes", base / "acks"
            self._receipt(receipts, "DUMMY_TIMING_EVENT")
            service = supervisor_module.Supervisor(state, notes)
            notified = service.reconcile(receipts)
            with patch.object(supervisor_module, "prompt_controller", return_value=(0, "")) as prompt:
                self.assertEqual(["DUMMY_TIMING_EVENT"], service.wake_controller_once(notified, "arc-controller", acknowledgements, 1))
                self.assertEqual([], service.wake_controller_once(notified, "arc-controller", acknowledgements, 1))
                supervisor_module.print_heartbeat(supervisor_module.heartbeat_snapshot(service, None, "LOCAL_ONLY", None))
                self.assertEqual(1, prompt.call_count)
            service.state["last_heartbeat"] = {"remote_status": "LOW_FREQUENCY_RECONCILIATION_OK"}
            service.save()
            restarted = supervisor_module.Supervisor(state, notes)
            self.assertEqual("LOW_FREQUENCY_RECONCILIATION_OK", restarted.state["last_heartbeat"]["remote_status"])
            self.assertEqual([], restarted.reconcile(receipts))
            self.assertEqual(1, len(list(notes.glob("*.json"))))


    def test_model_named_scientific_receipt_fetches_and_wakes_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            receipt = {
                "round_id": "RECONSTRUCTED_FOUNDATION_V2_V2_009",
                "round_class": "SCIENTIFIC",
                "protocol_id": "FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V2",
                "status": "SUCCESS",
                "scientific_training_started": True,
                "worker_exit_code": 0,
                "optimizer_steps": 172,
                "processed_tokens": 2017545,
                "scientific_gpu_training_seconds": 3739.0,
                "ledger_snapshot_sha256": "a" * 64,
                "training_result_sha256": "b" * 64,
                "checkpoint_identities": [],
                "final_audit_accessed": False,
            }
            remote_path = "/workspace/arc2/orchestration/rounds/ROUND_RECONSTRUCTED_FOUNDATION_V2_V2_009/RECONSTRUCTED_FOUNDATION_V2_V2_009_TERMINAL_RECEIPT.json"
            payload = json.dumps({"path": remote_path, "content_b64": base64.b64encode(json.dumps(receipt).encode("utf-8")).decode("ascii")})
            with patch.object(supervisor_module, "remote_shell", return_value=payload):
                cache = supervisor_module.fetch_remote_receipts("unused", "/workspace/arc2/orchestration", base / "cache")
            cached = cache / "ROUND_RECONSTRUCTED_FOUNDATION_V2_V2_009" / "RECONSTRUCTED_FOUNDATION_V2_V2_009_TERMINAL_RECEIPT.json"
            self.assertTrue(cached.is_file())
            service = supervisor_module.Supervisor(base / "state.json", base / "notes")
            notified = service.reconcile(cache)
            self.assertEqual(["RECONSTRUCTED_FOUNDATION_V2_V2_009"], notified)
            with patch.object(supervisor_module, "prompt_controller", return_value=(0, "")) as prompt:
                self.assertEqual(["RECONSTRUCTED_FOUNDATION_V2_V2_009"], service.wake_controller_once(notified, "arc-controller", base / "acks", 1))
                self.assertEqual([], service.wake_controller_once(notified, "arc-controller", base / "acks", 1))
                self.assertEqual(1, prompt.call_count)


if __name__ == "__main__":
    unittest.main()
