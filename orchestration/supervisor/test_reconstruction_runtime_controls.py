from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "scripts" / "preflight_foundation_v2_reconstruction_v1.py"
LAUNCHER = ROOT / "scripts" / "run_foundation_v2_reconstruction_v1.py"


class ReconstructionRuntimeControlsTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        payload = root / "payload.txt"; payload.write_text("frozen", encoding="utf-8")
        import hashlib
        binding = root / "binding.json"
        binding.write_text(json.dumps({"schema_version": 1, "status": "FROZEN_PENDING_PREFLIGHT", "source": {"commit": "dummy"}, "required_files": [{"path": str(payload), "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}], "command": ["dummy"], "output": {"parent": str(root)}, "dataset_contracts": []}), encoding="utf-8")
        preflight = root / "preflight.json"
        self.assertEqual(0, subprocess.run([sys.executable, str(PREFLIGHT), "--binding", str(binding), "--output", str(preflight)], capture_output=True).returncode)
        ledger = root / "ledger.jsonl"; snapshot = root / "snapshot.json"
        ledger.write_text('{"cap_seconds": 28800, "gpu_training_started": false, "record_type": "LEDGER_INITIALIZATION", "schema_version": 1}\n', encoding="utf-8")
        return binding, preflight, ledger, snapshot

    def test_dummy_reservation_rejection_and_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); _, preflight, ledger, snapshot=self.fixture(root)
            rejected=subprocess.run([sys.executable,str(LAUNCHER),"--attempt","too-large","--max-seconds","7200.1","--preflight",str(preflight),"--dummy","--","x"],capture_output=True,text=True)
            self.assertNotEqual(0,rejected.returncode); self.assertIn("RECONSTRUCTION_RESERVATION_EXCEEDED",rejected.stderr)
            ok=subprocess.run([sys.executable,str(LAUNCHER),"--attempt","short","--max-seconds","2","--preflight",str(preflight),"--ledger",str(ledger),"--snapshot",str(snapshot),"--interval",str(root/'i.json'),"--dummy","--",sys.executable,"-c","print('dummy')"],capture_output=True,text=True)
            self.assertEqual(0,ok.returncode,ok.stderr)
            self.assertEqual(2,len(ledger.read_text(encoding="utf-8").splitlines()))
            self.assertGreater(json.loads(snapshot.read_text(encoding="utf-8"))["charged_new_gpu_training_seconds"],0)

    def test_dummy_active_cap_interrupts_and_finalizes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); _, preflight, ledger, snapshot=self.fixture(root)
            final = root/'final.txt'; final_json=root/'final.json'; final_json.write_text(json.dumps([sys.executable,"-c",f"from pathlib import Path; Path(r'{final}').write_text('finalized')"]),encoding='utf-8')
            run=subprocess.run([sys.executable,str(LAUNCHER),"--attempt","cap","--max-seconds","0.1","--preflight",str(preflight),"--ledger",str(ledger),"--snapshot",str(snapshot),"--interval",str(root/'i.json'),"--finalize-command-json",str(final_json),"--dummy","--",sys.executable,"-c","import time; time.sleep(2)"],capture_output=True,text=True)
            self.assertEqual(0,run.returncode,run.stderr); self.assertTrue(final.is_file()); self.assertIn("ACTIVE_CAP_INTERRUPT_CHECKPOINT_FINALIZE_REQUIRED",run.stdout)


if __name__ == "__main__":
    unittest.main()
