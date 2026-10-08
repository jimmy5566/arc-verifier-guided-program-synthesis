from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = ROOT / "orchestration" / "watchdog" / "arc2_watchdog.ps1"


class WatchdogSimpleTests(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("powershell") is None:
            self.skipTest("PowerShell unavailable")

    def test_crash_is_restarted_once_and_runs_are_sequential(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); worker = root / "worker.ps1"; marker = root / "first.marker"; log = root / "launches.log"; state = root / "state.json"
            worker.write_text("param([string]$Marker,[string]$Log)\nAdd-Content -LiteralPath $Log -Value $PID\nif (-not (Test-Path -LiteralPath $Marker)) { New-Item -ItemType File -Path $Marker | Out-Null; exit 1 }\nexit 0\n", encoding="utf-8")
            args = f'-NoProfile -ExecutionPolicy Bypass -File "{worker}" -Marker "{marker}" -Log "{log}"'
            result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(WATCHDOG), "-SupervisorFilePath", "powershell.exe", "-SupervisorArguments", args, "-StatePath", str(state), "-MaxRestarts", "2", "-PollSeconds", "1"], capture_output=True, text=True, check=False, timeout=30)
            self.assertEqual(0, result.returncode, result.stderr)
            saved = json.loads(state.read_text(encoding="utf-8-sig"))
            pids = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual("STOPPED_CLEANLY", saved["status"])
            self.assertEqual(1, saved["restart_count"])
            self.assertEqual(2, len(pids))
            self.assertEqual(2, len(set(pids)))

    def test_terminal_workflow_stops_before_starting_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); workflow = root / "workflow.json"; state = root / "state.json"
            workflow.write_text(json.dumps({"schema_version": 1, "experiment_terminal": True}), encoding="utf-8")
            result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(WATCHDOG), "-SupervisorCommand", "exit 77", "-StatePath", str(state), "-ExperimentTerminalStatePath", str(workflow), "-MaxRestarts", "2"], capture_output=True, text=True, check=False, timeout=30)
            self.assertEqual(0, result.returncode, result.stderr)
            saved = json.loads(state.read_text(encoding="utf-8-sig"))
            self.assertEqual("EXPERIMENT_TERMINAL", saved["status"])
            self.assertIsNone(saved["supervisor_pid"])


if __name__ == "__main__":
    unittest.main()
