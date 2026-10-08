from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "prepare_base_only_remote_first_data_v2.py"


class RemoteFirstDatasetTests(unittest.TestCase):
    def test_cpu_generator_is_deterministic_and_has_no_historical_replay_identity(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            first, second = Path(root) / "first", Path(root) / "second"
            for output in (first, second):
                subprocess.run([sys.executable, str(GENERATOR), "--output", str(output)], cwd=ROOT, check=True)
            a = json.loads((first / "REMOTE_DATA_IDENTITY_V1.json").read_text(encoding="utf-8"))
            b = json.loads((second / "REMOTE_DATA_IDENTITY_V1.json").read_text(encoding="utf-8"))
            self.assertEqual(a["files"], b["files"])
            self.assertEqual(a["files"]["TRAIN.jsonl"]["records"], 4004)
            self.assertIn("RETENTION_TRAIN", {json.loads(line)["role"] for line in (first / "TRAIN.jsonl").read_text(encoding="utf-8").splitlines()})
            self.assertIn("historical replay-00000.parquet", a["prohibitions"])

