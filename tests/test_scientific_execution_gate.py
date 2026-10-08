from __future__ import annotations
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_scientific_execution_gate.py"
NAMES = ["frozen_baseline_identity","no_eval60_gold_in_training_path","no_diagnostic_gold_in_training_labels","surface_isolation","baseline_collector_target_blind","model_runtime_identity","training_budget_available","final_audit_unopened","provenance_clean"]
class GateTests(unittest.TestCase):
    def test_pass_requires_every_named_condition(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/"in.json"; output=root/"out.json"
            source.write_text(json.dumps({"conditions":{name:{"status":"PASS"} for name in NAMES}}),encoding="utf-8")
            self.assertEqual(subprocess.run([sys.executable,str(SCRIPT),"--input",str(source),"--output",str(output)]).returncode,0)
            self.assertTrue(json.loads(output.read_text())["AUTO_SCIENTIFIC_EXECUTION_AUTHORIZED"])
    def test_one_blocker_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/"in.json"; output=root/"out.json"
            conditions={name:{"status":"PASS"} for name in NAMES}; conditions["frozen_baseline_identity"]={"status":"BLOCKED"}
            source.write_text(json.dumps({"conditions":conditions}),encoding="utf-8")
            self.assertNotEqual(subprocess.run([sys.executable,str(SCRIPT),"--input",str(source),"--output",str(output)]).returncode,0)
            self.assertFalse(json.loads(output.read_text())["scientific_training_authorized"])
if __name__ == "__main__": unittest.main()
