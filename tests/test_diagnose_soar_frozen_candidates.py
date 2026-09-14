import json
import subprocess
import sys
from pathlib import Path


def test_diagnostic_is_train_only_and_reports_funnel(tmp_path: Path):
    artifact = tmp_path / "soar.json"
    output = tmp_path / "diagnostic.json"
    artifact.write_text(json.dumps({"records": {"task": {"candidate_programs": [{
        "raw_model_output": "```python\\ndef transform(x): return x\\n```",
        "extracted_code": "def transform(x): return x",
        "all_train_exact": False,
        "verification": {"parse_valid": True, "static_safe": True, "train_pair_count": 2, "train_pass_count": 1,
                         "train_execution": [{"process_started": True, "status": "SUCCESS", "grid": [[1]], "ok": True},
                                             {"process_started": True, "status": "SUCCESS", "grid": [[0]], "ok": False}]},
    }]}}}), encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "diagnose_soar_frozen_candidates.py"
    subprocess.run([sys.executable, str(script), "--artifact", str(artifact), "--output", str(output)], check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["funnel"]["parse_valid"] == 1
    assert result["tasks"]["task"]["near_miss"] is True
    assert "No solution path" in result["leakage_audit"]
