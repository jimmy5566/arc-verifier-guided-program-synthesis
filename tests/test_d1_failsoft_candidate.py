"""Static/CPU audit of the unpublished fail-soft V2 candidate."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "governance" / "releases" / "d1-failsoft-v2-candidate"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_identity_and_archive_are_self_consistent() -> None:
    candidate = json.loads((CANDIDATE / "CANDIDATE_MANIFEST.json").read_text(encoding="utf-8"))
    source = json.loads((CANDIDATE / "dataset" / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    archive_path = CANDIDATE / "dataset" / source["archive"]
    assert candidate["release_id"] == "d1-failsoft-v2-candidate"
    assert candidate["scientific_algorithm_changed"] is False
    assert candidate["runtime_resilience_changed"] is True
    assert candidate["kaggle_published"] is False and candidate["competition_submitted"] is False
    assert sha256(archive_path) == source["archive_sha256"] == candidate["source_archive_sha256"]
    assert sha256(CANDIDATE / "dataset" / source["config"]) == source["config_sha256"] == candidate["config_sha256"]
    with tarfile.open(archive_path) as archive:
        members = {item.name: hashlib.sha256(archive.extractfile(item).read()).hexdigest() for item in archive.getmembers() if item.isfile()}
    assert members == source["file_sha256"]
    assert {"scripts/run_d1_failsoft_4gpu.py", "scripts/build_d1_failsoft_submission.py", "src/inference/d1_failsoft_runtime.py"}.issubset(members)
    assert not any(token in name.lower() for name in members for token in ("solution", "credential", "kaggle.json", ".safetensors", ".bin", ".pt", ".pth"))


def test_candidate_notebook_is_real_shared_failsoft_release_not_a_smoke() -> None:
    notebook = json.loads((CANDIDATE / "notebook" / "arc2-d1-failsoft-v2-candidate.ipynb").read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][0]["source"])
    compile(code, "arc2-d1-failsoft-v2-candidate", "exec")
    assert "run_d1_failsoft_4gpu.py" in code
    assert "build_d1_failsoft_submission.py" in code
    assert "arc-agi_test_challenges.json" in code
    assert "smoke_task_ids" not in code and "evaluation_challenges" not in code
    assert "FAST_SAVE" in code and "FULL_RERUN" in code
    assert "competitions submit" not in code and "kernels push" not in code
    full_body = code.split("def full_rerun", 1)[1].split("def main", 1)[0]
    assert "[[0]]" not in full_body
    assert "submission.unlink(missing_ok=True)" in full_body


def test_candidate_config_retains_exact_science_and_adds_no_runtime_science() -> None:
    candidate = json.loads((CANDIDATE / "dataset" / "d1_failsoft_release_config.json").read_text(encoding="utf-8"))
    governance = json.loads((ROOT / "governance" / "configs" / "fixed4plus4_d1_example.json").read_text(encoding="utf-8"))["algorithm"]["release_config"]
    assert candidate == governance
    assert candidate["generation"]["portfolio"] == {
        "TTT24": ["flip_lr", "flip_ud", "transpose", "anti_transpose"],
        "TTT48": ["identity", "rot90", "flip_ud", "anti_transpose"],
    }
    assert candidate["ttt24_recipe"]["ttt_steps"] == 24
    assert candidate["ttt48_recipe"]["ttt_steps"] == 48
    assert candidate["scoring"]["selector"] == "D1_L_RRF_EXACT_B_RRF_TIE_BREAK"

