from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "experiments" / "l4_dual_ttt_dfs1024_bench_v1" / "kaggle_review"


def read(name: str) -> dict:
    return json.loads((REVIEW / name).read_text(encoding="utf-8"))


def sha256(name: str) -> str:
    return hashlib.sha256((REVIEW / name).read_bytes()).hexdigest()


def test_review_snapshot_is_exactly_the_unlaunched_commit_package() -> None:
    identity = read("SOURCE_IDENTITY.json")
    status = read("REVIEW_STATUS.json")
    package = read("PACKAGE_MANIFEST.json")
    assert identity["source_commit"] == "05584e40f3e81b5206b4f8ecc855592cc480428b"
    assert identity["scientific_base_commit"] == "95827d06b6f99c7c47007a1b1035a11274f8c9a8"
    assert package["source_commit"] == identity["source_commit"]
    assert status == {
        "competition_submissions": 0,
        "gpu_runs_started": 0,
        "kaggle_datasets_pushed": 0,
        "kaggle_notebooks_pushed": 0,
        "status": "READY_FOR_GITHUB_REVIEW_NOT_PUBLISHED_NOT_RUN",
    }
    notebook_name = "arc2-l4-dual-ttt-dfs1024-bench-v1.ipynb"
    assert package["notebook_sha256"] == sha256(notebook_name)
    assert package["kernel_metadata_sha256"] == sha256("kernel-metadata.json")
    assert package["resolved_config_sha256"] == sha256("benchmark_config_resolved.json")


def test_review_notebook_compiles_and_is_target_blind_one_shot() -> None:
    notebook = read("arc2-l4-dual-ttt-dfs1024-bench-v1.ipynb")
    assert len(notebook["cells"]) == 1
    code = "".join(notebook["cells"][0]["source"])
    compile(code, "arc2-l4-dual-ttt-dfs1024-bench-v1.ipynb", "exec")
    lowered = code.lower()
    assert "solutions" not in lowered
    assert "submission.json" not in lowered
    assert "--resume" not in code
    assert "RERUN_ENABLED = FALSE" in code
    assert "run_l4_dual_ttt_dfs1024_benchmark.py" in code


def test_review_config_and_metadata_keep_the_frozen_contract() -> None:
    config = read("benchmark_config_resolved.json")
    metadata = read("kernel-metadata.json")
    assert config["authoritative_source_commit"] == "05584e40f3e81b5206b4f8ecc855592cc480428b"
    assert config["search"]["max_expanded_nodes"] == 1024
    assert config["search"]["lane_count"] == 1
    assert config["search"]["batch2_cross_cell"] is False
    assert config["search"]["batch4_regret"] is False
    assert all(value is False for value in config["rerun"].values())
    assert metadata["machine_shape"] == "NvidiaL4"
    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is True
    assert metadata["is_private"] is True


def test_source_package_manifest_contains_no_answer_artifacts() -> None:
    manifest = read("SOURCE_PACKAGE_MANIFEST.json")
    assert manifest["contains_solution_artifacts"] is False
    forbidden_suffixes = ("solutions.json", "answers.json", "gold.json", "predictions.json")
    assert not any(name.lower().endswith(forbidden_suffixes) for name in manifest["files"])
