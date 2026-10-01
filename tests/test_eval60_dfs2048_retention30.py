from __future__ import annotations

import csv
import hashlib
import json
import copy
from pathlib import Path

from scripts.build_eval60_dfs2048_retention30_kaggle import (
    EXPERIMENT_ID,
    REQUIRED_OUTPUTS,
    notebook_source,
)
from scripts.run_eval60_dual_ttt_dfs1024_retention30 import (
    EXPECTED_CELLS,
    unclassified_lifecycle_summary,
)
from scripts.run_eval60_dual_ttt_dfs2048_retention30 import EXPECTED_MAX_EXPANDED_NODES
from scripts import run_eval60_dual_ttt_dfs1024_retention30 as shared_runner
from scripts.run_eval60_dual_ttt_dfs2048_retention30 import configure_shared_runner
from scripts.score_eval60_dfs2048_retention30 import budget_summary


ROOT = Path(__file__).resolve().parents[1]
OLD_EXPERIMENT = ROOT / "experiments" / "eval60_dual_ttt_dfs1024_retention30_v1"
EXPERIMENT = ROOT / "experiments" / "eval60_dual_ttt_dfs2048_retention30_v1"
BASELINE_RUN = ROOT / "artifacts" / "eval60_dual_ttt_dfs1024_retention30_v1_kaggle_output" / "analysis" / "eval60_dual_ttt_dfs1024_retention30_v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_01_exact_retention30_cohort_is_reused_byte_for_byte() -> None:
    for name in ("TARGET_BLIND_RUN_MANIFEST.json", "RETENTION30_OUTPUTS.json", "RETENTION30_COHORT_SHA256.txt", "HISTORICAL_GREEDY_SOURCE_CLASSES.csv"):
        assert (EXPERIMENT / name).read_bytes() == (OLD_EXPERIMENT / name).read_bytes()
    manifest = read_json(EXPERIMENT / "TARGET_BLIND_RUN_MANIFEST.json")
    assert (manifest["retention_output_count"], manifest["unique_task_count"], manifest["expected_logical_cells"]) == (30, 22, 240)
    assert manifest["retention30_output_sha256"] == "7d4767763f8a6d53fadbe64cb38e81e9f8bc6ce345c6934beb520756b84764e3"
    assert sha(EXPERIMENT / "TARGET_BLIND_RUN_MANIFEST.json") == "8069af5dacea3e8b4c97f7fef821e1a867ae2212dd0dec868e3793b3fbcee06f"


def test_02_successful_dfs1024_provenance_is_recorded_from_run_artifacts() -> None:
    provenance = read_json(EXPERIMENT / "PROVENANCE.json")
    assert provenance["dfs1024_source_commit"] == "537224fd35b7bbe2e261e1476d643ffbde588149"
    assert provenance["dfs1024_config_sha256"] == sha(BASELINE_RUN / "retention_config_frozen.json")
    assert provenance["dfs1024_cohort_sha256"] == (EXPERIMENT / "RETENTION30_COHORT_SHA256.txt").read_text(encoding="utf-8").strip()


def test_03_only_scientific_change_is_1024_to_2048() -> None:
    old = read_json(BASELINE_RUN / "retention_config_frozen.json")
    new = read_json(EXPERIMENT / "retention_config.json")
    assert new["search"]["max_expanded_nodes"] == EXPECTED_MAX_EXPANDED_NODES == 2048
    for payload in (old, new):
        for key in (
            "experiment_id", "benchmark_id", "scientific_reference", "authoritative_source_commit",
            "cohort_source_experiment_id", "dfs1024_source_commit", "dfs1024_config_sha256", "dfs1024_cohort_sha256",
        ):
            payload.pop(key, None)
    old["search"]["max_expanded_nodes"] = 2048
    assert old == new


def test_04_scientific_contract_and_timeout_are_frozen() -> None:
    config = read_json(EXPERIMENT / "retention_config.json")
    assert config["search"] == {
        "engine": "authoritative_scalar_d1_regret",
        "policy": "CUMULATIVE_REGRET_r=4.00",
        "max_expanded_nodes": 2048,
        "max_completed_candidates": 32,
        "max_new_tokens": 931,
        "max_score": 1.6094379124341003,
        "frontier_floor": 1,
        "local_time_limit_seconds": 540.0,
        "pad_token_id": 13,
        "arc_tokens": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
        "lane_count": 1,
        "diagnostic_trace": False,
        "batch2_cross_cell": False,
        "batch4_regret": False,
    }
    assert config["rerun"] == {"enabled": False, "auto_rerun": False, "retry_failed_task": False, "resume_completed": False}
    assert [row["geometry"] for row in config["augmentation_manifest"]["TTT24"]] == ["flip_lr", "flip_ud", "transpose", "anti_transpose"]
    assert [row["geometry"] for row in config["augmentation_manifest"]["TTT48"]] == ["identity", "rot90", "flip_ud", "anti_transpose"]


def test_04b_runtime_rejects_any_other_scientific_drift() -> None:
    names = (
        "EXPERIMENT_ID", "EXPECTED_BENCHMARK_ID", "EXPECTED_MAX_EXPANDED_NODES",
        "MANIFEST_EXPERIMENT_ID", "DEFAULT_SOURCE_BRANCH", "EXPECTED_SCIENTIFIC_CONFIG_SHA256",
    )
    old_globals = {name: getattr(shared_runner, name) for name in names}
    try:
        configure_shared_runner()
        config = read_json(EXPERIMENT / "retention_config.json")
        shared_runner.validate_experiment_config(config)
        drifted = copy.deepcopy(config)
        drifted["ttt24_recipe"]["learning_rate"] = 0.0001
        try:
            shared_runner.validate_experiment_config(drifted)
        except ValueError as exc:
            assert "scientific configuration hash mismatch" in str(exc)
        else:
            raise AssertionError("runtime accepted a changed learning rate")
    finally:
        for name, value in old_globals.items():
            setattr(shared_runner, name, value)


def test_05_static_timeout_audit_proves_540_seconds_nonbinding() -> None:
    with (BASELINE_RUN / "dfs_cells.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    min_nodes_per_second = min(float(row["nodes_per_second"]) for row in rows if float(row["nodes_per_second"]) > 0)
    max_cell_wall = max(float(row["cell_wall_s"]) for row in rows)
    projected_from_throughput = 2048 / min_nodes_per_second
    projected_from_observed_wall = 2 * max_cell_wall
    assert max(projected_from_throughput, projected_from_observed_wall) < 540.0
    assert 540.0 - max(projected_from_throughput, projected_from_observed_wall) > 350.0


def test_06_runner_is_thin_shared_implementation_and_target_blind() -> None:
    wrapper = (ROOT / "scripts" / "run_eval60_dual_ttt_dfs2048_retention30.py").read_text(encoding="utf-8")
    shared_runner = (ROOT / "scripts" / "run_eval60_dual_ttt_dfs1024_retention30.py").read_text(encoding="utf-8")
    assert "run_eval60_dual_ttt_dfs1024_retention30 as shared" in wrapper
    assert "EXPECTED_MAX_EXPANDED_NODES = 2048" in wrapper
    assert "--solutions" not in wrapper and "--solutions" not in shared_runner
    assert "candidate_grid" in shared_runner and "grid_sha256" in shared_runner and "candidate_artifact_sha256" in shared_runner
    assert 'experiment_id=str(config["experiment_id"])' in shared_runner
    assert '"benchmark_id": str(config["benchmark_id"])' in shared_runner


def test_07_target_blind_freeze_precedes_gold_scorer() -> None:
    source = notebook_source("owner", "dataset")
    runner = source.index("subprocess.run(runner_command")
    freeze = source.index("TARGET_BLIND_CANDIDATE_FREEZE.json")
    solutions = source.index("arc-agi_evaluation_solutions.json")
    scorer = source.index("subprocess.run(scorer_command")
    assert runner < freeze < solutions < scorer
    runner_line = next(line for line in source.splitlines() if line.startswith("runner_command"))
    assert "SOLUTIONS" not in runner_line
    assert "DFS1024_SOURCE_HIT_OVERLAP.csv" not in runner_line


def test_08_required_artifact_contract_contains_dfs2048_comparison() -> None:
    for name in (
        "dfs_candidates_frozen.json", "dfs1024_vs_dfs2048_retention.csv",
        "dfs1024_vs_dfs2048_summary.json", "DFS2048_RETENTION_REPORT.md",
        "TTT_STATE_COMPARABILITY.json", "TELEMETRY_SUMMARY.json",
    ):
        assert name in REQUIRED_OUTPUTS
    assert EXPECTED_CELLS == 240


def test_09_unclassified_lifecycle_is_split_without_relabeling_idle() -> None:
    intervals = [{"gpu_id": 0, "start_timestamp": 10.0, "end_timestamp": 20.0, "phase": "MODEL_LOAD"}]
    tagged = [
        {"gpu_id": 0, "timestamp": 5.0, "phase": "UNCLASSIFIED"},
        {"gpu_id": 0, "timestamp": 15.0, "phase": "UNCLASSIFIED"},
        {"gpu_id": 0, "timestamp": 25.0, "phase": "UNCLASSIFIED"},
        {"gpu_id": 0, "timestamp": 16.0, "phase": "IDLE"},
    ]
    summary = unclassified_lifecycle_summary(tagged, intervals)
    assert summary["UNCLASSIFIED_PRE_LEDGER"] == 1
    assert summary["UNCLASSIFIED_ACTIVE_WORKER_SPAN"] == 1
    assert summary["UNCLASSIFIED_POST_WORKER"] == 1
    assert summary["UNCLASSIFIED_ACTIVE_WORKER_FRACTION"] == 0.25


def test_10_startup_reporting_separates_barrier_from_model_load() -> None:
    source = (ROOT / "scripts" / "run_eval60_dual_ttt_dfs1024_retention30.py").read_text(encoding="utf-8")
    assert '"STARTUP_TO_BARRIER_WALL_S"' in source
    assert '"MODEL_LOAD_PER_WORKER_S"' in source
    assert '"MODEL_LOAD_MEAN_S"' in source
    assert '"MODEL_LOAD_MAX_S"' in source
    assert '"MODEL_LOAD_LEDGER_GPU_SECONDS"' in source
    assert '"model_load_wall_s":' not in source


def test_11_budget_summary_counts_termination_and_empty_pool_separately() -> None:
    rows = [
        {"budget_exhausted": "True", "search_exhausted": "False", "candidate_cap_reached": "False", "empty_pool": "False", "invalid_candidates": "2", "nodes_expanded": "2048", "completed_candidates": "3", "unique_candidates": "2", "termination_reason": "budget_exhausted"},
        {"budget_exhausted": "False", "search_exhausted": "True", "candidate_cap_reached": "False", "empty_pool": "True", "invalid_candidates": "0", "nodes_expanded": "12", "completed_candidates": "0", "unique_candidates": "0", "termination_reason": "search_exhausted"},
    ]
    summary = budget_summary(rows)
    assert summary["cells_total"] == 2
    assert (summary["budget_exhausted"], summary["search_exhausted"], summary["empty_pool"], summary["invalid_candidates"]) == (1, 1, 1, 2)


def test_12_no_publish_launch_or_rerun_path_in_builder() -> None:
    source = (ROOT / "scripts" / "build_eval60_dfs2048_retention30_kaggle.py").read_text(encoding="utf-8")
    assert "kaggle datasets version" not in source
    assert "kaggle kernels push" not in source
    assert "RERUN_ENABLED=FALSE" in source
    assert '"kaggle_run_started": False' in source
