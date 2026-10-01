from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_eval60_retention30_kaggle import EXPERIMENT_ID, notebook_source
from scripts.run_eval60_dual_ttt_dfs1024_retention30 import (
    EXPECTED_CELLS,
    EXPECTED_OUTPUTS,
    EXPECTED_VIEWS,
    _deduplicated_candidates,
    build_union,
    candidate_freeze,
    grid_key,
    persist_cell,
    tag_telemetry,
    validate_manifest,
    value_sha256,
)
from scripts.score_eval60_retention30 import score, verify_target_blind_freeze


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "eval60_dual_ttt_dfs1024_retention30_v1"


def load(name: str) -> dict:
    return json.loads((EXPERIMENT / name).read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_01_retention_cohort_is_exact_historical_oracle30() -> None:
    cohort = load("RETENTION30_OUTPUTS.json")
    assert cohort["historical_replay"] == {"denominator": 89, "pool_oracle": 30, "top1": 20, "top2": 28}
    assert len(cohort["outputs"]) == EXPECTED_OUTPUTS
    assert all(row["historical_full_pool_hit"] is True for row in cohort["outputs"])
    assert len({row["output_id"] for row in cohort["outputs"]}) == EXPECTED_OUTPUTS


def test_02_historical_source_classes_partition_30() -> None:
    cohort = load("RETENTION30_OUTPUTS.json")
    assert cohort["historical_source_counts"] == {"BOTH": 19, "TTT24_ONLY": 5, "TTT48_ONLY": 6}
    assert sum(cohort["historical_source_counts"].values()) == 30
    assert sum(row["historical_full_top2_hit"] for row in cohort["outputs"]) == 28


def test_03_gpu_manifest_has_no_solution_or_gold_grid() -> None:
    manifest = load("TARGET_BLIND_RUN_MANIFEST.json")
    serialized = json.dumps(manifest).lower()
    assert manifest["solutions_accessed"] is False
    assert "solution" not in serialized.replace('"solutions_accessed": false', "")
    assert "target_grid" not in serialized and "gold_grid" not in serialized
    runner = (ROOT / "scripts" / "run_eval60_dual_ttt_dfs1024_retention30.py").read_text(encoding="utf-8")
    assert "--solutions" not in runner


def test_04_views_and_representations_are_frozen() -> None:
    config = load("retention_config.json")
    assert EXPECTED_VIEWS["TTT24"] == ("flip_lr", "flip_ud", "transpose", "anti_transpose")
    assert EXPECTED_VIEWS["TTT48"] == ("identity", "rot90", "flip_ud", "anti_transpose")
    for source, expected in EXPECTED_VIEWS.items():
        rows = config["augmentation_manifest"][source]
        assert tuple(row["geometry"] for row in rows) == expected
        assert all(row["color_offset"] == 0 and row["pair_order"] == "canonical" for row in rows)


def test_05_independent_ttt_trajectories_reset_before_each_depth() -> None:
    source = (ROOT / "scripts" / "run_eval60_dual_ttt_dfs1024_retention30.py").read_text(encoding="utf-8")
    loop = source.index('for source, depth in (("TTT24", 24), ("TTT48", 48))')
    reset = source.index("set_peft_model_state_dict", loop)
    fit = source.index("_fit_task(", loop)
    assert loop < reset < fit
    assert "continue training" not in source


def test_06_scalar_regret_dfs1024_science_is_unchanged() -> None:
    config = load("retention_config.json")
    search = config["search"]
    assert search["policy"] == "CUMULATIVE_REGRET_r=4.00"
    assert search["max_expanded_nodes"] == 1024
    assert search["max_completed_candidates"] == 32
    assert search["max_new_tokens"] == 931
    assert search["lane_count"] == 1


def test_07_exact_240_logical_cells_and_manifest_hash() -> None:
    manifest = load("TARGET_BLIND_RUN_MANIFEST.json")
    assert manifest["retention_output_count"] == 30
    assert manifest["expected_logical_cells"] == EXPECTED_CELLS
    assert sum(len(indices) for indices in manifest["outputs_by_task"].values()) == 30
    assert value_sha256(manifest["output_ids"]) == manifest["retention30_output_sha256"]


def test_08_actual_grid_candidates_are_deduplicated_and_persisted(tmp_path: Path) -> None:
    base = {"valid_grid": True, "canonical_candidate": [[1, 2], [3, 4]], "candidate_token_ids": [1, 2], "cumulative_nll": 2.0, "candidate_discovery_order": 0, "candidate_discovery_timestamp_unix": 1.0, "terminal_node_id": 7, "forward_count_at_discovery": 3}
    row = {
        "task_id": "task", "output_index": 0, "depth": 24, "view": "flip_lr",
        "decoder_policy": "CUMULATIVE_REGRET_r=4.00", "prompt_tokens": 10, "prompt_sha256": "a",
        "checkpoint_sha256": "b", "candidate_count": 2, "valid_grid_count": 2,
        "candidates": [base, {**base, "cumulative_nll": 1.0, "candidate_discovery_order": 1}],
        "runtime_seconds": 1.0, "nodes_expanded": 5, "model_forwards": 2, "tokens_advanced": 3,
        "complete_candidate_count": 2, "unique_grid_count": 1, "mean_frontier_size": 1.0,
        "max_frontier_size": 2, "model_forward_seconds": 0.5, "peak_vram_mb": 100,
        "termination_reason": "candidate_budget", "budget_exhausted": False, "search_exhausted": False,
    }
    candidates = _deduplicated_candidates(row)
    assert len(candidates) == 1 and candidates[0]["support_count_in_cell"] == 2
    assert candidates[0]["candidate_grid"] == [[1, 2], [3, 4]]
    assert candidates[0]["first_discovery_node"] == "NOT_MEASURED"
    compact, path = persist_cell(output=tmp_path, row=row, source="TTT24", depth=24, worker_id=0)
    assert path.is_file() and compact["candidate_artifact_sha256"] == sha(path)
    assert json.loads(path.read_text())["candidates"][0]["candidate_grid"] == [[1, 2], [3, 4]]


def test_09_source_union_deduplicates_by_canonical_grid(tmp_path: Path) -> None:
    paths = []
    cells = []
    for source, view, grid in (("TTT24", "a", [[1]]), ("TTT24", "b", [[1]]), ("TTT48", "c", [[2]])):
        path = tmp_path / f"{source}-{view}.json"
        write_json(path, {"candidates": [{"grid_key": grid_key(grid), "candidate_grid": grid, "support_count_in_cell": 1, "cell_rank": 0, "cumulative_nll": 1.0, "first_discovery_node": "NOT_MEASURED"}]})
        paths.append(path)
        cells.append({"output_id": "x:o0", "source": source, "view": view, "ttt_depth": int(source[-2:]), "candidate_artifact": str(path), "candidate_artifact_sha256": sha(path)})
    union = build_union(cells)
    assert union["x:o0"]["candidate_count"] == 2
    one = next(row for row in union["x:o0"]["candidates"] if row["grid"] == [[1]])
    assert one["support_count"] == 2 and len(one["source_rows"]) == 2


def _make_fake_freeze(tmp_path: Path) -> tuple[Path, Path, Path]:
    output = tmp_path / "run"
    cells_dir = output / "target_blind" / "cells"
    cells_dir.mkdir(parents=True)
    output_ids = [f"t{o}:o0" for o in range(30)]
    manifest = {"experiment_id": EXPERIMENT_ID, "solutions_accessed": False, "retention_output_count": 30, "unique_task_count": 30, "expected_logical_cells": 240, "output_ids": output_ids, "retention30_output_sha256": value_sha256(output_ids), "task_ids": [f"t{o}" for o in range(30)], "outputs_by_task": {f"t{o}": [0] for o in range(30)}}
    write_json(output / "TARGET_BLIND_RUN_MANIFEST.json", manifest)
    retention_rows = []
    solutions = {}
    cell_hashes = {}
    source24 = {}
    source48 = {}
    full = {}
    all_cells = []
    for o, output_id in enumerate(output_ids):
        task = f"t{o}"
        target = [[o % 10]]
        solutions[task] = [target]
        hist = "BOTH" if o < 19 else "TTT24_ONLY" if o < 24 else "TTT48_ONLY"
        retention_rows.append({"task_id": task, "output_index": 0, "output_id": output_id, "historical_full_pool_hit": True, "historical_full_top1_hit": o < 20, "historical_full_top2_hit": o < 28, "historical_ttt24_pool_hit": hist in {"BOTH", "TTT24_ONLY"}, "historical_ttt48_pool_hit": hist in {"BOTH", "TTT48_ONLY"}, "historical_source_class": hist})
        # Deterministic fake post-search truth table: both, 24-only, 48-only, neither.
        class_after = ("BOTH", "TTT24_ONLY", "TTT48_ONLY", "NEITHER")[o % 4]
        pools = {}
        for source in ("TTT24", "TTT48"):
            source_has = class_after == "BOTH" or class_after == f"{source}_ONLY"
            candidate_grid = target if source_has else [[(o + 1) % 10]]
            key = grid_key(candidate_grid)
            pools[source] = {"cell_ids": [], "candidate_count": 1, "candidates": [{"grid_key": key, "grid": candidate_grid, "source_rows": [], "support_count": 4}]}
            for view in EXPECTED_VIEWS[source]:
                name = f"{task}__o0__d{int(source[-2:])}__{view}.json"
                path = cells_dir / name
                cell = {"experiment_id": EXPERIMENT_ID, "task_id": task, "output_index": 0, "output_id": output_id, "source": source, "ttt_depth": int(source[-2:]), "view": view, "candidates": [{"grid_key": key, "candidate_grid": candidate_grid, "support_count_in_cell": 1, "cell_rank": 0, "cumulative_nll": 1.0, "first_discovery_node": "NOT_MEASURED"}], "solutions_accessed": False}
                write_json(path, cell)
                cell_hashes[name] = sha(path)
                all_cells.append(cell)
                pools[source]["cell_ids"].append(f"{source}:{view}")
        source24[output_id], source48[output_id] = pools["TTT24"], pools["TTT48"]
        candidates = {row["grid_key"]: row for row in [*pools["TTT24"]["candidates"], *pools["TTT48"]["candidates"]]}
        full[output_id] = {"cell_ids": [*pools["TTT24"]["cell_ids"], *pools["TTT48"]["cell_ids"]], "candidate_count": len(candidates), "candidates": list(candidates.values())}
    retention = {"experiment_id": EXPERIMENT_ID, "retention30_output_sha256": manifest["retention30_output_sha256"], "outputs": retention_rows}
    retention_path = tmp_path / "retention.json"
    solutions_path = tmp_path / "solutions.json"
    write_json(retention_path, retention)
    write_json(solutions_path, solutions)
    aggregates = {"dfs_candidates_frozen.json": all_cells, "ttt24_union_candidates_frozen.json": source24, "ttt48_union_candidates_frozen.json": source48, "dual_union_candidates_frozen.json": full}
    aggregate_hashes = {}
    for name, payload in aggregates.items():
        write_json(output / name, payload)
        aggregate_hashes[name] = sha(output / name)
    freeze = {"experiment_id": EXPERIMENT_ID, "status": "TARGET_BLIND_CANDIDATES_FROZEN", "solutions_accessed": False, "retention30_output_sha256": manifest["retention30_output_sha256"], "output_count": 30, "logical_cell_count": 240, "candidate_cell_artifacts": cell_hashes, "aggregate_artifacts": aggregate_hashes}
    write_json(output / "TARGET_BLIND_CANDIDATE_FREEZE.json", freeze)
    return output, retention_path, solutions_path


def test_10_gold_hit_truth_table_and_post_freeze_scoring(tmp_path: Path) -> None:
    output, retention, solutions = _make_fake_freeze(tmp_path)
    result = score(output, solutions, retention)
    summary = result["summary"]
    assert (summary["both_hits"], summary["ttt24_only_hits"], summary["ttt48_only_hits"], summary["neither_hits"]) == (8, 8, 7, 7)
    assert sum(summary[key] for key in ("both_hits", "ttt24_only_hits", "ttt48_only_hits", "neither_hits")) == 30
    assert result["decision"]["selector_claim"] == "NONE"


def test_11_gold_scoring_rejects_mutated_candidate_before_opening_solutions(tmp_path: Path) -> None:
    output, retention, solutions = _make_fake_freeze(tmp_path)
    victim = next((output / "target_blind" / "cells").glob("*.json"))
    victim.write_text("{}\n", encoding="utf-8")
    solutions.unlink()  # Proves failure precedes any attempted Gold read.
    with pytest.raises(RuntimeError, match="changed after freeze"):
        score(output, solutions, retention)


def test_12_phase_ledger_defines_all_required_training_and_dfs_phases() -> None:
    source = (ROOT / "scripts" / "run_eval60_dual_ttt_dfs1024_retention30.py").read_text(encoding="utf-8")
    for phase in ("MODEL_LOAD", "TTT24_TRAIN", "TTT24_DFS", "RESET_TRANSITION", "TTT48_TRAIN", "TTT48_DFS", "SERIALIZE", "IDLE"):
        assert f'"{phase}"' in source


def test_13_telemetry_join_is_deterministic_and_overlap_fails() -> None:
    rows = [{"timestamp": 2.0, "gpu_id": 0, "gpu_util_percent": 50}]
    intervals = [{"gpu_id": 0, "worker_id": 0, "phase": "TTT24_DFS", "task_id": "x", "output_index": 0, "source": "TTT24", "view": "flip_lr", "start_timestamp": 1.0, "end_timestamp": 3.0}]
    assert tag_telemetry(rows, intervals)[0]["phase"] == "TTT24_DFS"
    with pytest.raises(RuntimeError, match="overlapping"):
        tag_telemetry(rows, [*intervals, {**intervals[0], "phase": "IDLE"}])


def test_14_rerun_retry_resume_are_all_disabled() -> None:
    assert load("retention_config.json")["rerun"] == {"enabled": False, "auto_rerun": False, "retry_failed_task": False, "resume_completed": False}


def test_15_batch2_is_disabled() -> None:
    assert load("retention_config.json")["search"]["batch2_cross_cell"] is False


def test_16_batch4_is_disabled() -> None:
    assert load("retention_config.json")["search"]["batch4_regret"] is False


def test_17_manifest_validation_uses_selected_outputs_not_all_task_outputs() -> None:
    manifest = load("TARGET_BLIND_RUN_MANIFEST.json")
    challenges = {task_id: {"train": [], "test": [{"input": [[0]]} for _ in range(max(indices) + 1)]} for task_id, indices in manifest["outputs_by_task"].items()}
    result = validate_manifest(manifest, challenges)
    assert result == {"task_ids": manifest["task_ids"], "num_tasks": 22, "num_test_outputs": 30, "num_dfs_cells": 240}


def test_18_notebook_separates_target_blind_runner_from_gold_scorer() -> None:
    source = notebook_source("owner", "dataset")
    assert source.index("subprocess.run(runner_command") < source.index("TARGET_BLIND_CANDIDATE_FREEZE.json") < source.index("subprocess.run(scorer_command")
    runner_line = next(line for line in source.splitlines() if line.startswith("runner_command"))
    assert "SOLUTIONS" not in runner_line
    assert '"--solutions"' in source and '"--retention"' in source
    assert "kaggle datasets version" not in source and "kaggle kernels push" not in source
