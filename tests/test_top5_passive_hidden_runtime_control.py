"""CPU-only guards for the Top5 passive hidden-state runtime protocol.

These tests deliberately exercise orchestration evidence only.  They do not
construct a model, score a target, or change the frozen search implementation.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import pytest

from scripts import run_top5_passive_hidden_state_micro12_v1 as top5
from scripts.build_search_order_p3_lds_micro12_cohort import sha_file
from scripts.run_ttt24_aug8_r1024_core_v1 import _atomic_json, _verify


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_gzip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("{}\n")


def _frozen_output(root: Path, output_id: str, *, gold_loaded: bool = False) -> None:
    raw, checkpoints, eos, receipt, ledger, verification, frontier_file, frontier_summary = top5.frontier.output_paths(root, output_id)
    hidden, metadata, summary = top5.hidden_paths(root, output_id)
    _write_json(raw, {"status": "COMPLETE", "gold_loaded": gold_loaded})
    _write_json(checkpoints, {})
    _write_gzip(eos)
    _write_json(receipt, {"status": "COMPLETE"})
    _write_gzip(frontier_file)
    _write_json(frontier_summary, {})
    hidden.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        hidden,
        hidden=np.zeros((1, 3, 2560), dtype=np.float16),
        layers_0_based=np.asarray([11, 23, 35], dtype=np.int16),
        work_item_id=np.asarray([1], dtype=np.int64),
        expanded_nodes_at_capture=np.asarray([1], dtype=np.int64),
    )
    _write_gzip(metadata)
    _write_json(summary, {
        "status": "PASS", "gold_loaded": False, "extra_model_forwards": 0,
        "integrity": {
            "npz_metadata_alignment": "PASS", "unique_cell_work_item": "PASS",
            "retained_lineage_join": "PASS", "nll_rank_1_to_5": "PASS",
        },
    })
    files = (raw, checkpoints, eos, receipt, frontier_file, frontier_summary, hidden, metadata, summary)
    _atomic_json(ledger, {"files": {str(path.relative_to(root)): sha_file(path) for path in files}})
    _atomic_json(verification, _verify(root, ledger))


def _cohort() -> list[dict[str, str]]:
    smoke = [
        {"output_id": "1818057f:o0", "profile": "PROFILE_S"},
        {"output_id": "80a900e0:o0", "profile": "PROFILE_M"},
        {"output_id": "36a08778:o1", "profile": "PROFILE_L_LOW"},
    ]
    return smoke + [{"output_id": f"other{i}:o0", "profile": "PROFILE_S"} for i in range(9)]


def test_verified_smoke_copy_becomes_an_official_output_without_a_worker(tmp_path: Path) -> None:
    source = tmp_path / "gate1" / "GATE1_SMOKE"
    destination = tmp_path / "gate2"
    _frozen_output(source, "1818057f:o0")
    top5.copy_frozen_output(source, destination, "1818057f:o0")
    assert top5.output_is_frozen(destination, "1818057f:o0")
    assert not (destination / "WORKER_LOGS").exists()


def test_incomplete_or_gold_loaded_smoke_output_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _frozen_output(source, "1818057f:o0")
    top5.hidden_paths(source, "1818057f:o0")[0].unlink()
    with pytest.raises(RuntimeError, match="UNVERIFIED_PARTIAL_OUTPUT_REFUSED"):
        top5.copy_frozen_output(source, tmp_path / "destination", "1818057f:o0")
    gold = tmp_path / "gold"
    _frozen_output(gold, "80a900e0:o0", gold_loaded=True)
    with pytest.raises(RuntimeError, match="FROZEN_OUTPUT_INVALID"):
        top5.output_is_frozen(gold, "80a900e0:o0")


def test_gate2_schedules_exactly_nine_non_smoke_outputs() -> None:
    cohort = _cohort()
    remaining = top5.select_remaining_outputs(cohort, cohort[:3])
    assert len(remaining) == 9
    assert {row["output_id"] for row in remaining}.isdisjoint(top5.SMOKE_OUTPUT_IDS)
    with pytest.raises(RuntimeError, match="SMOKE_COHORT_IDENTITY"):
        top5.select_remaining_outputs(cohort, cohort[:2] + [cohort[3]])


def test_smoke_deadline_and_new_output_cutoff_are_hard(tmp_path: Path) -> None:
    clock = {"GPU_EXPERIMENT_START": time.monotonic() - top5.SMOKE_DEADLINE_SECONDS - 1.0}
    with pytest.raises(RuntimeError, match="TIME_BUDGET_SMOKE_TOO_SLOW"):
        top5.require_smoke_deadline(tmp_path, clock)
    assert json.loads((tmp_path / "TIME_BUDGET_STATUS.json").read_text())["classification"] == "TIME_BUDGET_SMOKE_TOO_SLOW"
    cutoff_clock = {"GPU_EXPERIMENT_START": 1.0}
    assert top5.new_output_allowed(cutoff_clock, now=top5.NEW_OUTPUT_CUTOFF_SECONDS - 0.1)
    assert not top5.new_output_allowed(cutoff_clock, now=top5.NEW_OUTPUT_CUTOFF_SECONDS + 1.0)


def test_projection_gate_and_time_budget_receipt_preserve_frozen_evidence(tmp_path: Path) -> None:
    cohort = _cohort()
    historical = [
        {"output_id": row["output_id"], "profile": row["profile"], "wall_seconds": "100"}
        for row in cohort
    ]
    smoke = [
        {"output_id": row["output_id"], "profile": row["profile"], "wall_seconds": "110"}
        for row in cohort[:3]
    ]
    projection = top5.build_projection(
        elapsed_seconds=400.0, historical_rows=historical, smoke_rows=smoke, remaining_outputs=cohort[3:],
    )
    assert projection["status"] == "PASS"
    assert projection["estimated_remaining_generation_seconds"] == pytest.approx(990.0)
    too_slow = top5.build_projection(
        elapsed_seconds=4000.0, historical_rows=historical,
        smoke_rows=[{**row, "wall_seconds": "400"} for row in smoke], remaining_outputs=cohort[3:],
    )
    assert too_slow["status"] == "FAIL"
    preserved = tmp_path / "already_frozen.json"
    preserved.write_text("immutable", encoding="utf-8")
    top5.write_time_budget_status(
        tmp_path, "TIME_BUDGET_INCOMPLETE", {"GPU_EXPERIMENT_START": time.monotonic() - 1.0}, completed_outputs=3,
    )
    assert preserved.read_text(encoding="utf-8") == "immutable"
