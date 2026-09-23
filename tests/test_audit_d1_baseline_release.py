from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_d1_baseline_release import _cross_score_anchor_state, _production_static_audit


def test_cross_score_anchor_audit_detects_logged_mismatch(tmp_path: Path) -> None:
    log = tmp_path / "log.json"
    log.write_text(json.dumps([{"data": "RuntimeError: TTT48 anchor mismatch for abcdef12: 0.001 > 0.0001"}]), encoding="utf-8")
    result = _cross_score_anchor_state(log)
    assert result["status"] == "FAILED_NUMERICAL_PARITY"
    assert result["anchor_mismatch_task_ids"] == ["abcdef12"]


def test_static_production_audit_identifies_only_the_unbound_live_worker() -> None:
    result = _production_static_audit()
    assert result["status"] == "FAIL"
    assert result["issues"] == ["LIVE_D1_WORKER_BOOTSTRAP_UNBOUND: CPU route is verified, but the exact CUDA TTT24/48 worker has not been parity-bound"]
