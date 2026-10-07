from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "orchestration" / "supervisor" / "arc2_supervisor.py"
SPEC = importlib.util.spec_from_file_location("arc2_supervisor", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def receipt(root: Path, round_id: str, status: str = "SUCCESS") -> Path:
    path = root / f"ROUND_{round_id}" / f"ROUND_{round_id}_TERMINAL_RECEIPT.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"round_id": round_id, "status": status, "scientific_acceptance": "NOT_EVALUATED"}), encoding="utf-8")
    return path


def test_reconcile_is_idempotent_across_restart(tmp_path: Path) -> None:
    root, state, notifications = tmp_path / "receipts", tmp_path / "state.json", tmp_path / "notifications"
    receipt(root, "001")
    first = MODULE.Supervisor(state, notifications)
    assert first.reconcile(root) == ["001"]
    assert len(list(notifications.glob("*.json"))) == 1
    restarted = MODULE.Supervisor(state, notifications)
    assert restarted.reconcile(root) == []
    assert len(list(notifications.glob("*.json"))) == 1
    assert restarted.state["rounds"]["001"]["controller_notification_sent"] is True


def test_non_success_process_outcome_not_scientific_success(tmp_path: Path) -> None:
    root, state, notifications = tmp_path / "receipts", tmp_path / "state.json", tmp_path / "notifications"
    receipt(root, "002", "TRAIN_FAILED")
    supervisor = MODULE.Supervisor(state, notifications)
    assert supervisor.reconcile(root) == ["002"]
    note = json.loads(next(notifications.glob("*.json")).read_text(encoding="utf-8"))
    assert note["terminal_status"] == "TRAIN_FAILED"
    assert note["scientific_acceptance"] == "NOT_INFERRED_FROM_PROCESS_RECEIPT"


def test_receipt_hash_conflict_requires_director_review(tmp_path: Path) -> None:
    root, state, notifications = tmp_path / "receipts", tmp_path / "state.json", tmp_path / "notifications"
    path = receipt(root, "003")
    supervisor = MODULE.Supervisor(state, notifications)
    supervisor.reconcile(root)
    path.write_text(json.dumps({"round_id": "003", "status": "OOM"}), encoding="utf-8")
    supervisor = MODULE.Supervisor(state, notifications)
    assert supervisor.reconcile(root) == []
    assert supervisor.state["director_review_required"] is True
    assert supervisor.state["rounds"]["003"]["receipt_identity_conflict"] is True
