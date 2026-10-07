from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_remote_contract_forbids_scientific_interpretation() -> None:
    contract = json.loads((ROOT / "orchestration/runpod/REMOTE_RUNTIME_CONTRACT.json").read_text(encoding="utf-8"))
    assert contract["terminal_states"] == ["SUCCESS", "TRAIN_FAILED", "OOM", "INFRA_FAILED", "INTERRUPTED"]
    assert contract["no_model_loading_in_smoke_tests"] is True
    assert "never accepts" in contract["scientific_interpretation"]


def test_wrapper_is_detached_and_writes_receipts() -> None:
    wrapper = (ROOT / "orchestration/runpod/arc2_remote_job.sh").read_text(encoding="utf-8")
    assert "nohup setsid" in wrapper
    assert "TERMINAL_RECEIPT.json" in wrapper
    assert "events.jsonl" in wrapper


def test_watchdog_scope_is_process_only() -> None:
    protocol = json.loads((ROOT / "orchestration/watchdog/WATCHDOG_PROTOCOL.json").read_text(encoding="utf-8"))
    assert protocol["role"] == "non-scientific process watchdog"
    assert "scientific decisions" in protocol["forbidden"]
