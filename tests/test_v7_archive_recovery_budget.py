from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('recovery', Path(__file__).resolve().parents[1]/'scripts/run_v7_eval60_archive_recovery.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_deadline_is_hard_and_leaves_closing_reserve():
    assert recovery.remaining_seconds((datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(), 3600) == 0
    budget = recovery.remaining_seconds((datetime.now(timezone.utc)+timedelta(seconds=1000)).isoformat(), 3600)
    assert 398 <= budget <= 400
    assert recovery.remaining_seconds((datetime.now(timezone.utc)+timedelta(hours=8)).isoformat(), 1800) == 1800


def test_recovery_cannot_invoke_generation_or_training():
    code = Path(recovery.__file__).read_text(encoding='utf-8')
    assert "ROOT/'scripts/freeze_phase3_archive.py'" in code
    assert 'run_v7_ttt24_eval60_train.py' not in code
    assert 'run_eval60_ready_restore_v1.py' not in code
    assert 'exist_ok=False' in code
