from __future__ import annotations
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import freeze_e04_e_equal_slot_marker_replay_pilot as e04e
from scripts import launch_e04_e_equal_slot_marker_replay_pilot as launcher


def test_e04e_v2_freezer_and_accounting() -> None:
    control, treatment, meta = e04e.build_schedules()
    e04e.verify_schedules(control, treatment)
    accounting = e04e.accounting(control, treatment)
    assert meta['fixed_replay_slots'] and len(meta['fixed_replay_slots']) == 192
    assert len(meta['swap_slots']) == 96
    assert sum(x['role'] == 'FIXED_TURN_ROTATION' for x in control) == 96
    assert sum(x['replay_source'] == 'E04_C_BYTE_IDENTICAL_NO_TRANSFORM' for x in control) == 192
    assert sum(x['replay_source'] == 'E04_D_MATCHED_MARKER_REPLAY_SWAP' for x in control) == 96
    assert accounting['arms']['CONTROL']['slot_count'] == 384
    assert accounting['arms']['TREATMENT']['slot_count'] == 384
    assert accounting['raw_token_difference_treatment_minus_control']['supervised_tokens'] != 0
    assert accounting['raw_token_difference_treatment_minus_control']['transformer_tokens'] != 0


def test_e04e_v2_rejects_cross_arm_role_mutation() -> None:
    control, treatment, _ = e04e.build_schedules()
    treatment[300]['role'] = 'wrong'
    try:
        e04e.verify_schedules(control, treatment)
    except e04e.FreezeFailure as exc:
        assert str(exc) == 'E04E_CROSS_ARM_METADATA'
    else:
        raise AssertionError('cross-arm role mutation accepted')


def test_e04e_remote_launcher_script_has_valid_bash_syntax() -> None:
    script = launcher.remote_script(
        {
            'output_root': '/workspace/arc2/e04e/run_001_nonce',
            'protocol_id': 'E04E_TEST',
            'worker_source_commit': 'a' * 40,
            'nonce': 'nonce',
        },
        'b' * 40,
        'experiments/e04e/BINDING.json',
        'c' * 64,
    )
    check = subprocess.run(['bash', '-n'], input=script, text=True, capture_output=True)
    assert check.returncode == 0, check.stderr
