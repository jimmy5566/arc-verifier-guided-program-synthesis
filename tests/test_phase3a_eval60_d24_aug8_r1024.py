from pathlib import Path

import scripts.run_phase3a_eval60_d24_aug8_r1024 as phase3
from scripts.run_phase3a_eval60_d24_aug8_r1024 import _existing_output_gate, _freeze_output, _profile_name
from scripts.run_ttt24_aug8_r1024_core_v1 import _atomic_json


def test_profile_mapping_uses_existing_frozen_boundaries() -> None:
    assert _profile_name(2048) == "PROFILE_S"
    assert _profile_name(2049) == "PROFILE_M"
    assert _profile_name(2654) == "PROFILE_L_LOW"
    assert _profile_name(4097) == "PROFILE_L_HIGH"
    assert _profile_name(6494) == "PROFILE_XL_CONSERVATIVE"


def test_d48_uses_the_same_phase3_controller_with_a_distinct_identity() -> None:
    phase3._configure_depth(48)
    try:
        assert phase3.DEPTH == 48
        assert phase3.EXPERIMENT == "PHASE3B_EVAL60_D48_AUG8_R1024_V1"
        assert phase3._other_depth_status() == "SEPARATE"
    finally:
        phase3._configure_depth(24)


def test_existing_output_requires_an_independent_hash_ledger(tmp_path: Path) -> None:
    selected = {"output_id": "abc:o0"}
    assert _existing_output_gate(tmp_path, selected) is None
    raw = tmp_path / "RAW_OUTPUTS" / "abc_o0.json"; raw.parent.mkdir()
    raw.write_text("{}", encoding="utf-8")
    try:
        _existing_output_gate(tmp_path, selected)
    except RuntimeError as error:
        assert "UNVERIFIED_PARTIAL_OUTPUT_REFUSED" in str(error)
    else:
        raise AssertionError("partial output was incorrectly reusable")
    checkpoint = tmp_path / "OUTPUT_CHECKPOINTS" / "abc_o0.json"; checkpoint.parent.mkdir(); checkpoint.write_text("{}", encoding="utf-8")
    eos = tmp_path / "EOS_EVENTS" / "abc_o0.jsonl.gz"; eos.parent.mkdir(); eos.write_bytes(b"not-a-gzip-needed-for-hash")
    receipt = tmp_path / "OUTPUT_RECEIPTS" / "abc_o0.json"; receipt.parent.mkdir(); _atomic_json(receipt, {"status": "COMPLETE"})
    frozen = _freeze_output(tmp_path, selected)
    assert frozen == {"output_id": "abc:o0", "resumed": False, "hash_checked": 4}
    result = _existing_output_gate(tmp_path, selected)
    assert result == {"output_id": "abc:o0", "resumed": True, "hash_checked": 4}
