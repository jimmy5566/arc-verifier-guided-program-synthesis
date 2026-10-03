from pathlib import Path

from scripts.run_phase3a_eval60_d24_aug8_r1024 import _existing_output_gate, _profile_name
from scripts.run_ttt24_aug8_r1024_core_v1 import _atomic_json, _sha_file


def test_profile_mapping_uses_existing_frozen_boundaries() -> None:
    assert _profile_name(2048) == "PROFILE_S"
    assert _profile_name(2049) == "PROFILE_M"
    assert _profile_name(2654) == "PROFILE_L_LOW"
    assert _profile_name(4097) == "PROFILE_L_HIGH"
    assert _profile_name(6494) == "PROFILE_XL_CONSERVATIVE"


def test_existing_output_requires_an_independent_hash_ledger(tmp_path: Path) -> None:
    selected = {"output_id": "abc:o0"}
    assert _existing_output_gate(tmp_path, selected) is None
    raw = tmp_path / "RAW_OUTPUTS" / "abc__o0.json"; raw.parent.mkdir()
    raw.write_text("{}", encoding="utf-8")
    try:
        _existing_output_gate(tmp_path, selected)
    except RuntimeError as error:
        assert "UNVERIFIED_PARTIAL_OUTPUT_REFUSED" in str(error)
    else:
        raise AssertionError("partial output was incorrectly reusable")
    checkpoint = tmp_path / "OUTPUT_CHECKPOINTS" / "abc__o0.json"; checkpoint.parent.mkdir(); checkpoint.write_text("{}", encoding="utf-8")
    eos = tmp_path / "EOS_EVENTS" / "abc__o0.jsonl.gz"; eos.parent.mkdir(); eos.write_bytes(b"not-a-gzip-needed-for-hash")
    receipt = tmp_path / "OUTPUT_RECEIPTS" / "abc__o0.json"; receipt.parent.mkdir(); _atomic_json(receipt, {"status": "COMPLETE"})
    files = [raw, checkpoint, eos, receipt]
    ledger = tmp_path / "OUTPUT_HASHES" / "abc__o0.json"
    _atomic_json(ledger, {"files": {str(path.relative_to(tmp_path)): _sha_file(path) for path in files}})
    verification = tmp_path / "OUTPUT_HASH_VERIFICATION" / "abc__o0.json"; _atomic_json(verification, {"status": "PASS"})
    result = _existing_output_gate(tmp_path, selected)
    assert result == {"output_id": "abc:o0", "resumed": True, "hash_checked": 4}
