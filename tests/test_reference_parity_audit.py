from __future__ import annotations

from pathlib import Path

from scripts.audit_reference_parity import build_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_reference_parity_matrix_has_required_categories_and_frozen_classifications() -> None:
    rows, provenance = build_matrix(ROOT / "tmp/reference_nvarc", ROOT / "tmp/reference_nvarc_kernel")
    assert len(rows) >= 40
    assert {row["area"] for row in rows} == {"MODEL / TOKENIZER", "TASK SERIALIZATION", "AUGMENTATION", "TTT", "DECODING", "CANDIDATE PROCESSING"}
    assert {row["status"] for row in rows} == {"CONFIRMED_MATCH", "CONFIRMED_MISMATCH", "UNVERIFIED"}
    assert any(row["item"] == "LoRA rank" and row["status"] == "CONFIRMED_MISMATCH" and row["impact"] == "HIGH" for row in rows)
    assert any(row["item"] == "primary decoding" and row["status"] == "CONFIRMED_MISMATCH" and row["impact"] == "HIGH" for row in rows)
    assert provenance["reference_repository_commit"] == "846d0198efa752534594e321fc3289fc0a06c657"
