import json
from pathlib import Path

from foundation_capability_bank_v3_1.pipeline import (
    SOURCE_COMMIT, V3_FINGERPRINT, V3_TRAIN_SHA256, SURFACES,
    measurement_rows,
)

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts/foundation_capability_bank_v3_1"


def read(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


def test_source_and_v3_bank_pins():
    assert SOURCE_COMMIT == "75c5546add5dd62f4338a1cc08e3e84ad2417a91"
    assert V3_FINGERPRINT == "80149142606717eb617f415c486fa1c26b19e9a868404a9870efc0e501ceff2f"
    assert V3_TRAIN_SHA256 == "27b3b495ce9538683d70c48fd3b58e462e0efd773042012b88467f958f0b7504"
    assert read("V3_ATOMIC_TRAINING_BANK_INTEGRITY.json")["status"] == "PASS"


def test_all_83_capabilities_have_honest_measurement_types():
    rows = measurement_rows()
    assert len(rows) == 83
    assert {x["measurement_type"] for x in rows} == {"DIRECT_ATOMIC", "MINIMAL_CONTRAST", "COMPOSITE_ONLY"}
    assert all(x["interpretation_limit"] for x in rows)


def test_realized_parameter_surfaces_and_domains():
    audit = read("PARAMETER_DOMAIN_MECHANICAL_AUDIT.json")
    assert set(SURFACES) == {x["axis"] for x in audit["checks"]}
    assert audit["episode_count"] >= 9 * 32
    assert audit["status"] == "PASS"
    assert all(x["actual_values_read_from_tasks"] for x in audit["checks"])


def test_composition_stage_contracts():
    c1 = read("C1_PARTITION_V3_1.json")
    assert c1["partition_counts"] == {"C1_DEV_ISOLATED": 10, "C1_TRAIN_CANDIDATES": 10}
    assert all(x["atomic_primitives_train_seen"] and x["exact_two_step_absent_from_atomic_basis"] for x in c1["programs"])
    c2 = read("C2_DEPENDENCY_CONTRACT_V3_1.json")
    assert c2["status"] == "PASS"
    assert all(x["future_classification"] in {"SEEN_2STEP_SUBPROGRAMS_UNSEEN_3STEP_COMPOSITION", "ATOMIC_PRIMITIVES_SEEN_ONLY"} for x in c2["programs"])


def test_static_audits_and_final_gate():
    for name in (
        "DIAGNOSTIC_TRAIN_EXCLUSION_AUDIT_V3_1.json",
        "SEMANTIC_NONDEGENERACY_AUDIT_V3_1.json",
        "DEMONSTRATION_IDENTIFIABILITY_AUDIT_V3_1.json",
        "V3_1_CONTEXT_LENGTH_AUDIT.json",
        "V3_1_TOKENIZER_PARITY_AUDIT.json",
        "V3_1_MULTIPROCESS_DETERMINISM_AUDIT.json",
    ):
        assert read(name)["status"] == "PASS", name
    gate = read("FOUNDATION_DIAGNOSTIC_V1_1_GATE.json")
    assert gate["GPU_DIAGNOSTIC_READY"] is True
    assert gate["GPU_TRAINING_STARTED"] is False
    assert gate["DIAGNOSTIC_TRAIN_OVERLAP"] == 0
    assert gate["META_HOLDOUT_MODEL_ACCESSED"] is False
    assert gate["EVAL60_GOLD_ACCESSED"] is gate["KAGGLE_GOLD_ACCESSED"] is gate["ARC_HIDDEN_GOLD_ACCESSED"] is False
