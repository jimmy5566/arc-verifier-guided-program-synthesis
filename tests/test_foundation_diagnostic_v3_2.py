import json
from pathlib import Path

from foundation_capability_bank_v3_2.pipeline import COMPOSITION_SPECS, SOURCE_COMMIT, minimal_task

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts/foundation_diagnostic_v3_2_validity"


def read(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


def test_source_pin_and_frozen_bank():
    assert SOURCE_COMMIT == "d39d7cfc768da01dafa318357f1196d387d4a0b1"
    assert read("V3_ATOMIC_TRAINING_BANK_INTEGRITY.json")["status"] == "PASS"


def test_conditional_tasks_expose_both_branches():
    for capability in ("if/else on attribute", "if/else on relation", "if/else on count", "branch-balanced conditional tasks"):
        _, evidence = minimal_task(capability, 3, 1)
        branches = [x["branch"] for x in evidence["training_examples"]]
        assert branches.count("TRUE") == branches.count("FALSE") == 2
    audit = read("CONDITIONAL_BRANCH_IDENTIFIABILITY_AUDIT.json")
    assert audit["status"] == "PASS"
    assert audit["tasks_with_both_branches"] == audit["total_tasks"]


def test_true_identifiability_and_shortcut_audits():
    ident = read("DEMONSTRATION_IDENTIFIABILITY_AUDIT_V3_2.json")
    assert ident["claim_scope"] == "UNIQUE_WITHIN_FROZEN_DIAGNOSTIC_HYPOTHESIS_SET"
    assert ident["status"] == "PASS"
    assert ident["non_unique_or_inconsistent"] == []
    assert read("RELATION_POSITION_SHORTCUT_AUDIT.json")["status"] == "PASS"
    assert read("MINIMAL_CONTRAST_SCENE_DIVERSITY_AUDIT.json")["status"] == "PASS"


def test_parameter_and_composition_breadth():
    assert read("PARAMETER_NUISANCE_DIVERSITY_AUDIT.json")["status"] == "PASS"
    comp = read("COMPOSITION_DIAGNOSTIC_DEV_V1_2.json")
    assert set(comp["category_counts"]) == {x[0] for x in COMPOSITION_SPECS}
    assert all(x >= 8 for x in comp["category_counts"].values())
    assert comp["C1_DEV_ISOLATED_MODEL_ACCESSED"] is False
    assert comp["C2_DEV_ISOLATED_MODEL_ACCESSED"] is False


def test_overlap_tokenizer_and_gate():
    overlap = read("DIAGNOSTIC_TRAIN_EXCLUSION_AUDIT_V3_2.json")
    assert overlap["status"] == "PASS"
    assert "individual_train_pair_sha256" in overlap["keys_actually_compared"]
    assert read("V3_2_REAL_TOKENIZER_PARITY_AUDIT.json")["status"] == "PASS"
    assert read("V3_2_TOKENIZER_CONTEXT_AUDIT.json")["status"] == "PASS"
    assert read("V3_2_MULTIPROCESS_DETERMINISM_AUDIT.json")["status"] == "PASS"
    gate = read("FOUNDATION_DIAGNOSTIC_V1_2_GATE.json")
    assert gate["GPU_DIAGNOSTIC_READY"] is True
    assert gate["GPU_TRAINING_STARTED"] is False
    assert gate["META_HOLDOUT_MODEL_ACCESSED"] is False
