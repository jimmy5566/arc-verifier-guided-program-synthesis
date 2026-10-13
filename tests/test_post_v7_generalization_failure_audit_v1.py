from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_post_v7_generalization_failure_v1.py"
SPEC = importlib.util.spec_from_file_location("post_v7_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_original_shortcuts_are_exactly_reproduced() -> None:
    import json

    schedule = json.loads(MODULE.SCHEDULE.read_text(encoding="utf-8"))
    audit, rows = MODULE.original_audit(schedule)
    assert len(rows) == 48
    assert audit["topology_target_contingency"] == {"L": {1: 16}, "T": {2: 16}, "Z": {3: 16}}
    assert audit["topology_majority_predictor"]["correct"] == 48
    assert audit["first_demo_count_predictor"]["correct"] == 48
    assert audit["same_count_by_demo_position"]["1"]["correct"] == 48
    assert audit["same_count_by_demo_position"]["2"]["correct"] == 0
    assert audit["same_count_by_demo_position"]["3"]["correct"] == 0
    assert audit["query_foreground_colors"] == {4: 48}
    assert audit["query_rotation_phase_mod_4"] == {3: 48}


def test_counterbalanced_panel_breaks_shortcuts_and_preserves_oracle() -> None:
    import json

    schedule = json.loads(MODULE.SCHEDULE.read_text(encoding="utf-8"))
    _, rows = MODULE.original_audit(schedule)
    panel = MODULE.deconfounded_train_panel(rows)
    validation = panel["validation"]
    assert validation["records"] == 48
    assert validation["topology_target_mutual_information_bits"] == 0.0
    assert validation["first_demo_query_matches"] == 16
    assert validation["demo_order_pairs_oracle_invariant"] == 48
    assert validation["historical_exact_task_collisions"] == 0
    for record in panel["records"]:
        task = record["task"]
        for pair in [*task["train"], *task["test"]]:
            assert MODULE.count_components(pair["input"]) == MODULE.output_count(pair["output"])
            assert MODULE.foreground_color(pair["input"]) == MODULE.foreground_color(pair["output"])


def test_objective_implementation_masks_shifts_indices_and_mass() -> None:
    import json

    schedule = json.loads(MODULE.SCHEDULE.read_text(encoding="utf-8"))
    audit = MODULE.objective_implementation_audit(schedule)
    assert audit["status"] == "CONFIRMED_CODE_PAIR_LOCATOR_SEMANTIC_DEFECT_EQUAL_SLOT_CONTRACT_PASS"
    assert audit["checks"]["user_mask_and_assistant_identity_pass"] == 48
    assert audit["checks"]["causal_shift_valid_label_count_60_pass"] == 48
    assert audit["checks"]["code_pair_locator_pass"] == 48
    assert audit["checks"]["code_pair_documented_semantic_locator_pass"] == 0
    assert audit["checks"]["located_code_tokens"] == 384
    assert audit["checks"]["weight_mass_preservation_pass"] == 48
    mass = audit["nominal_equal_slot_objective_mass"]
    assert mass["count_primary"]["fraction"] == 0.125
    assert mass["object_primary"]["fraction"] == 0.125
    assert mass["protected_replay"]["fraction"] == 0.75
    assert mass["count_query_all_tokens_fraction_of_total"] == 0.03125
    assert audit["confirmed_code_pair_defect"]["implemented_nonzero_patterns_by_count"]["2"] == audit["confirmed_code_pair_defect"]["implemented_nonzero_patterns_by_count"]["3"]


def test_fresh_ood_is_target_blind_and_paired() -> None:
    panel = MODULE.fresh_ood_prompts()
    assert panel["canonical_units"] == 48
    assert panel["prompt_records"] == 96
    assert len(panel["target_commitments"]) == 96
    assert all(len({MODULE.rotate(points, turns) for turns in range(4)}) == 4 for points in MODULE.OOD_TOPOLOGIES.values())
    groups = {}
    for record in panel["records"]:
        assert "output" not in record["prompt"]["test"][0]
        groups.setdefault(record["canonical_unit_id"], []).append(record)
    assert len(groups) == 48
    assert all(len(records) == 2 for records in groups.values())
    for records in groups.values():
        assert records[0]["prompt"]["test"] == records[1]["prompt"]["test"]
        hashes_a = sorted(MODULE.digest(pair) for pair in records[0]["prompt"]["train"])
        hashes_b = sorted(MODULE.digest(pair) for pair in records[1]["prompt"]["train"])
        assert hashes_a == hashes_b
