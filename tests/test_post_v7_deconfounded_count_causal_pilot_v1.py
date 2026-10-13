import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_post_v7_deconfounded_count_causal_pilot_v1.py"
SPEC = importlib.util.spec_from_file_location("freeze_deconfounded", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_only_count_slots_change_and_budget_matches() -> None:
    control, treatment, prompts = MODULE.schedule_and_prompts()
    validation = MODULE.validate(control, treatment, prompts)
    assert validation["status"] == "PASS_CPU_ONLY_SCIENTIFIC_PACKAGE_VALIDATION"
    assert validation["checks"]["count_slots_changed"] == 48
    assert validation["checks"]["non_count_slots_byte_identical"] == 336
    assert validation["checks"]["supervised_tokens_per_arm"] == 27360


def test_treatment_breaks_known_shortcuts() -> None:
    _, treatment, _ = MODULE.schedule_and_prompts()
    rows = [row for row in treatment["episodes"] if row["role"] == "PRIMARY_CONNECTED_COMPONENT_COUNT"]
    contingency = {}
    first_matches = 0
    for row in rows:
        query = sum(value != 0 for grid in row["task"]["test"][0]["output"] for value in grid)
        first = sum(value != 0 for grid in row["task"]["train"][0]["output"] for value in grid)
        contingency.setdefault(row["topology"], {}).setdefault(query, 0)
        contingency[row["topology"]][query] += 1
        first_matches += query == first
    assert first_matches == 16
    assert all(set(counts) == {1, 2, 3} and len(set(counts.values())) == 1 for counts in contingency.values())


def test_all_inference_inputs_are_target_blind() -> None:
    _, _, prompts = MODULE.schedule_and_prompts()
    assert all("output" not in test for record in prompts["records"] for test in record["prompt"]["test"])
    ood = json.loads(MODULE.FRESH_OOD.read_text(encoding="utf-8"))
    assert all("output" not in test for record in ood["records"] for test in record["prompt"]["test"])


def test_protocol_excludes_code_pair_and_query_only() -> None:
    frozen = MODULE.protocol()
    assert frozen["fixed"]["objective"].startswith("Original Equal-Slot")
    assert frozen["conditional_query_only_followup"]["run_now"] is False
    assert frozen["status"] == "CPU_SCIENTIFIC_CONFIG_FROZEN_GPU_NOT_AUTHORIZED"
