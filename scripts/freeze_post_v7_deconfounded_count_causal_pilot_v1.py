"""Freeze the CPU-only package for the post-V7 deconfounded Count pilot.

No model is loaded and no execution binding, nonce, or GPU authorization is
created.  The only training difference is the 48 Count-primary tasks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
AUDIT = PILOT / "post_v7_generalization_mechanism_audit_v1"
OUT = PILOT / "deconfounded_count_curriculum_causal_pilot_v1"
CONTROL = PILOT / "TREATMENT_SCHEDULE.json"
PANEL = AUDIT / "COUNTERBALANCED_TRAIN_DERIVED_TASKS.json"
FRESH_OOD = AUDIT / "FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json"
RETENTION = PILOT / "measurement_contract_v1/PROTECTED_CAPABILITY_INPUT_PROMPTS.json"
COMPOSITION = PILOT / "scientific_successor_v2/COMPOSITION_SENTINEL_INPUT_PROMPTS.json"
V2_RESULT = PILOT / "independent_confirmation_v2/run_f66fe54019ea096e6bba081131922b02/RESULT_MANIFEST.json"
V7_PROTOCOL = PILOT / "v7_connected_component_count_fit_vs_generation_diagnostic_v1/CPU_ONLY_FROZEN_PROTOCOL.json"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def mutual_information(rows: list[tuple[str, int]]) -> float:
    total = len(rows)
    joint = Counter(rows)
    xs = Counter(x for x, _ in rows)
    ys = Counter(y for _, y in rows)
    return sum((n / total) * math.log2((n * total) / (xs[x] * ys[y])) for (x, y), n in joint.items())


def target_blind_task(task: dict[str, Any]) -> dict[str, Any]:
    prompt = copy.deepcopy(task)
    for test in prompt["test"]:
        test.pop("output", None)
    return prompt


def schedule_and_prompts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    control = copy.deepcopy(load(CONTROL))
    panel = load(PANEL)
    control["arm"] = "CONTROL_ORIGINAL_CONFOUNDED_COUNT_CURRICULUM"
    control["protocol_id"] = "POST_V7_DECONFOUNDED_COUNT_CURRICULUM_CAUSAL_PILOT_V1"
    source_ids_added = 0
    for index, episode in enumerate(control["episodes"]):
        if "source_id" not in episode["task"]:
            episode["task"]["source_id"] = f"POST_V7_CONTROL:{episode['role']}:{index:03d}"
            source_ids_added += 1
    if source_ids_added != 288:
        raise AssertionError("REPLAY_SOURCE_ID_NORMALIZATION_COUNT")
    control["execution_metadata_normalization"] = {
        "source_ids_added": source_ids_added,
        "affected_role": "PROTECTED_REPLAY",
        "scientific_content_changed": False,
        "reason": "task_to_sample requires source_id; identical deterministic metadata is applied before arm derivation.",
    }
    replacements = {record["slot_index"]: record for record in panel["records"]}
    treatment = copy.deepcopy(control)
    treatment["arm"] = "TREATMENT_DECONFOUNDED_COUNT_CURRICULUM"
    treatment["protocol_id"] = "POST_V7_DECONFOUNDED_COUNT_CURRICULUM_CAUSAL_PILOT_V1"
    replaced = 0
    for episode in treatment["episodes"]:
        if episode["role"] != "PRIMARY_CONNECTED_COMPONENT_COUNT":
            continue
        record = replacements[replaced]
        episode["task"] = copy.deepcopy(record["task"])
        episode["topology"] = record["topology"]
        episode["canonical_base_id"] = digest(record["task"])
        episode["curriculum_provenance"] = {
            "panel_id": panel["panel_id"],
            "panel_slot_index": record["slot_index"],
            "task_sha256": record["task_sha256"],
        }
        replaced += 1
    if replaced != 48:
        raise AssertionError("COUNT_REPLACEMENT_COUNT")

    records: list[dict[str, Any]] = []
    commitments: list[dict[str, Any]] = []
    for arm, schedule in (("CONTROL_ORIGINAL_CONFOUNDED", control), ("TREATMENT_DECONFOUNDED", treatment)):
        rows = [row for row in schedule["episodes"] if row["role"] == "PRIMARY_CONNECTED_COMPONENT_COUNT"]
        for index, row in enumerate(rows):
            record_id = f"{arm}:COUNT_TRAIN_FIT:{index:03d}"
            records.append({
                "record_id": record_id,
                "arm": arm,
                "schedule_slot": row["slot"],
                "prompt": target_blind_task(row["task"]),
            })
            commitments.append({"record_id": record_id, "target_sha256": digest(row["task"]["test"][0]["output"])})
    prompts = {
        "schema_version": 1,
        "protocol_id": "POST_V7_DECONFOUNDED_COUNT_CURRICULUM_CAUSAL_PILOT_V1",
        "status": "TARGET_BLIND_TRAIN_FIT_PROMPTS_FROZEN",
        "arms": ["CONTROL_ORIGINAL_CONFOUNDED", "TREATMENT_DECONFOUNDED"],
        "records": records,
        "target_commitments": commitments,
        "target_blind_query": True,
        "scoring_boundary": "Derive synthetic TRAIN targets only after raw predictions are frozen; this is not independent confirmation evidence.",
    }
    return control, treatment, prompts


def count_tokens(schedule: dict[str, Any]) -> tuple[int, dict[str, int]]:
    sys.path.insert(0, str(ROOT / "src"))
    from training_data.pipeline import task_to_sample

    by_role: Counter[str] = Counter()
    for row in schedule["episodes"]:
        by_role[row["role"]] += task_to_sample(row["task"])["assistant_token_count"]
    return sum(by_role.values()), dict(sorted(by_role.items()))


def validate(control: dict[str, Any], treatment: dict[str, Any], prompts: dict[str, Any]) -> dict[str, Any]:
    if len(control["episodes"]) != 384 or len(treatment["episodes"]) != 384:
        raise AssertionError("SCHEDULE_LENGTH")
    roles_control = Counter(row["role"] for row in control["episodes"])
    roles_treatment = Counter(row["role"] for row in treatment["episodes"])
    expected_roles = Counter({"PRIMARY_CONNECTED_COMPONENT_COUNT": 48, "PRIMARY_OBJECT_EXTRACTION": 48, "PROTECTED_REPLAY": 288})
    if roles_control != expected_roles or roles_treatment != expected_roles:
        raise AssertionError("ROLE_COUNTS")
    changed: list[int] = []
    non_count_identical = 0
    for left, right in zip(control["episodes"], treatment["episodes"], strict=True):
        if (left["optimizer_step"], left["slot"], left["role"]) != (right["optimizer_step"], right["slot"], right["role"]):
            raise AssertionError("SCHEDULE_ALIGNMENT")
        if left["role"] == "PRIMARY_CONNECTED_COMPONENT_COUNT":
            changed.append(left["slot"])
        else:
            non_count_identical += int(canonical_bytes(left) == canonical_bytes(right))
    if len(changed) != 48 or non_count_identical != 336:
        raise AssertionError("PRINCIPAL_VARIABLE_ISOLATION")

    count_rows = [row for row in treatment["episodes"] if row["role"] == "PRIMARY_CONNECTED_COMPONENT_COUNT"]
    topology_count: list[tuple[str, int]] = []
    first_matches = 0
    all_source_ids = True
    for row in count_rows:
        task = row["task"]
        all_source_ids &= bool(task.get("source_id"))
        query_count = sum(value != 0 for grid_row in task["test"][0]["output"] for value in grid_row)
        first_count = sum(value != 0 for grid_row in task["train"][0]["output"] for value in grid_row)
        topology_count.append((row["topology"], query_count))
        first_matches += int(query_count == first_count)
    mi = mutual_information(topology_count)
    if abs(mi) > 1e-12 or first_matches != 16 or not all_source_ids:
        raise AssertionError("DECONFOUNDING_VALIDATION")

    control_tokens, control_by_role = count_tokens(control)
    treatment_tokens, treatment_by_role = count_tokens(treatment)
    if control_tokens != treatment_tokens or control_by_role != treatment_by_role:
        raise AssertionError("TOKEN_BUDGET_MISMATCH")
    if control_tokens != 27360 or control_by_role != {
        "PRIMARY_CONNECTED_COMPONENT_COUNT": 2880,
        "PRIMARY_OBJECT_EXTRACTION": 2880,
        "PROTECTED_REPLAY": 21600,
    }:
        raise AssertionError("TOKEN_BUDGET_UNEXPECTED")
    if len(prompts["records"]) != 96 or len(prompts["target_commitments"]) != 96:
        raise AssertionError("TRAIN_FIT_PROMPT_COUNT")
    if any("output" in test for record in prompts["records"] for test in record["prompt"]["test"]):
        raise AssertionError("TRAIN_FIT_TARGET_LEAK")

    ood = load(FRESH_OOD)
    if len(ood["records"]) != 96 or ood["canonical_units"] != 48:
        raise AssertionError("OOD_PANEL_COUNTS")
    if any("output" in test for record in ood["records"] for test in record["prompt"]["test"]):
        raise AssertionError("OOD_TARGET_LEAK")
    if not RETENTION.exists() or not COMPOSITION.exists():
        raise AssertionError("PROTECTED_PANEL_MISSING")

    return {
        "schema_version": 1,
        "status": "PASS_CPU_ONLY_SCIENTIFIC_PACKAGE_VALIDATION",
        "checks": {
            "episodes_per_arm": 384,
            "optimizer_steps_per_arm": len({row["optimizer_step"] for row in control["episodes"]}),
            "role_counts": dict(expected_roles),
            "count_slots_changed": len(changed),
            "non_count_slots_byte_identical": non_count_identical,
            "all_treatment_count_tasks_have_source_id": all_source_ids,
            "replay_source_ids_added_per_arm": control["execution_metadata_normalization"]["source_ids_added"],
            "treatment_topology_count_mutual_information_bits": mi,
            "treatment_first_demo_query_matches": first_matches,
            "supervised_tokens_per_arm": control_tokens,
            "supervised_tokens_by_role": control_by_role,
            "train_fit_target_blind_records": len(prompts["records"]),
            "fresh_ood_canonical_units": ood["canonical_units"],
            "fresh_ood_paired_prompt_records": len(ood["records"]),
            "fresh_ood_test_targets_present": False,
            "model_load": "NOT_RUN",
            "gpu_execution": "NOT_RUN",
        },
        "objective": {
            "name": "ORIGINAL_EQUAL_SLOT_ALL_ASSISTANT_CE",
            "code_pair_reweighting": False,
            "query_only": False,
            "nominal_mass": {"count": 0.125, "object": 0.125, "replay": 0.75, "count_final_query": 0.03125},
        },
    }


def protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": "POST_V7_DECONFOUNDED_COUNT_CURRICULUM_CAUSAL_PILOT_V1",
        "status": "CPU_SCIENTIFIC_CONFIG_FROZEN_GPU_NOT_AUTHORIZED",
        "hypothesis": "Counterbalancing topology, count, demo order, color, and rotation in Count training improves free-running structural-OOD Count accuracy relative to the original shortcut-confounded curriculum.",
        "estimand": "Treatment minus Control paired exact-grid accuracy on 48 fresh structural-OOD canonical units, clustering the two demo orders within canonical unit.",
        "arms": {
            "CONTROL": "Fresh V7 initialization; original 48 confounded Count tasks; unchanged 48 Object tasks and 288 replay tasks.",
            "TREATMENT": "Fresh V7 initialization; 48 counterbalanced Count tasks; all 336 non-Count slots identical to Control.",
        },
        "fixed": {
            "starting_checkpoint": {"name": "V7", "adapter_model_sha256": "1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3"},
            "objective": "Original Equal-Slot all-assistant CE; no Code-Pair weighting; no Query-Only masking.",
            "optimizer_updates": 96,
            "physical_slots_per_update": 4,
            "slots_per_arm": 384,
            "supervised_tokens_per_arm": 27360,
            "decoder": "native Greedy",
            "object_and_replay_slots": "byte-identical across arms",
        },
        "principal_variable": "Count curriculum identifiability/counterbalancing only",
        "evaluation": {
            "train_fit": "48 Count training queries per arm; synthetic targets opened only after raw prediction freeze.",
            "primary_ood": "48 fresh structural-OOD canonical units, two fixed demo orders each; exact grid; paired canonical-unit bootstrap CI.",
            "retention": "Existing protected target-blind panel; paired family-level and aggregate exact-grid deltas.",
            "composition": "Existing non-duplicative Count-then-Object-Extraction sentinel; no-harm only.",
            "references": "V7 is required. Historical V2 Control/Treatment are descriptive references only and are not causal arms in this experiment.",
        },
        "predeclared_rules": {
            "valid_train_fit": "Treatment >=43/48 exact; otherwise classify as supervision/optimization failure.",
            "primary_success": "Treatment-Control >=6/48 canonical units and paired canonical-unit bootstrap 95% lower bound >0.",
            "order_invariance": "Treatment correct under both orders on >=80% of its OOD-correct canonical units and not worse than Control.",
            "retention": "No protected family loses >1 correct versus Control and aggregate protected delta >=-2.",
            "composition": "No loss >1 correct versus Control; this sentinel cannot establish compositional mastery.",
            "promotion": "No promotion from this pilot; a new independent frozen confirmation is required.",
        },
        "conditional_query_only_followup": {
            "run_now": False,
            "trigger": "Equal-Slot plus deconfounded data has query train-fit <43/48 while demonstration-output fit >=90%, or a separately authorized component-gradient measurement finds demo-versus-query cosine <=0.",
            "comparison_if_triggered": "Equal-Slot CE versus Query-Only CE on identical deconfounded data in a separate experiment.",
        },
        "compute_request_not_authorization": {"devices": 1, "minimum": "BF16-capable >=24GB", "arms": 2, "execution": "sequential fresh V7 initializations", "hard_cap_gpu_hours": 2.0},
        "boundaries": ["no GPU launch", "no model load", "no Gold/dGold", "no V2 target reuse", "no execution binding or nonce", "no checkpoint promotion"],
    }


def main() -> None:
    control, treatment, prompts = schedule_and_prompts()
    validation = validate(control, treatment, prompts)
    write_json(OUT / "TREATMENT_SCHEDULE.json", treatment)
    write_json(OUT / "CONTROL_SCHEDULE.json", control)
    write_json(OUT / "TRAIN_FIT_INPUT_PROMPTS.json", prompts)
    write_json(OUT / "PROTOCOL.json", protocol())
    write_json(OUT / "CPU_VALIDATION.json", validation)
    sources = {
        "control_schedule": CONTROL,
        "counterbalanced_panel": PANEL,
        "fresh_ood_prompts": FRESH_OOD,
        "retention_prompts": RETENTION,
        "composition_prompts": COMPOSITION,
        "v2_result": V2_RESULT,
        "v7_protocol": V7_PROTOCOL,
    }
    outputs = [OUT / "CONTROL_SCHEDULE.json", OUT / "TREATMENT_SCHEDULE.json", OUT / "TRAIN_FIT_INPUT_PROMPTS.json", OUT / "PROTOCOL.json", OUT / "CPU_VALIDATION.json"]
    manifest = {
        "schema_version": 1,
        "experiment_id": "POST_V7_DECONFOUNDED_COUNT_CURRICULUM_CAUSAL_PILOT_V1",
        "status": "CPU_PACKAGE_COMPLETE_GPU_NOT_AUTHORIZED",
        "source_identities": {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": file_digest(path)} for name, path in sources.items()},
        "output_identities": {path.name: file_digest(path) for path in outputs},
        "next_gate": "Synchronous scientific review of this single frozen package, then fresh explicit GPU execution authorization and a separate execution binding.",
    }
    write_json(OUT / "MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "output": str(OUT), "validation": validation["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
