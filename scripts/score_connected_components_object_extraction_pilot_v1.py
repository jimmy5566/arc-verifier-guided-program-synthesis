"""Complete CPU-only post-freeze scorer for the E1 atomic pilot.

Targets are reconstructed only after all six raw artifacts pass identity and
completeness checks.  The scorer implements the frozen paired outcome, Batch1
validity, protected retention, and promotion contract.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
PRIMARY_TAGS = ("V7", "CONTROL_NARROW_STRUCTURE", "TREATMENT_STRUCTURE_DIVERSE")
ALL_TAGS = (
    "V7", "V7_B1", "CONTROL_NARROW_STRUCTURE", "CONTROL_NARROW_STRUCTURE_B1",
    "TREATMENT_STRUCTURE_DIVERSE", "TREATMENT_STRUCTURE_DIVERSE_B1",
)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canon(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
def fail(reason): raise RuntimeError("E1_SCORE_" + reason)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canon(value) + b"\n")
    os.replace(temporary, path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    loaded = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str((ROOT / path).parent))
    spec.loader.exec_module(loaded)
    return loaded


def _dataset_rows(binding, arm=None):
    worker = module("scripts/run_connected_components_object_extraction_pilot_v1.py", "e1_worker_for_score")
    return worker._datasets(binding, arm)


def _expected_sets(binding):
    sets = {
        "V7": {row["record_id"] for row in _dataset_rows(binding)},
        "CONTROL_NARROW_STRUCTURE": {row["record_id"] for row in _dataset_rows(binding, "CONTROL_NARROW_STRUCTURE")},
        "TREATMENT_STRUCTURE_DIVERSE": {row["record_id"] for row in _dataset_rows(binding, "TREATMENT_STRUCTURE_DIVERSE")},
    }
    for tag in PRIMARY_TAGS:
        sets[tag + "_B1"] = set(binding["evaluation"]["b1_subset_by_checkpoint"][tag])
    return sets


def _validate_construct(binding):
    path = binding.get("scientific_construct_validation_path")
    expected = binding.get("scientific_construct_validation_sha256")
    if not path or not expected: fail("SCIENTIFIC_CONSTRUCT_SUCCESSOR_REQUIRED")
    full = ROOT / path
    if not full.is_file() or sha(full) != expected: fail("SCIENTIFIC_CONSTRUCT_IDENTITY")
    result = json.loads(full.read_text())
    if result.get("status") != "PASS_CPU_ONLY_INDEPENDENT_SEMANTIC_ORACLE_VALIDATION": fail("SCIENTIFIC_CONSTRUCT_STATUS")
    if result.get("v1_composition_disposition") != "INVALID_CONSTRUCT_PRESERVED_NOT_USED_BY_SUCCESSOR": fail("SCIENTIFIC_CONSTRUCT_V1_DISPOSITION")
    return result


def _load_raw(binding, output):
    if sha(ROOT / binding["score_entrypoint_path"]) != binding["score_entrypoint_sha256"]: fail("SCORER_IDENTITY")
    _validate_construct(binding)
    freeze_path = output / "RAW_FREEZE_RECEIPT.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("status") != "COMPLETE_RAW_FROZEN_NO_TARGETS" or freeze.get("targets_accessed") is not False: fail("RAW_FREEZE_REQUIRED")
    if set(freeze.get("raw_sha256", {})) != set(ALL_TAGS): fail("RAW_ARTIFACT_SET")
    expected_sets = _expected_sets(binding)
    raw = {}
    for tag in ALL_TAGS:
        path = output / f"{tag}_RAW.jsonl"
        if not path.is_file() or sha(path) != freeze["raw_sha256"][tag]: fail("RAW_IDENTITY:" + tag)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        ids = [row.get("record_id") for row in rows]
        if len(ids) != len(set(ids)): fail("RAW_DUPLICATE_RECORD:" + tag)
        if set(ids) != expected_sets[tag]: fail("RAW_COMPLETENESS:" + tag)
        raw[tag] = rows
    return freeze, raw


def _target_context(binding):
    oracle = module("scripts/connected_components_object_extraction_semantic_oracle.py", "e1_independent_oracle")
    task_rows, family = {}, {}
    for arm in (None, "CONTROL_NARROW_STRUCTURE", "TREATMENT_STRUCTURE_DIVERSE"):
        for row in _dataset_rows(binding, arm): task_rows.setdefault(row["record_id"], row["task"])
    configured = binding.get("dataset_paths", {})
    defaults = {"OOD": "OOD_INPUT_PROMPTS.json", "RETENTION": "RETENTION_INPUT_PROMPTS.json", "COMPOSITION": "COMPOSITION_SENTINEL_INPUT_PROMPTS.json"}
    for split, name in defaults.items():
        path = ROOT / configured[split] if split in configured else BASE / name
        for index, row in enumerate(json.loads(path.read_text())): family[f"{split}:{index}"] = row["latent_condition"]["family"]
    protected_rows = json.loads((BASE / "measurement_contract_v1/PROTECTED_CAPABILITY_INPUT_PROMPTS.json").read_text())["records"]
    protected = {"PROTECTED:" + row["protected_id"]: row for row in protected_rows}
    v3 = module("scripts/e04_orientation_fixed_demonstration_baseline_v3.py", "e1_v3_target")
    bases = {row["canonical_base_id"]: row for row in v3.v2.canonical_bases("VALIDATION")}
    train = {}
    for arm, filename in (("CONTROL_NARROW_STRUCTURE", "CONTROL_SCHEDULE.json"), ("TREATMENT_STRUCTURE_DIVERSE", "TREATMENT_SCHEDULE.json")):
        for row in json.loads((BASE / filename).read_text())["episodes"]:
            train[f"TRAIN_FIT:{arm}:{row['slot']}"] = row["task"]["test"][0]["output"]

    def expected(record_id):
        if record_id in family:
            return oracle.target_for_family(family[record_id], task_rows[record_id]["test"][0]["input"])
        if record_id in train: return train[record_id]
        if record_id in protected:
            row = protected[record_id]
            return v3.target(bases[row["canonical_base_id"]], row["control_marker_turn"], row["condition"], row["source_row_index"])["target"]
        fail("UNKNOWN_RECORD:" + record_id)

    return expected, family, {record_id: row["condition"] for record_id, row in protected.items()}, task_rows


def _score_rows(raw, expected, task_rows):
    scored = {}
    for tag, values in raw.items():
        rows = {}
        for row in values:
            record_id = row["record_id"]
            if row.get("prompt_sha256") != hashlib.sha256(canon(task_rows[record_id])).hexdigest(): fail("PROMPT_IDENTITY:" + tag + ":" + record_id)
            rows[record_id] = {**row, "exact": bool(row.get("parser_valid")) and row.get("parsed_grid") == expected(record_id)}
        scored[tag] = rows
    return scored


def _paired_bootstrap(left, right, record_ids, seed=20261012, replicates=10000):
    differences = [int(left[x]["exact"]) - int(right[x]["exact"]) for x in record_ids]
    rng, draws = random.Random(seed), []
    for _ in range(replicates): draws.append(sum(differences[rng.randrange(len(differences))] for _ in differences))
    draws.sort()
    return {"delta_correct": sum(differences), "paired_bootstrap_95pct_ci_correct": [draws[int(.025 * replicates)], draws[int(.975 * replicates) - 1]], "replicates": replicates, "seed": seed}


def _batch1_gate(binding, scored):
    thresholds, report, valid = binding["evaluation"]["batch1_gate"], {}, True
    for tag in PRIMARY_TAGS:
        primary, b1 = scored[tag], scored[tag + "_B1"]
        ids = binding["evaluation"]["b1_subset_by_checkpoint"][tag]
        n = len(ids)
        metrics = {
            "generated_token_canonical_agreement": sum(primary[x].get("generated_token_ids") == b1[x].get("generated_token_ids") for x in ids) / n,
            "parsed_grid_agreement": sum(primary[x].get("parsed_grid") == b1[x].get("parsed_grid") for x in ids) / n,
            "scorer_agreement": sum(primary[x]["exact"] == b1[x]["exact"] for x in ids) / n,
            "aggregate_exact_rate_delta": abs(sum(primary[x]["exact"] for x in ids) / n - sum(b1[x]["exact"] for x in ids) / n),
            "records": n, "record_ids": ids,
        }
        passed = metrics["generated_token_canonical_agreement"] >= thresholds["generated_token_canonical_agreement_minimum"] and metrics["parsed_grid_agreement"] >= thresholds["parsed_grid_agreement_minimum"] and metrics["scorer_agreement"] >= thresholds["scorer_agreement_minimum"] and metrics["aggregate_exact_rate_delta"] <= thresholds["aggregate_exact_rate_delta_maximum"]
        metrics["status"] = "PASS" if passed else thresholds.get("failure", "INVALID_NOT_INTERPRETABLE")
        report[tag], valid = metrics, valid and passed
    return {"status": "PASS" if valid else "INVALID_NOT_INTERPRETABLE", "by_checkpoint": report}


def _aggregate(scored, family, protected_condition):
    by_artifact = {}
    for tag, rows in scored.items():
        buckets = {}
        for row in rows.values():
            bucket = buckets.setdefault(row["split"], {"rows": 0, "parser_valid": 0, "exact": 0})
            bucket["rows"] += 1; bucket["parser_valid"] += int(bool(row.get("parser_valid"))); bucket["exact"] += int(row["exact"])
        by_artifact[tag] = {"by_split": buckets}
    control, treatment, v7 = scored["CONTROL_NARROW_STRUCTURE"], scored["TREATMENT_STRUCTURE_DIVERSE"], scored["V7"]
    ood_ids = sorted(x for x in control if x.startswith("OOD:"))
    primary = _paired_bootstrap(treatment, control, ood_ids)
    by_family = {}
    for index, name in enumerate(("CONNECTED_COMPONENT_COUNT", "OBJECT_EXTRACTION")):
        by_family[name] = _paired_bootstrap(treatment, control, [x for x in ood_ids if family[x] == name], seed=20261013 + index)
    preservation = {}
    for arm_index, arm in enumerate(("CONTROL_NARROW_STRUCTURE", "TREATMENT_STRUCTURE_DIVERSE")):
        arm_rows, results = scored[arm], {}
        for index, condition in enumerate(("FIXED_TURN_ROTATION_CONTROL", "MARKER_BINDING_CONTROL")):
            ids = sorted(x for x, value in protected_condition.items() if value == condition)
            results[condition] = _paired_bootstrap(arm_rows, v7, ids, seed=20261100 + arm_index * 10 + index)
        ids = sorted(x for x in arm_rows if x.startswith("RETENTION:"))
        results["NO_TRANSFORM_RETENTION_CONTROL"] = _paired_bootstrap(arm_rows, v7, ids, seed=20261200 + arm_index)
        preservation[arm] = results
    train_fit = {}
    for arm in ("CONTROL_NARROW_STRUCTURE", "TREATMENT_STRUCTURE_DIVERSE"):
        ids = sorted(x for x in scored[arm] if x.startswith("TRAIN_FIT:"))
        train_fit[arm] = {}
        for name, parity in (("CONNECTED_COMPONENT_COUNT", 0), ("OBJECT_EXTRACTION", 1)):
            selected = [x for x in ids if int(x.rsplit(":", 1)[1]) % 2 == parity]
            train_fit[arm][name] = {"exact": sum(scored[arm][x]["exact"] for x in selected), "rows": len(selected)}
    composition_ids = sorted(x for x in control if x.startswith("COMPOSITION:"))
    composition = {"V7": sum(v7[x]["exact"] for x in composition_ids), "CONTROL_NARROW_STRUCTURE": sum(control[x]["exact"] for x in composition_ids), "TREATMENT_STRUCTURE_DIVERSE": sum(treatment[x]["exact"] for x in composition_ids), "rows": len(composition_ids), "treatment_minus_control": _paired_bootstrap(treatment, control, composition_ids, seed=20261300)}
    return by_artifact, primary, by_family, preservation, train_fit, composition


def score(binding_path: Path, output: Path):
    binding = json.loads(binding_path.read_text())
    freeze, raw = _load_raw(binding, output)
    expected, family, protected_condition, task_rows = _target_context(binding)
    scored = _score_rows(raw, expected, task_rows)
    b1 = _batch1_gate(binding, scored)
    by_artifact, primary, by_family, preservation, train_fit, composition = _aggregate(scored, family, protected_condition)
    family_ok = all(value["delta_correct"] >= -1 for value in by_family.values())
    preservation_ok = all(value["delta_correct"] >= -1 for results in preservation.values() for value in results.values())
    primary_ok = primary["delta_correct"] >= 4 and primary["paired_bootstrap_95pct_ci_correct"][0] > 0
    promoted = b1["status"] == "PASS" and primary_ok and family_ok and preservation_ok
    result = {
        "status": "CPU_POST_FREEZE_DETERMINISTIC_SCORE_COMPLETE", "binding_sha256": sha(binding_path), "raw_freeze_sha256": sha(output / "RAW_FREEZE_RECEIPT.json"), "raw_artifacts_complete": True,
        "batch1_validity": b1, "by_artifact": by_artifact, "paired_primary_ood_treatment_minus_control": primary, "paired_ood_by_family": by_family,
        "protected_retention_each_arm_minus_v7": preservation, "train_fit": train_fit, "composition_sentinel": composition,
        "promotion": {"decision": "PROMOTE" if promoted else "DO_NOT_PROMOTE", "primary_rule_pass": primary_ok, "family_harm_rule_pass": family_ok, "preservation_rule_pass": preservation_ok, "batch1_rule_pass": b1["status"] == "PASS"},
        "targets_opened_after_raw_freeze": True,
    }
    write(output / "SCORE_RECEIPT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--binding", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(score(args.binding, args.output_root), sort_keys=True))
