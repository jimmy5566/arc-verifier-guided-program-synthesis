"""CPU-only E4-C audit and a controlled query-local Code-Pair selector.

This module consumes only frozen raw prediction JSONL files and public score
receipts. It deliberately rejects target sidecars and never imports a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.e4b_corrected_code_pair_supervision import count_code_pair_label_indices

FORBIDDEN_PATH_TERMS = ("SIDECAR", "TARGET")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_target_path(path: Path) -> None:
    if any(term in path.name.upper() for term in FORBIDDEN_PATH_TERMS):
        raise ValueError(f"TARGET_REUSE_FORBIDDEN:{path.name}")


def load_raw(path: Path) -> list[dict]:
    _reject_target_path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def arm_delta_summary(control: Iterable[dict], treatment: Iterable[dict]) -> dict:
    control, treatment = list(control), list(treatment)
    if [row["record_id"] for row in control] != [row["record_id"] for row in treatment]:
        raise ValueError("RAW_RECORD_IDENTITY_MISMATCH")
    by_split: dict[str, Counter] = defaultdict(Counter)
    for left, right in zip(control, treatment):
        split = left["split"]
        if split == "RETENTION_SECONDARY":
            split += ":MARKER_BINDING" if "MARKER_BINDING" in left["record_id"] else ":FIXED_TURN_ROTATION"
        out = by_split[split]
        out["records"] += 1
        out["generated_token_ids_changed"] += left["generated_token_ids"] != right["generated_token_ids"]
        out["parsed_grid_changed"] += left["parsed_grid"] != right["parsed_grid"]
        out["control_parser_invalid"] += not left["parser_valid"]
        out["treatment_parser_invalid"] += not right["parser_valid"]
    return {key: dict(value) for key, value in sorted(by_split.items())}


def raw_order_agreement(rows: Iterable[dict]) -> dict:
    pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["split"] != "OOD_INDEPENDENT":
            continue
        ident = row["record_id"]
        if ident.endswith(":ORDER_A"):
            pairs[ident.removesuffix(":ORDER_A")]["A"] = row
        elif ident.endswith(":ORDER_B"):
            pairs[ident.removesuffix(":ORDER_B")]["B"] = row
        else:
            raise ValueError(f"OOD_ORDER_LABEL_INVALID:{ident}")
    if not pairs or any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("OOD_ORDER_PAIR_INCOMPLETE")
    exact_grid = sum(pair["A"]["parsed_grid"] == pair["B"]["parsed_grid"] for pair in pairs.values())
    exact_tokens = sum(pair["A"]["generated_token_ids"] == pair["B"]["generated_token_ids"] for pair in pairs.values())
    return {"canonical_units": len(pairs), "exact_parsed_grid_agreement": exact_grid, "exact_generated_token_agreement": exact_tokens}


def query_local_code_pair_indices(task: dict) -> tuple[int, int]:
    """Return only the final-query Count code pair; demonstrations remain unfocused."""
    indices, _counts = count_code_pair_label_indices(task)
    if len(indices) != 8:
        raise ValueError("E4B_CODE_PAIR_INDEX_COUNT")
    return tuple(indices[-2:])


def query_local_code_pair_weights(labels: list[int], task: dict, multiplier: float = 2.0):
    """Mass-preserving control/treatment weights for the selected final query."""
    focused = query_local_code_pair_indices(task)
    valid = [index for index, value in enumerate(labels) if value != -100]
    if len(focused) != 2 or len(set(focused)) != 2 or any(index not in valid for index in focused):
        raise ValueError("QUERY_LOCAL_CODE_PAIR_INDEX_INVALID")
    if not multiplier > 0 or len(valid) <= len(focused):
        raise ValueError("QUERY_LOCAL_LOSS_MASS_PARAMETERS")
    compensation = (len(valid) - multiplier * len(focused)) / (len(valid) - len(focused))
    if compensation <= 0:
        raise ValueError("QUERY_LOCAL_LOSS_MASS_MISMATCH")
    control = {index: 1.0 for index in valid}
    treatment = {index: (multiplier if index in focused else compensation) for index in valid}
    if abs(sum(control.values()) - sum(treatment.values())) > 1e-12:
        raise ValueError("QUERY_LOCAL_LOSS_MASS_MISMATCH")
    return control, treatment


def _load_score(path: Path) -> dict:
    _reject_target_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run(e4a_dir: Path, e4b_dir: Path, schedule_path: Path, output_dir: Path) -> dict:
    for directory in (e4a_dir, e4b_dir):
        if not directory.is_dir():
            raise ValueError(f"EVIDENCE_DIR_MISSING:{directory}")
    files = {
        "e4a_score": e4a_dir / "SCORE_RECEIPT.json",
        "e4b_score": e4b_dir / "SCORE_RECEIPT.json",
        "e4b_equal_raw": e4b_dir / "EQUAL_SLOT_5E5_RAW.jsonl",
        "e4b_code_pair_raw": e4b_dir / "CODE_PAIR_5E5_RAW.jsonl",
        "e4b_v7_raw": e4b_dir / "V7_RAW.jsonl",
        "schedule": schedule_path,
    }
    if any(not path.is_file() for path in files.values()):
        missing = [name for name, path in files.items() if not path.is_file()]
        raise ValueError(f"REQUIRED_INPUT_MISSING:{','.join(missing)}")
    e4a_score, e4b_score = _load_score(files["e4a_score"]), _load_score(files["e4b_score"])
    equal, code_pair, v7 = load_raw(files["e4b_equal_raw"]), load_raw(files["e4b_code_pair_raw"]), load_raw(files["e4b_v7_raw"])
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    episodes = schedule["episodes"]
    count_tasks = [row["task"] for row in episodes if row["role"] == "PRIMARY_CONNECTED_COMPONENT_COUNT"]
    if len(count_tasks) != 48:
        raise ValueError("PRIMARY_COUNT_SCHEDULE_NOT_48")
    selected = [query_local_code_pair_indices(task) for task in count_tasks]
    if any(len(x) != 2 for x in selected):
        raise ValueError("QUERY_LOCAL_SELECTOR_INVALID")
    result = {
        "status": "PASS_CPU_ONLY_NO_MODEL_NO_TARGET_REUSE",
        "inputs": {name: {"path": str(path).replace("\\", "/"), "sha256": sha256_file(path)} for name, path in files.items()},
        "e4a_recorded_score": {"decision": e4a_score["decision"], "primary": e4a_score["primary_new_independent_ood"], "secondary": e4a_score["secondary_descriptive"]},
        "e4b_recorded_score": {"decision": e4b_score["decision"], "primary": e4b_score["primary_new_independent_ood"], "secondary": e4b_score["secondary_descriptive"]},
        "raw_prediction_audit": {
            "code_pair_vs_equal_slot": arm_delta_summary(equal, code_pair),
            "equal_slot_order_agreement": raw_order_agreement(equal),
            "code_pair_order_agreement": raw_order_agreement(code_pair),
            "v7_order_agreement": raw_order_agreement(v7),
        },
        "controlled_intervention_cpu_validation": {
            "name": "QUERY_LOCAL_CODE_PAIR_MASS_PRESERVING",
            "count_primary_slots": len(count_tasks),
            "focused_tokens_per_slot": 2,
            "all_query_indices_distinct_from_demo_indices": all(set(p[-2:]).isdisjoint(p[:-2]) for p, _ in (count_code_pair_label_indices(t) for t in count_tasks)),
            "model_loaded": False,
            "gpu_used": False,
            "targets_opened": False,
        },
        "interpretation": {
            "established": "E4-B Code-Pair changed 10/48 train raw grids but 15/96 structural-OOD order records; its recorded +2/48 robust-OOD effect was below the causal gate while Marker Binding retention declined by 2/16.",
            "conditional_binding": "SUPPORTED_AS_A_FALSIFIABLE_MECHANISM_NOT_PROVEN: harm localized in scored Marker Binding while raw changes occurred in both protected families.",
            "object_centric_representation": "NOT_ESTABLISHED: train-fit improvement without causal OOD improvement does not demonstrate an object-centric representation.",
            "atomic_to_compositional_transfer": "NOT_ESTABLISHED: composition was unchanged and structural OOD did not meet the gate.",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = output_dir / "FROZEN_PREDICTION_MECHANISM_AUDIT.json"
    audit.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "status": result["status"], "audit_path": str(audit).replace("\\", "/"), "audit_sha256": sha256_file(audit),
        "model_loaded": False, "gpu_used": False, "target_sidecar_opened": False,
    }
    (output_dir / "CPU_VALIDATION_RECEIPT.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e4a-evidence", type=Path, required=True)
    parser.add_argument("--e4b-evidence", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.e4a_evidence, args.e4b_evidence, args.schedule, args.output_dir)
    print(result["status"])


if __name__ == "__main__":
    main()
