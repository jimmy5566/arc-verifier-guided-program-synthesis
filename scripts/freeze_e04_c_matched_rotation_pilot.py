"""Freeze target-blind TRAIN-only paired inputs for the E04-C repair pilot.

This constructor deliberately imports neither torch nor a tokenizer.  It only
materializes the two possible supervision arms from the already-frozen E04
TRAIN generator.  Exact tokenizer equality, optimizer budget and protected
replay remain prelaunch gates; this file cannot authorize execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from scripts import e04_orientation_marker_counterfactual_v2 as e04

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e04_c_matched_fixed_turn_rotation_repair_pilot_v1"
TREATMENT = "FIXED_TURN_ROTATION_CONTROL"
CONTROL = "NO_TRANSFORM_RETENTION_CONTROL"


class FreezeFailure(RuntimeError):
    pass


def canon(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(canon(value)).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        handle.write(data)
        tmp = Path(handle.name)
    tmp.replace(path)


def complete_task(base: dict, turn: int, condition: str) -> dict:
    """Build a native TRAIN task without touching validation-sidecar rows."""
    prompt = e04.task_prompt(base, turn, condition)
    return {
        "train": prompt["train"],
        "test": [{"input": prompt["test"][0]["input"], "output": e04.target_grid(base, turn, condition)}],
    }


def pairs() -> list[dict]:
    result: list[dict] = []
    for base in e04.canonical_bases("TRAIN"):
        for turn in e04.TURNS:
            pair_id = digest({"canonical_base_id": base["canonical_base_id"], "turn": turn})
            treatment = complete_task(base, turn, TREATMENT)
            control = complete_task(base, turn, CONTROL)
            result.append(
                {
                    "pair_id": pair_id,
                    "canonical_base_id": base["canonical_base_id"],
                    "train_cell": {"shape_orientation": base["shape_orientation"], "reference_relation": base["reference_relation"]},
                    "turn": turn,
                    "treatment": treatment,
                    "control": control,
                    "treatment_task_sha256": digest(treatment),
                    "control_task_sha256": digest(control),
                }
            )
    return result


def validate(rows: list[dict]) -> None:
    if len(rows) != 288 * 4 or len({row["pair_id"] for row in rows}) != len(rows):
        raise FreezeFailure("E04_C_PAIR_COUNT")
    cells: dict[tuple[int, str], set[str]] = {}
    for row in rows:
        treatment, control = row["treatment"], row["control"]
        if treatment["test"][0]["input"] != control["test"][0]["input"]:
            raise FreezeFailure("E04_C_QUERY_INPUT_NOT_MATCHED")
        if len(treatment["train"]) != len(control["train"]) != 4:
            raise FreezeFailure("E04_C_ROLE_SLOT_MISMATCH")
        if treatment["test"][0]["output"] == control["test"][0]["output"]:
            raise FreezeFailure("E04_C_ROTATION_CONTROL_COLLISION")
        cell = row["train_cell"]
        if e04.RELATIONS[cell["shape_orientation"]] == cell["reference_relation"]:
            raise FreezeFailure("E04_C_VALIDATION_CELL_EXPOSURE")
        cells.setdefault((cell["shape_orientation"], cell["reference_relation"]), set()).add(row["canonical_base_id"])
    if len(cells) != 12 or any(len(ids) != 24 for ids in cells.values()):
        raise FreezeFailure("E04_C_TRAIN_CELL_COVERAGE")


def freeze(destination: Path = OUT) -> dict:
    if destination.exists():
        raise FreezeFailure("E04_C_REFUSE_OVERWRITE")
    rows = pairs()
    validate(rows)
    destination.mkdir(parents=True)
    treatment = [{k: row[k] for k in ("pair_id", "canonical_base_id", "train_cell", "turn", "treatment", "treatment_task_sha256")} for row in rows]
    control = [{k: row[k] for k in ("pair_id", "canonical_base_id", "train_cell", "turn", "control", "control_task_sha256")} for row in rows]
    atomic_write(destination / "TREATMENT_FIXED_TURN_ROTATION_TRAIN.jsonl", b"".join(canon(row) + b"\n" for row in treatment))
    atomic_write(destination / "CONTROL_NO_TRANSFORM_TRAIN.jsonl", b"".join(canon(row) + b"\n" for row in control))
    manifest = {
        "schema_version": 1,
        "protocol_id": "E04_C_MATCHED_FIXED_TURN_ROTATION_REPAIR_PILOT_V1",
        "status": "CPU_ONLY_TRAIN_ONLY_MATCHED_COHORT_FROZEN",
        "treatment_condition": TREATMENT,
        "control_condition": CONTROL,
        "pairs": len(rows),
        "training_split_only": True,
        "validation_rows_read": 0,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "gpu_used": False,
        "optimizer_constructed": False,
        "execution_authorized": False,
        "prelaunch_gates_remaining": [
            "exact_token_and_supervised_token_equality_under_frozen_v7_tokenizer",
            "equal_step_role_slot_length_and_protected_replay_schedule",
            "frozen_paired_uncertainty_minimum_effect_retention_runtime_and_cost_rules",
            "new_director_prelaunch_review",
        ],
        "treatment_sha256": sha(destination / "TREATMENT_FIXED_TURN_ROTATION_TRAIN.jsonl"),
        "control_sha256": sha(destination / "CONTROL_NO_TRANSFORM_TRAIN.jsonl"),
        "generator_source_sha256": sha(Path(e04.__file__)),
    }
    atomic_write(destination / "MANIFEST.json", json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.out), sort_keys=True))


if __name__ == "__main__":
    main()
