"""CPU-only E04 V2 matched marker-counterfactual cohort constructor and validator.

This module deliberately has no model, torch, tokenizer, or remote imports.
"""
from __future__ import annotations
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_marker_counterfactual_v2"
DECISION = ROOT / "orchestration/director/responses/DIRECTOR_E04_SCIENTIFIC_RECOVERY_V2_DECISION.json"
V1_DIR = ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_frozen_cohort_v1"
RELATIONS = ("NORTH", "EAST", "SOUTH", "WEST")
LAYOUTS = ("CENTER", "NW", "NE", "SW", "SE", "MID_WEST", "MID_EAST", "MID_NORTH", "MID_SOUTH", "OFFSET_A", "OFFSET_B", "OFFSET_C")
CONDITIONS = ("ROTATION_TARGET", "FIXED_TURN_ROTATION_CONTROL", "MARKER_BINDING_CONTROL", "NO_TRANSFORM_RETENTION_CONTROL")
TURNS = (0, 1, 2, 3)
FORBIDDEN = ("GOLD", "DGOLD", "FINAL_AUDIT")

class V2Failure(RuntimeError):
    pass

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canon(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

def digest(value) -> str:
    return hashlib.sha256(canon(value)).hexdigest()

def color_tuple(index: int) -> tuple[int, int, int, int]:
    """24 four-distinct tuples; each role has every colour 2 or 3 times."""
    if not 0 <= index < 24:
        raise V2Failure("INVALID_COLOR_INDEX")
    return tuple(((index + offset) % 9) + 1 for offset in (0, 2, 4, 6))

def cells(split: str):
    if split not in {"TRAIN", "VALIDATION"}:
        raise V2Failure("INVALID_SPLIT")
    return [(orientation, relation) for orientation, relation in ((o, r) for o in range(4) for r in RELATIONS)
            if (RELATIONS[orientation] == relation) == (split == "VALIDATION")]

def canonical_bases(split: str) -> list[dict]:
    count = 24 if split == "TRAIN" else 12
    rows = []
    for cell_index, (orientation, relation) in enumerate(cells(split)):
        for index in range(count):
            base = {
                "shape_orientation": orientation,
                "reference_relation": relation,
                "topology": "single_4_connected_L_TRIOMINO",
                "layout_template": LAYOUTS[index % len(LAYOUTS)],
                "color_role_tuple": list(color_tuple(index)),
                "instance_index": index,
                "seed": 810_000 + 104_729 * cell_index + 7_919 * index,
            }
            base["canonical_base_id"] = digest(base)
            rows.append(base)
    return rows

def rotate(points, turn: int):
    return [((y, x) if turn == 0 else (x, 2-y) if turn == 1 else (2-y, 2-x) if turn == 2 else (2-x, y)) for y, x in points]

SHAPE = ((0, 0), (1, 0), (2, 0), (2, 1))
POSITIONS = {"CENTER": (3, 3), "NW": (1, 1), "NE": (1, 5), "SW": (5, 1), "SE": (5, 5), "MID_WEST": (3, 1), "MID_EAST": (3, 5), "MID_NORTH": (1, 3), "MID_SOUTH": (5, 3), "OFFSET_A": (2, 2), "OFFSET_B": (2, 4), "OFFSET_C": (4, 2)}
RELATION_MARKERS = {"NORTH": (0, 4), "EAST": (4, 8), "SOUTH": (8, 4), "WEST": (4, 0)}

def input_grid(base: dict, turn: int) -> list[list[int]]:
    bg, shape_color, marker_color, _ = base["color_role_tuple"]
    grid = [[bg] * 9 for _ in range(9)]
    top, left = POSITIONS[base["layout_template"]]
    for y, x in rotate(SHAPE, base["shape_orientation"]):
        grid[top+y][left+x] = shape_color
    relation_y, relation_x = RELATION_MARKERS[base["reference_relation"]]
    grid[relation_y][relation_x] = marker_color
    # A four-position explicit turn marker, spatially isolated from reference marker.
    grid[0][turn] = marker_color
    return grid

def target_grid(base: dict, turn: int, condition: str) -> list[list[int]]:
    bg, _, _, output_color = base["color_role_tuple"]
    if condition == "MARKER_BINDING_CONTROL":
        return [[output_color if position == turn else bg for position in range(4)]]
    effective_turn = turn if condition == "ROTATION_TARGET" else 1 if condition == "FIXED_TURN_ROTATION_CONTROL" else 0
    output = [[bg] * 3 for _ in range(3)]
    for y, x in rotate(SHAPE, (base["shape_orientation"] + effective_turn) % 4):
        output[y][x] = output_color
    return output

def demonstration_base(base: dict, turn: int) -> dict:
    # Same colour-role convention and layout; demonstrations vary only native task factors.
    value = dict(base)
    value["shape_orientation"] = (base["shape_orientation"] + turn + 1) % 4
    value["layout_template"] = LAYOUTS[(LAYOUTS.index(base["layout_template"]) + 3 * (turn + 1)) % len(LAYOUTS)]
    return value

def task_prompt(base: dict, query_turn: int, condition: str) -> dict:
    train = []
    for demo_turn in TURNS:
        demo = demonstration_base(base, demo_turn)
        train.append({"input": input_grid(demo, demo_turn), "output": target_grid(demo, demo_turn, condition)})
    return {"train": train, "test": [{"input": input_grid(base, query_turn)}]}

def task_rows(split: str) -> tuple[list[dict], list[dict]]:
    inputs, targets = [], []
    for base in canonical_bases(split):
        for turn in TURNS:
            for condition in CONDITIONS:
                task_id = digest({"base": base["canonical_base_id"], "turn": turn, "condition": condition})
                prompt = task_prompt(base, turn, condition)
                inputs.append({"task_id": task_id, "prompt": prompt})
                targets.append({"task_id": task_id, "canonical_base_id": base["canonical_base_id"], "split": split,
                                "condition": condition, "control_marker_turn": turn,
                                "target": target_grid(base, turn, condition), "output_color": base["color_role_tuple"][3]})
    return inputs, targets

def validate_bases(train: list[dict], validation: list[dict]) -> None:
    if len(train) != 288 or len(validation) != 48:
        raise V2Failure("CANONICAL_BASE_COUNT")
    if len({row["canonical_base_id"] for row in train + validation}) != 336:
        raise V2Failure("CANONICAL_BASE_DUPLICATE")
    if {tuple((r["shape_orientation"], r["reference_relation"])) for r in train} != set(cells("TRAIN")):
        raise V2Failure("TRAIN_STRUCTURAL_HOLDOUT")
    if {tuple((r["shape_orientation"], r["reference_relation"])) for r in validation} != set(cells("VALIDATION")):
        raise V2Failure("VALIDATION_STRUCTURAL_HOLDOUT")
    for split, rows, expected in (("TRAIN", train, 24), ("VALIDATION", validation, 12)):
        by_cell = defaultdict(list)
        for row in rows:
            if len(set(row["color_role_tuple"])) != 4 or not all(1 <= c <= 9 for c in row["color_role_tuple"]):
                raise V2Failure("INVALID_COLOR_ROLES")
            by_cell[(row["shape_orientation"], row["reference_relation"])].append(row)
        for unit in by_cell.values():
            if len(unit) != expected:
                raise V2Failure("CELL_BASE_COUNT")
            for role in range(4):
                freq = [sum(r["color_role_tuple"][role] == color for r in unit) for color in range(1, 10)]
                if max(freq) - min(freq) > 1:
                    raise V2Failure("ROLE_COLOR_IMBALANCE")
            for layout in LAYOUTS:
                tuples = [tuple(r["color_role_tuple"]) for r in unit if r["layout_template"] == layout]
                if len(tuples) != len(set(tuples)):
                    raise V2Failure("LAYOUT_COLOR_SHORTCUT")

def scrub_query_marker(prompt: dict, bg: int) -> dict:
    value = json.loads(json.dumps(prompt))
    for col in range(4):
        value["test"][0]["input"][0][col] = bg
    return value

def validate_tasks(inputs: list[dict], targets: list[dict], expected_bases: int) -> None:
    if len(inputs) != expected_bases * 16 or len(targets) != len(inputs):
        raise V2Failure("TASK_ROW_COUNT")
    input_by_id = {row["task_id"]: row for row in inputs}
    if len(input_by_id) != len(inputs) or {row["task_id"] for row in targets} != set(input_by_id):
        raise V2Failure("TASK_TARGET_MAPPING")
    by_base_condition = defaultdict(list)
    for target in targets:
        prompt = input_by_id[target["task_id"]]["prompt"]
        if set(prompt) != {"train", "test"} or set(prompt["test"][0]) != {"input"}:
            raise V2Failure("TARGET_LEAKAGE_IN_INPUT_PROMPT")
        serialized = canon(prompt).decode("utf-8")
        if any(word.lower() in serialized.lower() for word in ("condition", "split", "hash", "validation")):
            raise V2Failure("METADATA_LEAKAGE_IN_INPUT_PROMPT")
        output_color = target["output_color"]
        if not all(any(output_color in pixels for pixels in row["output"]) for row in prompt["train"]):
            raise V2Failure("OUTPUT_COLOR_NOT_OBSERVABLE")
        by_base_condition[(target["canonical_base_id"], target["condition"])].append(target)
    for (base_id, condition), group in by_base_condition.items():
        if {row["control_marker_turn"] for row in group} != set(TURNS):
            raise V2Failure("MARKER_COUNTERFACTUAL_QUARTET_MISSING")
        # Input prompts differ in their explicit query marker only; native demonstrations stay fixed.
        signatures = set()
        values = set()
        for row in group:
            prompt = input_by_id[row["task_id"]]["prompt"]
            background = prompt["test"][0]["input"][8][8]
            signatures.add(digest(scrub_query_marker(prompt, background)))
            values.add(digest(row["target"]))
        if len(signatures) != 1:
            raise V2Failure("MARKER_QUARTET_NUISANCE_NOT_IDENTICAL")
        if condition in {"ROTATION_TARGET", "MARKER_BINDING_CONTROL"} and len(values) != 4:
            raise V2Failure("MARKER_BLIND_PREDICTOR_NOT_EXCLUDED")
    # Condition is encoded only by demonstrations; every base has observably distinct native tasks.
    by_base = defaultdict(dict)
    for target in targets:
        prompt = input_by_id[target["task_id"]]["prompt"]
        if target["control_marker_turn"] == 0:
            by_base[target["canonical_base_id"]][target["condition"]] = digest(prompt["train"])
    for demonstrated in by_base.values():
        if set(demonstrated) != set(CONDITIONS) or len(set(demonstrated.values())) != 4:
            raise V2Failure("CONDITION_NATIVE_TASK_UNOBSERVABLE")

def write_json(path: Path, value) -> None:
    path.write_bytes(json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n")

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"".join(canon(row) + b"\n" for row in rows))

def freeze(destination: Path = OUT) -> dict:
    if destination.exists():
        raise V2Failure("REFUSE_OVERWRITE_FROZEN_V2_COHORT")
    if not DECISION.is_file():
        raise V2Failure("DIRECTOR_DECISION_MISSING")
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    if decision.get("response_id") != "DIRECTOR_E04_SCIENTIFIC_RECOVERY_V2_DECISION":
        raise V2Failure("DIRECTOR_DECISION_IDENTITY")
    train, validation = canonical_bases("TRAIN"), canonical_bases("VALIDATION")
    validate_bases(train, validation)
    train_inputs, train_targets = task_rows("TRAIN")
    validation_inputs, validation_targets = task_rows("VALIDATION")
    validate_tasks(train_inputs, train_targets, 288)
    validate_tasks(validation_inputs, validation_targets, 48)
    destination.mkdir(parents=True)
    write_jsonl(destination / "TRAIN_NATIVE_TASK_INPUTS.jsonl", train_inputs)
    write_jsonl(destination / "VALIDATION_INPUT_PROMPTS.jsonl", validation_inputs)
    write_jsonl(destination / "TARGET_SCORER_SIDECAR.jsonl", train_targets + validation_targets)
    write_json(destination / "CANONICAL_BASES.json", {"TRAIN": train, "VALIDATION": validation})
    v1_reference = {"source_commit": "7a54066fd503d36b7154d69ab82e4e2727e335b1", "status": "PRESERVED_BUT_SUPERSEDED_BEFORE_GPU_EXECUTION", "v1_manifest_sha256": sha(V1_DIR / "MANIFEST.json"), "v1_cohort_sha256": sha(V1_DIR / "COHORT.jsonl")}
    write_json(destination / "V1_REFERENCE.json", v1_reference)
    manifest = {
        "protocol_id": "E04_ORIENTATION_V2_MARKER_COUNTERFACTUAL_NATIVE_TASK_COHORT",
        "status": "CPU_ONLY_FROZEN_INPUTS_AND_SEPARATE_TARGET_SIDECAR",
        "director_decision_sha256": sha(DECISION),
        "statistical_unit": "CANONICAL_BASE_TUPLE_MATCHED_COUNTERFACTUAL_SET",
        "canonical_base_counts": {"TRAIN": 288, "VALIDATION": 48},
        "views_per_canonical_base": 16,
        "native_condition_tasks": list(CONDITIONS), "explicit_marker_turns": list(TURNS),
        "validation_task_rows": len(validation_inputs), "all_task_rows": len(train_inputs) + len(validation_inputs),
        "input_only_validation_prompt_sha256": sha(destination / "VALIDATION_INPUT_PROMPTS.jsonl"),
        "target_scorer_sidecar_sha256": sha(destination / "TARGET_SCORER_SIDECAR.jsonl"),
        "minimum_conditional_support_per_reserved_cell": 8,
        "bootstrap": {"seed": 20261010, "replicates": 10000, "unit": "canonical_base_tuple"},
        "protected_retention_required": True,
        "boundaries": ["NO_GOLD", "NO_DGOLD", "NO_FINAL_AUDIT", "NO_MODEL_LOADING", "NO_GPU", "NO_GENERATION", "NO_TRAINING"],
        "v1_reference": v1_reference,
    }
    write_json(destination / "MANIFEST.json", manifest)
    return manifest

def deterministic_regeneration_report(destination: Path = OUT) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        regenerated = Path(tmp) / "regen"
        freeze(regenerated)
        names = ("TRAIN_NATIVE_TASK_INPUTS.jsonl", "VALIDATION_INPUT_PROMPTS.jsonl", "TARGET_SCORER_SIDECAR.jsonl", "CANONICAL_BASES.json", "V1_REFERENCE.json", "MANIFEST.json")
        comparisons = {name: sha(destination / name) == sha(regenerated / name) for name in names}
    report = {"status": "PASS" if all(comparisons.values()) else "FAIL", "all_byte_identical": all(comparisons.values()), "files": comparisons, "serialization": "UTF-8 canonical JSONL with LF"}
    write_json(destination / "DETERMINISTIC_REGENERATION_REPORT.json", report)
    return report

def validate_frozen(destination: Path = OUT) -> dict:
    manifest = json.loads((destination / "MANIFEST.json").read_text(encoding="utf-8"))
    inputs = []
    for name in ("TRAIN_NATIVE_TASK_INPUTS.jsonl", "VALIDATION_INPUT_PROMPTS.jsonl"):
        inputs.extend(json.loads(line) for line in (destination / name).read_text(encoding="utf-8").splitlines())
    targets = [json.loads(line) for line in (destination / "TARGET_SCORER_SIDECAR.jsonl").read_text(encoding="utf-8").splitlines()]
    bases = json.loads((destination / "CANONICAL_BASES.json").read_text(encoding="utf-8"))
    validate_bases(bases["TRAIN"], bases["VALIDATION"])
    validate_tasks(inputs[:4608], targets[:4608], 288)
    validate_tasks(inputs[4608:], targets[4608:], 48)
    if manifest["validation_task_rows"] != 768 or manifest["views_per_canonical_base"] != 16:
        raise V2Failure("MANIFEST_COUNT")
    return {"status": "PASS", "model_loaded": False, "gpu_used": False, "optimizer_steps": 0, "validation_task_rows": 768, "target_blind_prompts": True}

if __name__ == "__main__":
    manifest = freeze()
    report = deterministic_regeneration_report()
    print(json.dumps({"manifest": manifest, "regeneration": report}, sort_keys=True))
