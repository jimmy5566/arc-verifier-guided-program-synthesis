"""Freeze E4I matched Inside/Contains curricula and fresh target-blind panels."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.e4g_inside_contains_atomic import task as legacy_task
from scripts.e4h_inside_contains_counterfactual_ood import semantic_roles, target_from_roles
from training_data.pipeline import task_to_sample

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1"
E4G = BASE / "e4g_inside_contains_atomic_plasticity_v1"
E4H = BASE / "e4h_inside_contains_counterfactual_ood_v1"
OUT = BASE / "e4i_inside_contains_deconfounded_order_balanced_training_v1"
GRID_SIZE = 22
TRAIN_COLOR_PAIRS = ((1, 6), (6, 1), (2, 5), (5, 2), (3, 7), (7, 3))
OOD_COLOR_PAIRS = ((1, 7), (7, 1), (2, 6), (6, 2), (3, 5), (5, 3))
OOD_SHAPES = ((5, 8), (8, 5), (6, 7), (7, 6))
OOD_LAYOUTS = ((2, 2), (2, 12), (8, 2), (8, 12))


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canon(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canon(value) + b"\n")


def blind(task):
    value = copy.deepcopy(task)
    value["test"][0].pop("output")
    return value


def with_source(row):
    task = copy.deepcopy(row["task"])
    nested = task.get("source_id")
    enclosing = row.get("source_id")
    if nested is not None and enclosing is not None and nested != enclosing:
        raise ValueError("SOURCE_ID_CONFLICT")
    resolved = nested if nested is not None else enclosing
    if not isinstance(resolved, str) or not resolved:
        raise ValueError("SOURCE_ID_MISSING")
    task["source_id"] = resolved
    return task


def treatment_spec(index):
    direction = "INSIDE" if index % 2 == 0 else "CONTAINS"
    size = 3 + ((index // 2) % 3)
    layout = (0, 3)[(index // 6) % 2]
    frame, inner = TRAIN_COLOR_PAIRS[index // 8]
    order = "A" if (index // 2) % 2 == 0 else "B"
    return direction, size, layout, frame, inner, order


def build_schedules():
    control = json.loads((E4G / "TRAINING_SCHEDULE.json").read_text())["episodes"]
    if len(control) != 384:
        raise ValueError("CONTROL_SLOTS")
    treatment = []
    primary_index = 0
    for row in control:
        item = copy.deepcopy(row)
        if row["role"] == "PRIMARY_CONNECTED_COMPONENT_COUNT":
            direction, size, layout, frame, inner, order = treatment_spec(primary_index)
            task, _ = legacy_task(direction, "SINGLE_FRAME", size, layout, frame, inner, order)
            item["task"] = task
            item["source_id"] = f"E4I_DECONFOUNDED_INSIDE_CONTAINS:{row['slot']}"
            item["structural_family"] = "INSIDE_CONTAINS_DECONFOUNDED_ORDER_BALANCED"
            primary_index += 1
        treatment.append(item)
    if primary_index != 48:
        raise ValueError("TREATMENT_PRIMARY_SLOTS")
    for left, right in zip(control, treatment, strict=True):
        if left["role"] != "PRIMARY_CONNECTED_COMPONENT_COUNT" and digest(left) != digest(right):
            raise ValueError("PROTECTED_SLOT_CHANGED")
        left_sample = task_to_sample(with_source(left))
        right_sample = task_to_sample(with_source(right))
        signature_left = (left_sample["sequence_length"], left_sample["assistant_token_count"])
        signature_right = (right_sample["sequence_length"], right_sample["assistant_token_count"])
        if signature_left != signature_right:
            raise ValueError("TOKEN_BUDGET_MISMATCH")
    return control, treatment


def train_fit_records():
    records = []
    targets = {}
    for index in range(48):
        direction, size, layout, frame, inner, _ = treatment_spec(index)
        canonical = f"E4I_TRAIN_{index:02d}"
        for order in ("A", "B"):
            full, _ = legacy_task(direction, "SINGLE_FRAME", size, layout, frame, inner, order)
            target = full["test"][0]["output"]
            record_id = f"TRAIN_FIT:{canonical}:{order}"
            records.append(
                {
                    "record_id": record_id,
                    "canonical_unit_id": canonical,
                    "demo_order": order,
                    "input_task": blind(full),
                    "target_sha256": digest(target),
                    "construct": {
                        "direction": direction,
                        "frame_color": frame,
                        "inner_color": inner,
                        "numeric_order": "FRAME_LT_INNER" if frame < inner else "FRAME_GT_INNER",
                        "size": size,
                        "layout": layout,
                    },
                }
            )
            targets[record_id] = target
    return records, targets


def ring_cells(top, left, height, width):
    cells = set()
    for col in range(left, left + width):
        cells.add((top, col))
        cells.add((top + height - 1, col))
    for row in range(top, top + height):
        cells.add((row, left))
        cells.add((row, left + width - 1))
    return cells


def put(grid, row, col, color):
    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
        raise ValueError("GRID_BOUNDS")
    if grid[row][col] not in (0, color):
        raise ValueError("GRID_COLLISION")
    grid[row][col] = color


def place_cluster(grid, color, count, left):
    placed = 0
    for row in range(17, 22):
        columns = range(left, left + 5) if row % 2 else range(left + 4, left - 1, -1)
        for col in columns:
            if placed == count:
                return
            put(grid, row, col, color)
            placed += 1
    if placed != count:
        raise ValueError("CLUSTER_CAPACITY")


def render_ood(direction, frame, inner, shape_index, layout_index):
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    grid[0][0] = 8 if direction == "INSIDE" else 9
    height, width = OOD_SHAPES[shape_index]
    top, left = OOD_LAYOUTS[layout_index]
    border = ring_cells(top, left, height, width)
    for row, col in border:
        put(grid, row, col, frame)
    put(grid, top + height // 2, left + width // 2, inner)
    place_cluster(grid, inner, len(border), 1 if layout_index % 2 == 0 else 16)
    put(grid, 21, 11, frame)
    roles = semantic_roles(grid)
    if roles["frame"] != frame or roles["inner"] != inner:
        raise ValueError("SPATIAL_ROLE_ORACLE")
    counts = Counter(value for row in grid for value in row if value in (frame, inner))
    if counts[frame] != counts[inner]:
        raise ValueError("COLOR_COUNT_BALANCE")
    return grid


def ood_target(grid):
    marker = grid[0][0]
    direction = "INSIDE" if marker == 8 else "CONTAINS" if marker == 9 else None
    if direction is None:
        raise ValueError("MARKER")
    roles = semantic_roles(grid)
    return target_from_roles(direction, roles["frame"], roles["inner"])


def fixed_demos():
    specs = (("CONTAINS", 4, 1, 0, 0), ("INSIDE", 6, 3, 1, 1))
    result = {}
    for direction, frame, inner, shape, layout in specs:
        grid = render_ood(direction, frame, inner, shape, layout)
        result[direction] = {"input": grid, "output": ood_target(grid)}
    return result


def ood_task(direction, frame, inner, shape_index, layout_index, order):
    query = render_ood(direction, frame, inner, shape_index, layout_index)
    demos = fixed_demos()
    other = "CONTAINS" if direction == "INSIDE" else "INSIDE"
    sequence = [copy.deepcopy(demos[direction]), copy.deepcopy(demos[other])]
    if order == "B":
        sequence.reverse()
    return {"train": sequence, "test": [{"input": query}]}, ood_target(query)


def ood_records():
    records = []
    targets = {}
    for pair_index, (frame, inner) in enumerate(OOD_COLOR_PAIRS):
        for repeat in range(8):
            direction = "INSIDE" if repeat % 2 == 0 else "CONTAINS"
            shape = repeat // 2
            layout = (pair_index * 2 + repeat) % 4
            canonical = f"E4I_OOD_{pair_index:02d}_{repeat:02d}"
            for order in ("A", "B"):
                prompt, target = ood_task(direction, frame, inner, shape, layout, order)
                record_id = f"OOD:{canonical}:{order}"
                records.append(
                    {
                        "record_id": record_id,
                        "canonical_unit_id": canonical,
                        "demo_order": order,
                        "input_task": prompt,
                        "target_sha256": digest(target),
                        "construct": {
                            "direction": direction,
                            "frame_color": frame,
                            "inner_color": inner,
                            "numeric_order": "FRAME_LT_INNER" if frame < inner else "FRAME_GT_INNER",
                            "shape_index": shape,
                            "layout_index": layout,
                            "equal_global_color_counts": True,
                            "two_components_per_color": True,
                        },
                    }
                )
                targets[record_id] = target
    return records, targets


def validate_panels(train_records, train_targets, ood_rows, ood_targets):
    for label, records, targets in (
        ("TRAIN", train_records, train_targets),
        ("OOD", ood_rows, ood_targets),
    ):
        if len(records) != 96 or len(targets) != 96:
            raise ValueError(label + "_COUNT")
        units = {}
        for row in records:
            units.setdefault(row["canonical_unit_id"], []).append(row)
            if "output" in row["input_task"]["test"][0]:
                raise ValueError(label + "_TARGET_LEAK")
            if digest(targets[row["record_id"]]) != row["target_sha256"]:
                raise ValueError(label + "_COMMITMENT")
        if len(units) != 48 or any({x["demo_order"] for x in rows} != {"A", "B"} for rows in units.values()):
            raise ValueError(label + "_PAIRING")
    canonical = [row for row in ood_rows if row["demo_order"] == "A"]
    if Counter(row["construct"]["numeric_order"] for row in canonical) != {"FRAME_LT_INNER": 24, "FRAME_GT_INNER": 24}:
        raise ValueError("OOD_NUMERIC_ORDER")
    if Counter(row["construct"]["direction"] for row in canonical) != {"INSIDE": 24, "CONTAINS": 24}:
        raise ValueError("OOD_DIRECTION")
    old = json.loads((E4H / "COUNTERFACTUAL_OOD_INPUTS_AND_COMMITMENTS.json").read_text())["records"]
    if {row["query_input_sha256"] for row in old} & {digest(row["input_task"]["test"][0]["input"]) for row in ood_rows}:
        raise ValueError("E4H_QUERY_OVERLAP")
    return {
        "train_fit_canonical_units": 48,
        "ood_canonical_units": 48,
        "ood_prompt_records": 96,
        "ood_numeric_order": {"FRAME_LT_INNER": 24, "FRAME_GT_INNER": 24},
        "ood_direction": {"INSIDE": 24, "CONTAINS": 24},
        "e4h_query_overlap": 0,
        "target_blind": True,
        "semantic_oracle": "SPATIAL_RING_CONTAINMENT_WITH_EQUAL_COLOR_COUNTS",
    }


def build():
    control, treatment = build_schedules()
    train_rows, train_targets = train_fit_records()
    ood_rows, targets = ood_records()
    audit = validate_panels(train_rows, train_targets, ood_rows, targets)
    refs = json.loads((E4G / "PROTECTION_PANEL_REFERENCES.json").read_text())
    target_commitment = digest({**train_targets, **targets})
    write(OUT / "CONTROL_TRAINING_SCHEDULE.json", {"schema_version": 1, "episodes": control})
    write(OUT / "TREATMENT_TRAINING_SCHEDULE.json", {"schema_version": 1, "episodes": treatment})
    write(OUT / "TRAIN_FIT_INPUTS_AND_COMMITMENTS.json", {"schema_version": 1, "records": train_rows})
    write(OUT / "FRESH_OOD_INPUTS_AND_COMMITMENTS.json", {"schema_version": 1, "records": ood_rows})
    write(OUT / "PROTECTION_PANEL_REFERENCES.json", refs)
    protocol = {
        "schema_version": 1,
        "experiment_id": "E4I_MATCHED_DECONFOUNDED_ORDER_BALANCED_INSIDE_CONTAINS_TRAINING_V1",
        "hypothesis": "Counterbalancing demo order and color roles in Inside/Contains primary training improves genuine spatial-relation transfer relative to the shortcut-confounded E4G curriculum.",
        "arms": {
            "CONTROL": "Exact V7 plus the original E4G narrow primary curriculum",
            "TREATMENT": "Exact V7 plus an order-balanced and color-role-counterbalanced primary curriculum",
        },
        "fixed": {
            "v7_initialization": "EXACT",
            "optimizer": "PagedAdamW8bit",
            "learning_rate": 0.00005,
            "seed": 20261013,
            "updates": 96,
            "slots_per_update": 4,
            "slots": 384,
            "protected_nonprimary_slots": 336,
            "slot_coefficient": 0.25,
            "decoder": "NVARC_NATIVE_GREEDY",
        },
        "principal_difference": "Only the 48 Inside/Contains primary tasks differ; treatment counterbalances demo order and frame/inner numeric color roles.",
        "primary_endpoint": "TREATMENT_MINUS_CONTROL_FRESH_OOD_BOTH_ORDERS_EXACT_GRID_OVER_48_CANONICAL_UNITS",
        "gates": {
            "ood_delta_min": 6,
            "paired_bootstrap_lower_strictly_gt": 0,
            "treatment_order_invariance_min": 0.8,
            "retention_each_family_delta_min": 0,
            "retention_aggregate_delta_min": 0,
            "composition_delta_min": 0,
        },
        "decision": "REJECT_UNLESS_ALL_GATES_PASS; NO_PROMOTION_WITHOUT_LATER_INDEPENDENT_CONFIRMATION",
        "boundaries": ["RAW_FREEZE_BEFORE_TARGET_DERIVATION", "NO_GOLD_DGOLD_FINAL_AUDIT", "NO_OFFICIAL_SUBMISSION"],
    }
    write(OUT / "PROTOCOL.json", protocol)
    receipt = {
        "schema_version": 1,
        "status": "PASS_E4I_CPU_FREEZE_NO_MODEL_NO_GPU_NO_TARGET_SIDECAR",
        "audit": audit,
        "control_schedule_sha256": file_sha(OUT / "CONTROL_TRAINING_SCHEDULE.json"),
        "treatment_schedule_sha256": file_sha(OUT / "TREATMENT_TRAINING_SCHEDULE.json"),
        "train_fit_sha256": file_sha(OUT / "TRAIN_FIT_INPUTS_AND_COMMITMENTS.json"),
        "fresh_ood_sha256": file_sha(OUT / "FRESH_OOD_INPUTS_AND_COMMITMENTS.json"),
        "protection_references_sha256": file_sha(OUT / "PROTECTION_PANEL_REFERENCES.json"),
        "protocol_sha256": file_sha(OUT / "PROTOCOL.json"),
        "target_map_sha256_commitment": target_commitment,
        "protected_slots_byte_identical": 336,
        "primary_slots_per_arm": 48,
        "per_slot_sequence_and_supervised_token_counts_matched": 384,
        "targets_written": False,
    }
    write(OUT / "CPU_FREEZE_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
