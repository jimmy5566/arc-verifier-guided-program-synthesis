"""E4H: counterfactual Inside/Contains OOD freeze and E4G shortcut audit.

This module is CPU-only.  It creates target-blind prompts plus SHA256 target
commitments.  The actual target sidecar is generated only after raw model
predictions have been frozen by the execution worker.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1"
E4G = BASE / "e4g_inside_contains_atomic_plasticity_v1"
OUT = BASE / "e4h_inside_contains_counterfactual_ood_v1"
GRID_SIZE = 21
QUERY_COLOR_PAIRS = ((1, 6), (6, 1), (2, 5), (5, 2), (3, 7), (7, 3))
OLD_COLOR_PAIRS = ((1, 2), (4, 6))
SHAPES = ((5, 6), (6, 5), (5, 7), (7, 5))
LAYOUTS = ((2, 2), (2, 10), (7, 2), (7, 10))


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


def blank():
    return [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


def put(grid, row, col, color):
    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
        raise ValueError("GRID_BOUNDS")
    if grid[row][col] not in (0, color):
        raise ValueError("GRID_COLLISION")
    grid[row][col] = color


def ring_cells(top, left, height, width):
    cells = set()
    for col in range(left, left + width):
        cells.add((top, col))
        cells.add((top + height - 1, col))
    for row in range(top, top + height):
        cells.add((row, left))
        cells.add((row, left + width - 1))
    return cells


def connected_components(grid, color):
    unseen = {(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == color}
    result = []
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            for nxt in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if nxt in unseen:
                    unseen.remove(nxt)
                    component.add(nxt)
                    queue.append(nxt)
        result.append(component)
    return result


def semantic_roles(grid):
    """Recover frame/inner roles from spatial containment, not color values/counts."""
    colors = sorted({value for row in grid[1:] for value in row if value})
    if len(colors) != 2:
        raise ValueError("EXPECTED_TWO_OBJECT_COLORS")
    candidates = []
    for color in colors:
        other = colors[1] if color == colors[0] else colors[0]
        for component in connected_components(grid, color):
            if len(component) < 8:
                continue
            rows = [row for row, _ in component]
            cols = [col for _, col in component]
            top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
            expected = ring_cells(top, left, bottom - top + 1, right - left + 1)
            if component != expected:
                continue
            enclosed = [
                (row, col)
                for row in range(top + 1, bottom)
                for col in range(left + 1, right)
                if grid[row][col] == other
            ]
            if enclosed:
                candidates.append((color, other, (top, bottom, left, right)))
    if len(candidates) != 1:
        raise ValueError("UNIQUE_CONTAINMENT_RELATION_REQUIRED")
    frame, inner, bounds = candidates[0]
    return {"frame": frame, "inner": inner, "bounds": bounds}


def target_from_roles(direction, frame, inner):
    if direction == "INSIDE":
        return [[0, 0, 0], [0, inner, 0], [0, 0, 0]]
    if direction == "CONTAINS":
        return [[frame, frame, frame], [frame, 0, frame], [frame, frame, frame]]
    raise ValueError("DIRECTION")


def oracle_target(grid):
    marker = grid[0][0]
    direction = "INSIDE" if marker == 8 else "CONTAINS" if marker == 9 else None
    if direction is None:
        raise ValueError("MARKER")
    roles = semantic_roles(grid)
    return target_from_roles(direction, roles["frame"], roles["inner"])


def _place_cluster(grid, color, count, left):
    placed = 0
    for row in range(16, 20):
        cols = range(left, left + 5) if (row - 16) % 2 == 0 else range(left + 4, left - 1, -1)
        for col in cols:
            if placed >= count:
                return
            put(grid, row, col, color)
            placed += 1
    if placed != count:
        raise ValueError("DISTRACTOR_CAPACITY")


def render(direction, frame, inner, shape_index, layout_index):
    grid = blank()
    grid[0][0] = 8 if direction == "INSIDE" else 9
    height, width = SHAPES[shape_index]
    top, left = LAYOUTS[layout_index]
    border = ring_cells(top, left, height, width)
    for row, col in border:
        put(grid, row, col, frame)
    put(grid, top + height // 2, left + width // 2, inner)
    # Equalize global color counts and component counts.  Both colors have a
    # large component of |border| cells plus one singleton.  Only containment
    # identifies which color is the enclosing object and which is enclosed.
    cluster_left = 1 if layout_index % 2 else 15
    _place_cluster(grid, inner, len(border), cluster_left)
    put(grid, 19, 10, frame)
    roles = semantic_roles(grid)
    if roles["frame"] != frame or roles["inner"] != inner:
        raise ValueError("SEMANTIC_ORACLE_ROLE_MISMATCH")
    counts = Counter(value for row in grid for value in row if value in (frame, inner))
    if counts[frame] != counts[inner]:
        raise ValueError("COLOR_COUNT_NOT_EQUAL")
    return grid


def fixed_demonstrations():
    specs = (
        ("INSIDE", 4, 3, 0, 0),
        ("CONTAINS", 5, 1, 1, 1),
    )
    result = {}
    for direction, frame, inner, shape, layout in specs:
        grid = render(direction, frame, inner, shape, layout)
        result[direction] = {"input": grid, "output": oracle_target(grid)}
    return result


def task(direction, frame, inner, shape_index, layout_index, demo_order):
    query = render(direction, frame, inner, shape_index, layout_index)
    demos = fixed_demonstrations()
    other = "CONTAINS" if direction == "INSIDE" else "INSIDE"
    ordered = [copy.deepcopy(demos[direction]), copy.deepcopy(demos[other])]
    if demo_order == "B":
        ordered.reverse()
    return {"train": ordered, "test": [{"input": query}]}, oracle_target(query)


def build_panel():
    records = []
    targets = {}
    for pair_index, (frame, inner) in enumerate(QUERY_COLOR_PAIRS):
        for repeat in range(8):
            direction = "INSIDE" if repeat % 2 == 0 else "CONTAINS"
            shape_index = repeat // 2
            layout_index = (pair_index + repeat) % 4
            canonical_id = f"E4H_CF_{pair_index:02d}_{repeat:02d}"
            for order in ("A", "B"):
                prompt, target = task(direction, frame, inner, shape_index, layout_index, order)
                record_id = f"OOD:{canonical_id}:{order}"
                if any(demo["input"] == prompt["test"][0]["input"] for demo in prompt["train"]):
                    raise ValueError("QUERY_DEMO_DUPLICATION")
                records.append(
                    {
                        "record_id": record_id,
                        "canonical_unit_id": canonical_id,
                        "demo_order": order,
                        "input_task": prompt,
                        "target_sha256": digest(target),
                        "query_input_sha256": digest(prompt["test"][0]["input"]),
                        "prompt_sha256": digest(prompt),
                        "construct": {
                            "direction": direction,
                            "frame_color": frame,
                            "inner_color": inner,
                            "numeric_order": "FRAME_LT_INNER" if frame < inner else "FRAME_GT_INNER",
                            "shape_index": shape_index,
                            "layout_index": layout_index,
                            "equal_global_color_counts": True,
                            "two_components_per_color": True,
                        },
                    }
                )
                targets[record_id] = target
    return records, targets


def audit_e4g_shortcuts():
    schedule = json.loads((E4G / "TRAINING_SCHEDULE.json").read_text())["episodes"]
    primary = [row for row in schedule if row["role"] == "PRIMARY_CONNECTED_COMPONENT_COUNT"]
    training_demo1_query_match = sum(
        row["task"]["train"][0]["input"][0][0] == row["task"]["test"][0]["input"][0][0]
        for row in primary
    )
    ood = json.loads((E4G / "STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json").read_text())["records"]
    order_match = Counter()
    color_pairs = Counter()
    frame_lt_inner = 0
    numeric_shortcut_correct = 0
    for row in ood:
        query_marker = row["input_task"]["test"][0]["input"][0][0]
        demo1_marker = row["input_task"]["train"][0]["input"][0][0]
        order_match[row["demo_order"]] += int(query_marker == demo1_marker)
        construct = row["construct"]
        frame, inner = construct["frame_color"], construct["inner_color"]
        color_pairs[(frame, inner)] += 1
        frame_lt_inner += int(frame < inner)
        predicted = target_from_roles(construct["direction"], min(frame, inner), max(frame, inner))
        numeric_shortcut_correct += int(digest(predicted) == row["target_sha256"])
    return {
        "schema_version": 1,
        "classification": "CONFIRMED_CONSTRUCT_SHORTCUTS_NOT_RETROSPECTIVE_GATE_CHANGE",
        "original_joint_decision": "REJECT",
        "training_primary_tasks": len(primary),
        "training_demo1_matches_query_operation": training_demo1_query_match,
        "training_demo_orders": {"A": len(primary), "B": 0},
        "ood_records": len(ood),
        "ood_demo1_query_match": {"A": order_match["A"], "B": order_match["B"]},
        "ood_color_pairs": {f"{a},{b}": count for (a, b), count in sorted(color_pairs.items())},
        "ood_frame_color_lt_inner_color": frame_lt_inner,
        "marker_plus_numeric_order_shortcut_exact": numeric_shortcut_correct,
        "demo1_operation_shortcut": {"A_exact": 48, "B_exact": 0, "both_orders_canonical": 0},
        "causal_limit": "The frozen panel does not identify spatial relationship reasoning because marker plus numeric color order reconstructs every target.",
    }


def validate_panel(records, targets):
    if len(records) != 96 or len(targets) != 96:
        raise ValueError("EXPECTED_96_RECORDS")
    units = {}
    for row in records:
        if "output" in row["input_task"]["test"][0]:
            raise ValueError("TARGET_LEAK")
        units.setdefault(row["canonical_unit_id"], []).append(row)
        if digest(targets[row["record_id"]]) != row["target_sha256"]:
            raise ValueError("TARGET_COMMITMENT")
        if oracle_target(row["input_task"]["test"][0]["input"]) != targets[row["record_id"]]:
            raise ValueError("ORACLE")
    if len(units) != 48 or any({row["demo_order"] for row in rows} != {"A", "B"} for rows in units.values()):
        raise ValueError("PAIRED_ORDERS")
    canonical = [rows[0] for rows in units.values()]
    counts = Counter(row["construct"]["numeric_order"] for row in canonical)
    directions = Counter(row["construct"]["direction"] for row in canonical)
    pairs = Counter((row["construct"]["frame_color"], row["construct"]["inner_color"]) for row in canonical)
    if counts != {"FRAME_LT_INNER": 24, "FRAME_GT_INNER": 24}:
        raise ValueError("NUMERIC_ORDER_BALANCE")
    if directions != {"INSIDE": 24, "CONTAINS": 24}:
        raise ValueError("DIRECTION_BALANCE")
    if set(pairs) & set(OLD_COLOR_PAIRS) or set(pairs.values()) != {8}:
        raise ValueError("COLOR_PAIR_BALANCE")
    numeric_shortcut = 0
    for row in records:
        construct = row["construct"]
        guess = target_from_roles(
            construct["direction"],
            min(construct["frame_color"], construct["inner_color"]),
            max(construct["frame_color"], construct["inner_color"]),
        )
        numeric_shortcut += int(guess == targets[row["record_id"]])
    if numeric_shortcut != 48:
        raise ValueError("NUMERIC_SHORTCUT_NOT_CHANCE")
    return {
        "canonical_units": 48,
        "prompt_records": 96,
        "directions": dict(directions),
        "numeric_order": dict(counts),
        "query_color_pairs": {f"{a},{b}": count for (a, b), count in sorted(pairs.items())},
        "marker_plus_numeric_order_shortcut_exact": numeric_shortcut,
        "global_color_count_shortcut": "NON_IDENTIFYING_EQUAL_COUNTS_96_OF_96",
        "target_blind": True,
        "fixed_query_independent_demonstrations": True,
        "oracle": "CONNECTED_COMPONENT_RECTANGULAR_RING_CONTAINMENT",
    }


def build():
    shortcut = audit_e4g_shortcuts()
    records, targets = build_panel()
    audit = validate_panel(records, targets)
    package = {
        "schema_version": 1,
        "experiment_id": "E4H_INSIDE_CONTAINS_COUNTERFACTUAL_OOD_V1",
        "status": "FROZEN_TARGET_BLIND_CPU_VALIDATED",
        "records": records,
        "audit": audit,
        "target_sidecar_policy": "GENERATE_ONLY_AFTER_COMPLETE_RAW_FREEZE",
    }
    write(OUT / "E4G_SHORTCUT_AUDIT.json", shortcut)
    write(OUT / "COUNTERFACTUAL_OOD_INPUTS_AND_COMMITMENTS.json", package)
    protocol = {
        "schema_version": 1,
        "experiment_id": "E4H_INSIDE_CONTAINS_COUNTERFACTUAL_OOD_V1",
        "scientific_question": "Does the E4G checkpoint retain an Inside/Contains advantage after numeric-color, global-count, and demonstration-order shortcuts are neutralized?",
        "arms": ["V7_NO_UPDATE", "E4G_TRAINED_NO_UPDATE"],
        "training": "FORBIDDEN_INITIAL_DIAGNOSTIC",
        "decoder": "NVARC_NATIVE_GREEDY",
        "requested_batch_size": 32,
        "frozen_fallback_ladder": [32, 16, 8, 4, 1],
        "b1_validation": {
            "canonical_units": [f"E4H_CF_{pair:02d}_{repeat:02d}" for pair, repeat in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5))],
            "require_parser_and_exact_grid_agreement": True,
            "token_identity_required": False,
            "fail_closed_on_material_metric_drift": True,
        },
        "primary_endpoint": "E4G_TRAINED_MINUS_V7_BOTH_ORDERS_EXACT_GRID_OVER_48_CANONICAL_UNITS",
        "secondary_endpoints": [
            "ORDER_A_EXACT_GRID",
            "ORDER_B_EXACT_GRID",
            "BOTH_ANY_ORDER_INVARIANCE",
            "INSIDE_AND_CONTAINS_SUBGROUPS",
            "FRAME_LT_INNER_AND_FRAME_GT_INNER_SUBGROUPS",
        ],
        "uncertainty": "PAIRED_CANONICAL_UNIT_BOOTSTRAP_10000_REPLICATES_SEED_20261014",
        "interpretation": {
            "positive_transfer": "delta_at_least_6_and_bootstrap_lower_gt_0_and_order_invariance_at_least_0.8_with_no_numeric_order_subgroup_reversal",
            "shortcut_explained": "original gain materially attenuates or reverses on FRAME_GT_INNER and/or balanced orders",
            "inconclusive": "batch validation failure, incomplete raw freeze, or material subgroup instability",
        },
        "promotion": "FORBIDDEN_DIAGNOSTIC_ONLY",
        "target_policy": "NO_TARGET_SIDECAR_UNTIL_BOTH_ARMS_COMPLETE_AND_RAW_SHA256_FROZEN",
        "sealed_boundaries": ["GOLD", "DGOLD", "FINAL_AUDIT"],
        "official_submission": "FORBIDDEN",
    }
    write(OUT / "PROTOCOL.json", protocol)
    receipt = {
        "schema_version": 1,
        "status": "PASS_E4H_CPU_FREEZE_NO_MODEL_NO_TARGET_SIDECAR",
        "model_loaded": False,
        "gpu_used": False,
        "targets_opened": False,
        "sidecar_written": False,
        "panel_sha256": file_sha(OUT / "COUNTERFACTUAL_OOD_INPUTS_AND_COMMITMENTS.json"),
        "shortcut_audit_sha256": file_sha(OUT / "E4G_SHORTCUT_AUDIT.json"),
        "protocol_sha256": file_sha(OUT / "PROTOCOL.json"),
        "target_sidecar_sha256_commitment": digest(targets),
        "audit": audit,
        "b1_validation_subset": [f"E4H_CF_{pair:02d}_{repeat:02d}" for pair, repeat in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5))],
        "requested_batch_size": 32,
        "frozen_fallback_ladder": [32, 16, 8, 4, 1],
    }
    write(OUT / "CPU_FREEZE_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-target-sidecar", action="store_true")
    args = parser.parse_args()
    result = build()
    if args.print_target_sidecar:
        _, targets = build_panel()
        print(json.dumps({"targets": targets}, sort_keys=True))
    else:
        print(json.dumps(result, sort_keys=True))
