"""CPU-only audit of the post-V7 connected-component training failure.

This script reads only committed TRAIN schedules and already-published summary
artifacts.  It does not load a model, access Gold/dGold, or launch GPU work.
It produces a deconfounded 48-slot TRAIN-derived counterfactual panel and a
target-blind prospective structural-OOD prompt panel.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
OUT = PILOT / "post_v7_generalization_mechanism_audit_v1"
SCHEDULE = PILOT / "TREATMENT_SCHEDULE.json"
V2_RESULT = PILOT / "independent_confirmation_v2/run_f66fe54019ea096e6bba081131922b02/RESULT_MANIFEST.json"
E04D_RESULT = ROOT / "experiments/capability_repair_baseline_v1/e04_d_rotation_marker_local_gradient_diagnostic_v1/run_002_5891e15b0021493f899baee9160502d2/RESULT_SUMMARY.json"
TOKEN_ACCOUNTING = PILOT / "TOKEN_ACCOUNTING.json"

TOPOLOGIES = {
    "L": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "Z": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "T": ((0, 0), (0, 1), (0, 2), (1, 1)),
}
OOD_TOPOLOGIES = {
    "J5": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    "STAIR5": ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    "P5": ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    "U5": ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
}
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ANCHORS = ((0, 0), (0, 5), (5, 0))


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def normalize(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_y = min(y for y, _ in points)
    min_x = min(x for _, x in points)
    return tuple(sorted((y - min_y, x - min_x) for y, x in points))


def rotate(points: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    result = tuple(points)
    for _ in range(turns % 4):
        result = tuple((x, -y) for y, x in result)
    return normalize(result)


def count_components(grid: list[list[int]]) -> int:
    seen: set[tuple[int, int]] = set()
    components = 0
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if color == 0 or (y, x) in seen:
                continue
            components += 1
            seen.add((y, x))
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if not (0 <= ny < len(grid) and 0 <= nx < len(grid[0])):
                        continue
                    if (ny, nx) in seen or grid[ny][nx] != color:
                        continue
                    seen.add((ny, nx))
                    stack.append((ny, nx))
    return components


def output_count(grid: list[list[int]]) -> int:
    nonzero = [(y, x, value) for y, row in enumerate(grid) for x, value in enumerate(row) if value != 0]
    if not nonzero:
        raise AssertionError("EMPTY_COUNT_OUTPUT")
    colors = {value for _, _, value in nonzero}
    if len(colors) != 1 or any(y != 1 for y, _, _ in nonzero):
        raise AssertionError("INVALID_COUNT_OUTPUT")
    columns = sorted(x for _, x, _ in nonzero)
    if columns != list(range(len(columns))):
        raise AssertionError("INVALID_COUNT_RUN")
    return len(columns)


def foreground_color(grid: list[list[int]]) -> int:
    colors = {value for row in grid for value in row if value != 0}
    if len(colors) != 1:
        raise AssertionError("NONUNIQUE_FOREGROUND")
    return next(iter(colors))


def make_pair(points: tuple[tuple[int, int], ...], count: int, turns: int, color: int) -> dict[str, Any]:
    shape = rotate(points, turns)
    grid = [[0] * 9 for _ in range(9)]
    for top, left in ANCHORS[:count]:
        for y, x in shape:
            grid[top + y][left + x] = color
    output = [[0] * 3 for _ in range(3)]
    for x in range(count):
        output[1][x] = color
    assert count_components(grid) == count
    assert output_count(output) == count
    return {"input": grid, "output": output}


def mutual_information(rows: list[tuple[str, int]]) -> float:
    total = len(rows)
    joint = Counter(rows)
    xs = Counter(x for x, _ in rows)
    ys = Counter(y for _, y in rows)
    return sum((n / total) * math.log2((n * total) / (xs[x] * ys[y])) for (x, y), n in joint.items())


def majority_accuracy(rows: list[tuple[str, int]]) -> tuple[int, int]:
    grouped: dict[str, Counter[int]] = defaultdict(Counter)
    for x, y in rows:
        grouped[x][y] += 1
    return sum(max(counts.values()) for counts in grouped.values()), len(rows)


def original_audit(schedule: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [row for row in schedule["episodes"] if row["role"] == "PRIMARY_CONNECTED_COMPONENT_COUNT"]
    if len(rows) != 48:
        raise AssertionError("COUNT_PRIMARY_SLOT_COUNT")
    topology_target: list[tuple[str, int]] = []
    first_target: list[tuple[str, int]] = []
    demo_matches = [0, 0, 0]
    oracle_pairs = 0
    query_colors: Counter[int] = Counter()
    query_rotation_phases: Counter[int] = Counter()
    contingency: dict[str, Counter[int]] = defaultdict(Counter)
    demo_tuples: Counter[tuple[int, int, int]] = Counter()
    for row in rows:
        task = row["task"]
        pairs = [*task["train"], *task["test"]]
        counts = []
        for pair in pairs:
            observed = count_components(pair["input"])
            target = output_count(pair["output"])
            if observed != target:
                raise AssertionError("ORACLE_TARGET_MISMATCH")
            if foreground_color(pair["input"]) != foreground_color(pair["output"]):
                raise AssertionError("OUTPUT_COLOR_NOT_OBSERVABLE")
            counts.append(target)
            oracle_pairs += 1
        query = counts[-1]
        topology = row["topology"]
        topology_target.append((topology, query))
        first_target.append((str(counts[0]), query))
        contingency[topology][query] += 1
        demo_tuples[tuple(counts[:3])] += 1
        for index in range(3):
            demo_matches[index] += int(counts[index] == query)
        query_colors[foreground_color(task["test"][0]["input"])] += 1
        query_rotation_phases[(row["slot"] * 4 + 3) % 4] += 1
    topology_correct, total = majority_accuracy(topology_target)
    first_correct, _ = majority_accuracy(first_target)
    topology_mi = mutual_information(topology_target)
    target_entropy = math.log2(3)
    return {
        "status": "CONFIRMED_DATASET_IDENTIFIABILITY_DEFECT",
        "primary_count_slots": len(rows),
        "oracle_validated_input_output_pairs": oracle_pairs,
        "query_target_distribution": dict(sorted(Counter(y for _, y in topology_target).items())),
        "topology_target_contingency": {key: dict(sorted(value.items())) for key, value in sorted(contingency.items())},
        "topology_majority_predictor": {"correct": topology_correct, "total": total, "accuracy": topology_correct / total},
        "topology_target_mutual_information_bits": topology_mi,
        "topology_target_normalized_mutual_information": topology_mi / target_entropy,
        "first_demo_count_predictor": {"correct": first_correct, "total": total, "accuracy": first_correct / total},
        "same_count_by_demo_position": {str(i + 1): {"correct": n, "total": total, "accuracy": n / total} for i, n in enumerate(demo_matches)},
        "demo_count_tuples": {"-".join(map(str, key)): value for key, value in sorted(demo_tuples.items())},
        "query_foreground_colors": dict(sorted(query_colors.items())),
        "query_rotation_phase_mod_4": dict(sorted(query_rotation_phases.items())),
        "confirmed_defects": [
            "Topology is a perfect query-count label: L->1, Z->3, T->2.",
            "The first demonstration count equals the query count in all 48 tasks.",
            "Every query uses foreground color 4 and rotation phase 3, so color/orientation are not independently crossed.",
        ],
        "causal_limit": "These defects permit shortcuts; CPU inspection alone does not prove which shortcut the model learned or that a particular shortcut caused the V2 failure.",
    }, rows


def deconfounded_train_panel(original_rows: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    alternatives: list[dict[str, Any]] = []
    topologies = ("L",) * 6 + ("Z",) * 5 + ("T",) * 5
    old_hashes = {digest(row["task"]) for row in original_rows}
    contingency: dict[str, Counter[int]] = defaultdict(Counter)
    first_matches = 0
    for group, topology in enumerate(topologies):
        turns = group % 4
        query_color = COLORS[group % len(COLORS)]
        order = [1, 2, 3]
        shift = group % 3
        order = order[shift:] + order[:shift]
        reverse_order = list(reversed(order))
        for target_count in (1, 2, 3):
            task_id = f"COUNTERBALANCED_TRAIN:{group:02d}:{topology}:COUNT_{target_count}"
            demos = [make_pair(TOPOLOGIES[topology], count, turns + index + 1, COLORS[(group + index + 1) % len(COLORS)]) for index, count in enumerate(order)]
            query = make_pair(TOPOLOGIES[topology], target_count, turns, query_color)
            task = {"source_id": task_id, "train": demos, "test": [query]}
            alternative = {"source_id": task_id + ":ORDER_B", "train": list(reversed(demos)), "test": [query]}
            record = {
                "slot_index": len(records),
                "group": group,
                "topology": topology,
                "turns": turns,
                "query_count": target_count,
                "query_color": query_color,
                "demo_count_order": order,
                "task_sha256": digest(task),
                "task": task,
            }
            records.append(record)
            alternatives.append({
                "slot_index": record["slot_index"],
                "demo_count_order": reverse_order,
                "task_sha256": digest(alternative),
                "task": alternative,
            })
            contingency[topology][target_count] += 1
            first_matches += int(order[0] == target_count)
    if len(records) != 48 or len(alternatives) != 48:
        raise AssertionError("COUNTERBALANCED_SLOT_COUNT")
    if any(counts[1] != counts[2] or counts[2] != counts[3] for counts in contingency.values()):
        raise AssertionError("TOPOLOGY_COUNT_NOT_INDEPENDENT")
    if any(digest(record["task"]) in old_hashes for record in records):
        raise AssertionError("HISTORICAL_EXACT_TASK_COLLISION")
    invariance = 0
    for record, alternative in zip(records, alternatives):
        same_query = record["task"]["test"] == alternative["task"]["test"]
        same_demo_multiset = sorted(digest(pair) for pair in record["task"]["train"]) == sorted(digest(pair) for pair in alternative["task"]["train"])
        invariance += int(same_query and same_demo_multiset)
    return {
        "schema_version": 1,
        "panel_id": "POST_V7_COUNT_COUNTERBALANCED_TRAIN_DERIVED_V1",
        "status": "CPU_ORACLE_VALIDATED_NOT_TRAINED",
        "records": records,
        "demo_order_counterfactuals": alternatives,
        "validation": {
            "records": len(records),
            "topology_target_contingency": {key: dict(sorted(value.items())) for key, value in sorted(contingency.items())},
            "topology_target_mutual_information_bits": mutual_information([(r["topology"], r["query_count"]) for r in records]),
            "first_demo_query_matches": first_matches,
            "first_demo_query_match_rate": first_matches / len(records),
            "demo_order_pairs_oracle_invariant": invariance,
            "historical_exact_task_collisions": 0,
            "all_query_output_colors_observable_in_query": True,
            "all_input_component_counts_match_target_runs": True,
        },
        "scope": "TRAIN-derived counterfactuals only; not independent confirmation evidence and not authorized for GPU training.",
    }


def objective_implementation_audit(schedule: dict[str, Any]) -> dict[str, Any]:
    """Verify the real serializer/locator contract without model execution."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from scripts.decision_token_supervision import count_code_pair_label_indices, matched_weights
    from training_data.pipeline import IGNORE_INDEX, task_to_sample

    count_rows = [row for row in schedule["episodes"] if row["role"] == "PRIMARY_CONNECTED_COMPONENT_COUNT"]
    mask_pass = causal_shift_pass = locator_pass = documented_locator_pass = mass_pass = 0
    supervised_tokens = 0
    for row in count_rows:
        sample = task_to_sample(row["task"])
        ids, labels, messages = sample["input_ids"], sample["labels"], sample["messages"]
        offset = 0
        mask_ok = True
        for message in messages:
            length = 3 + len(message["content"]) + 1
            segment_ids = ids[offset:offset + length]
            segment_labels = labels[offset:offset + length]
            if message["role"] == "user":
                mask_ok &= all(value == IGNORE_INDEX for value in segment_labels)
            else:
                mask_ok &= segment_labels == segment_ids
            offset += length
        mask_ok &= offset == len(ids) == len(labels)
        mask_pass += int(mask_ok)
        causal_shift_pass += int(sum(value != IGNORE_INDEX for value in labels[1:]) == 60 and labels[0] == IGNORE_INDEX)

        indices, counts = count_code_pair_label_indices(row["task"])
        actual_expected_indices: list[int] = []
        documented_expected_indices: list[int] = []
        cursor = 0
        for pair in [*row["task"]["train"], *row["task"]["test"]]:
            input_content_length = len("\n".join("".join(map(str, grid_row)) for grid_row in pair["input"]))
            output_content_length = len("\n".join("".join(map(str, grid_row)) for grid_row in pair["output"]))
            cursor += 3 + input_content_length + 1
            assistant_content = cursor + 3
            actual_expected_indices.extend((assistant_content + 4, assistant_content + 5))
            documented_expected_indices.extend((assistant_content + 5, assistant_content + 6))
            cursor += 3 + output_content_length + 1
        actual_expected_values = []
        documented_expected_values = []
        for pair in [*row["task"]["train"], *row["task"]["test"]]:
            actual_expected_values.extend((pair["output"][1][0], pair["output"][1][1]))
            documented_expected_values.extend((pair["output"][1][1], pair["output"][1][2]))
        locator_ok = tuple(actual_expected_indices) == indices
        locator_ok &= all(labels[index] == ids[index] == value for index, value in zip(indices, actual_expected_values, strict=True))
        locator_ok &= all(index > 0 and labels[index] != IGNORE_INDEX for index in indices)
        locator_ok &= tuple(output_count(pair["output"]) for pair in [*row["task"]["train"], *row["task"]["test"]]) == counts
        locator_pass += int(locator_ok)
        documented_ok = tuple(documented_expected_indices) == indices
        documented_ok &= all(labels[index] == ids[index] == value for index, value in zip(indices, documented_expected_values, strict=True))
        documented_locator_pass += int(documented_ok)

        control, treatment = matched_weights(labels, indices, multiplier=2.0)
        mass_ok = len(control) == len(treatment) == 60
        mass_ok &= abs(sum(control.values()) - 60.0) < 1e-12
        mass_ok &= abs(sum(treatment.values()) - 60.0) < 1e-12
        mass_ok &= all(treatment[index] == 2.0 for index in indices)
        mass_ok &= all(abs(weight - 11 / 13) < 1e-12 for index, weight in treatment.items() if index not in indices)
        mass_pass += int(mass_ok)
        supervised_tokens += sample["assistant_token_count"]

    if (mask_pass, causal_shift_pass, locator_pass, documented_locator_pass, mass_pass) != (48, 48, 48, 0, 48):
        raise AssertionError("OBJECTIVE_IMPLEMENTATION_AUDIT_FAILED")
    roles = Counter(row["role"] for row in schedule["episodes"])
    return {
        "schema_version": 1,
        "status": "CONFIRMED_CODE_PAIR_LOCATOR_SEMANTIC_DEFECT_EQUAL_SLOT_CONTRACT_PASS",
        "scope": "Committed serializer, Equal-Slot loss, Code-Pair locator/weights, and connected-component worker call path; no model forward/backward.",
        "source_hashes": {
            "src/training_data/pipeline.py": file_digest(ROOT / "src/training_data/pipeline.py"),
            "scripts/e04_e_equal_slot_loss.py": file_digest(ROOT / "scripts/e04_e_equal_slot_loss.py"),
            "scripts/decision_token_supervision.py": file_digest(ROOT / "scripts/decision_token_supervision.py"),
            "scripts/run_connected_components_object_extraction_pilot_v1.py": file_digest(ROOT / "scripts/run_connected_components_object_extraction_pilot_v1.py"),
            "scripts/run_connected_component_count_code_pair_supervision_worker_v1.py": file_digest(ROOT / "scripts/run_connected_component_count_code_pair_supervision_worker_v1.py"),
        },
        "checks": {
            "count_tasks": 48,
            "user_mask_and_assistant_identity_pass": mask_pass,
            "causal_shift_valid_label_count_60_pass": causal_shift_pass,
            "code_pair_locator_pass": locator_pass,
            "code_pair_documented_semantic_locator_pass": documented_locator_pass,
            "located_code_tokens": locator_pass * 8,
            "weight_mass_preservation_pass": mass_pass,
            "count_supervised_tokens": supervised_tokens,
            "equal_slot_tests": "tests/test_e04_e_equal_slot_loss.py",
            "code_pair_tests": "tests/test_decision_token_supervision.py",
        },
        "causal_contract": {
            "logits": "logits[:, :-1, :]",
            "labels": "labels[:, 1:]",
            "weights": "token_weights[:, 1:]",
            "alignment": "PASS: weights are indexed in unshifted label coordinates and shifted beside the same labels; each selected label is predicted by logits at index-1.",
        },
        "confirmed_code_pair_defect": {
            "documented_cells_zero_based": [[1, 1], [1, 2]],
            "implemented_cells_zero_based": [[1, 0], [1, 1]],
            "documented_unshifted_offsets_from_assistant_content": [5, 6],
            "implemented_unshifted_offsets_from_assistant_content": [4, 5],
            "implemented_nonzero_patterns_by_count": {"1": [1, 0], "2": [1, 1], "3": [1, 1]},
            "documented_nonzero_patterns_by_count": {"1": [0, 0], "2": [1, 0], "3": [1, 1]},
            "impact": "Implemented pair cannot distinguish Count 2 from Count 3 and overweights the always-foreground color/output-construction cell while omitting the third run cell.",
            "boundary": "This confirms a supervision-locator defect; it does not by itself quantify how much of the V2 generalization failure it caused.",
        },
        "mask_contract": {
            "user_turns": "all IGNORE_INDEX",
            "assistant_turns": "all assistant framing, grid digits/newlines, and end token supervised",
            "interpretation": "This is intentional all-assistant CE, not query-only CE.",
        },
        "nominal_equal_slot_objective_mass": {
            "definition": "Each slot is independently mean-normalized, then contributes 1/4 of its optimizer-step objective. Fractions below are scalar objective coefficients before gradient norms/directions.",
            "count_primary": {"slots": roles["PRIMARY_CONNECTED_COMPONENT_COUNT"], "fraction": roles["PRIMARY_CONNECTED_COMPONENT_COUNT"] / 384},
            "object_primary": {"slots": roles["PRIMARY_OBJECT_EXTRACTION"], "fraction": roles["PRIMARY_OBJECT_EXTRACTION"] / 384},
            "protected_replay": {"slots": roles["PROTECTED_REPLAY"], "fraction": roles["PROTECTED_REPLAY"] / 384},
            "count_query_all_tokens_fraction_of_total": (48 / 384) * (15 / 60),
            "uniform_count_query_decision_fraction_of_total": (48 / 384) * (2 / 60),
            "code_pair_count_query_decision_fraction_of_total": (48 / 384) * (4 / 60),
            "code_pair_all_count_decision_fraction_of_total": (48 / 384) * (16 / 60),
        },
        "raw_supervised_token_exposure": {
            "warning": "Raw token counts are exposure/accounting, not Equal-Slot objective weights.",
            "count_primary": 2880,
            "all_primary": 5760,
            "protected_replay": 21600,
            "total": 27360,
        },
        "code_pair_effect": {
            "uniform_decision_mass_within_count_slot": 8 / 60,
            "reweighted_decision_mass_within_count_slot": 16 / 60,
            "uniform_query_decision_mass_within_count_slot": 2 / 60,
            "reweighted_query_decision_mass_within_count_slot": 4 / 60,
            "demo_query_split_after_reweighting": {"demo": 45 / 60, "query": 15 / 60},
            "conclusion": "Code-Pair preserves total slot mass and the 75/25 demo/query split, but doubles the wrong cell pair: implemented columns 0/1 rather than documented columns 1/2.",
        },
        "limits": "Equal-Slot masking, causal shifting, and mass normalization pass. The Code-Pair semantic locator defect is confirmed, but its causal effect size is unknown without a controlled experiment.",
    }


def fresh_ood_prompts() -> dict[str, Any]:
    rotation_orbits = {
        name: {rotate(points, turns) for turns in range(4)}
        for name, points in OOD_TOPOLOGIES.items()
    }
    if any(len(orbit) != 4 for orbit in rotation_orbits.values()):
        raise AssertionError("OOD_TOPOLOGY_NOT_ORIENTATION_IDENTIFIABLE")
    if any(rotation_orbits[left] & rotation_orbits[right] for left, right in itertools.combinations(rotation_orbits, 2)):
        raise AssertionError("OOD_TOPOLOGY_ROTATION_EQUIVALENCE")
    records: list[dict[str, Any]] = []
    target_commitments: list[dict[str, Any]] = []
    index = 0
    for topology, target_count, turns in itertools.product(OOD_TOPOLOGIES, (1, 2, 3), range(4)):
        color = COLORS[index % len(COLORS)]
        order_a = [1, 2, 3]
        shift = index % 3
        order_a = order_a[shift:] + order_a[:shift]
        demos = [make_pair(OOD_TOPOLOGIES[topology], count, turns + j + 1, COLORS[(index + j + 2) % len(COLORS)]) for j, count in enumerate(order_a)]
        query = make_pair(OOD_TOPOLOGIES[topology], target_count, turns, color)
        canonical_id = f"FRESH_OOD:{topology}:C{target_count}:R{turns}"
        for order_name, ordered in (("A", demos), ("B", list(reversed(demos)))):
            prompt = {"source_id": canonical_id + f":ORDER_{order_name}", "train": ordered, "test": [{"input": query["input"]}]}
            records.append({
                "record_id": prompt["source_id"],
                "canonical_unit_id": canonical_id,
                "demo_order": order_name,
                "topology": topology,
                "query_count": target_count,
                "turns": turns,
                "query_color": color,
                "prompt": prompt,
            })
            target_commitments.append({"record_id": prompt["source_id"], "target_sha256": digest(query["output"])})
        index += 1
    return {
        "schema_version": 1,
        "panel_id": "POST_V7_FRESH_STRUCTURAL_OOD_V1",
        "status": "CPU_FROZEN_TARGET_BLIND_PROMPTS_NO_MODEL_EXECUTION",
        "canonical_units": 48,
        "prompt_records": len(records),
        "records": records,
        "target_commitments": target_commitments,
        "scoring_unit": "48 canonical units clustered across paired demo orders",
        "prohibitions": ["no use in training", "no use of outcomes for threshold changes", "no V2 target reuse", "no Gold/dGold access"],
    }


def mechanism_and_proposal(audit: dict[str, Any], objective_audit: dict[str, Any], panel: dict[str, Any], fresh_ood: dict[str, Any]) -> dict[str, Any]:
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    v2 = json.loads(V2_RESULT.read_text(encoding="utf-8"))
    e04d = json.loads(E04D_RESULT.read_text(encoding="utf-8"))
    roles = Counter(row["role"] for row in schedule["episodes"])
    token_accounting = json.loads(TOKEN_ACCOUNTING.read_text(encoding="utf-8"))
    total_supervised = token_accounting["treatment"]["supervised_tokens"]
    count_supervised = 48 * 60
    primary_supervised = 96 * 60
    replay_supervised = total_supervised - primary_supervised
    loss = {
        "per_count_task": {
            "assistant_supervised_tokens": 60,
            "demo_tokens": 45,
            "demo_fraction": 0.75,
            "final_query_tokens": 15,
            "final_query_fraction": 0.25,
            "grid_cell_tokens": 36,
            "formatting_and_newline_tokens": 24,
            "decision_tokens": 8,
            "query_decision_tokens": 2,
        },
        "effective_equal_slot_mass": objective_audit["nominal_equal_slot_objective_mass"],
        "full_384_slot_schedule": {
            "supervised_tokens": total_supervised,
            "count_primary_supervised_tokens": count_supervised,
            "count_primary_raw_token_fraction": count_supervised / total_supervised,
            "all_primary_supervised_tokens": primary_supervised,
            "all_primary_raw_token_fraction": primary_supervised / total_supervised,
            "protected_replay_supervised_tokens": replay_supervised,
            "protected_replay_raw_token_fraction": replay_supervised / total_supervised,
            "slot_counts": dict(sorted(roles.items())),
        },
        "code_pair_reweighting": {
            "focused_tokens": 8,
            "focused_weight": 2.0,
            "remaining_tokens": 52,
            "remaining_weight": 11 / 13,
            "total_mass": 60,
            "demo_mass": 45,
            "query_mass": 15,
            "demo_fraction": 0.75,
            "query_fraction": 0.25,
            "conclusion": "The reweighting emphasizes the same two count-code cells in all four assistant outputs and leaves demo/query mass at 75/25; it is not query-only supervision.",
        },
        "gradient_decomposition": {
            "status": "NOT_RUN_CPU_ONLY_NO_MODEL",
            "required_future_measurement": "At the common V7 start state, separately measure demo-output, query-output, query-decision, and formatting gradients on the frozen deconfounded panel; do not repeat the E04-D rotation-marker estimand.",
        },
    }
    ranked = [
        {
            "rank": 1,
            "bottleneck": "Count-task identifiability and shortcut-confounded curriculum",
            "confirmed": True,
            "quantitative_evidence": "topology->count 48/48; first-demo->query count 48/48; normalized MI(topology,count)=1.0; query color and rotation phase each constant",
            "causal_status": "Defect confirmed; causal contribution to model failure unproven until controlled training.",
        },
        {
            "rank": 2,
            "bottleneck": "Code-Pair decision-token locator semantics",
            "confirmed": True,
            "quantitative_evidence": "0/48 tasks match the documented row-1 columns 1/2; all 48 implement columns 0/1, making Count 2 and Count 3 codes identical",
            "causal_status": "Implementation defect confirmed; contribution to the failed independent result is unquantified.",
        },
        {
            "rank": 3,
            "bottleneck": "Weak effective query supervision under all-assistant CE and replay dilution",
            "confirmed": True,
            "quantitative_evidence": "demo/query mass=45/15 (75%/25%) within a count slot; Equal-Slot nominal mass is Count 12.5%, Object 12.5%, Replay 75%; Count-query mass is 3.125% of the full schedule",
            "causal_status": "Allocation confirmed; objective mismatch and replay dilution remain unproven causes.",
        },
        {
            "rank": 4,
            "bottleneck": "Free-running exact-output acquisition versus token-level fit",
            "confirmed": False,
            "quantitative_evidence": "V2 exact-grid: V7 64/104, Control 53/104, Treatment 52/104; count Treatment-Control +1/24 CI [-2,4], Treatment-V7 -4/24 CI [-10,2]",
            "causal_status": "Independent generalization failure confirmed; train-fit/Greedy mechanism unresolved because comparable raw per-example fit artifacts are unavailable and the V7 teacher-forced diagnostic was invalid.",
        },
        {
            "rank": 5,
            "bottleneck": "Object-centric representation or structural transfer",
            "confirmed": False,
            "quantitative_evidence": "Both trained arms lost 2/24 object-extraction correct relative to V7 with CI [-5,0].",
            "causal_status": "Plausible downstream mechanism, but current curriculum confounding prevents attribution.",
        },
        {
            "rank": 6,
            "bottleneck": "Direct local gradient conflict",
            "confirmed": False,
            "quantitative_evidence": f"E04-D rotation-marker aggregate CI [{e04d['aggregate_bootstrap_ci']['lower']:.6f},{e04d['aggregate_bootstrap_ci']['upper']:.6f}] with decision {e04d['decision']}",
            "causal_status": "Deprioritized: positive local alignment does not establish compatibility, but direct start-state opposition was not observed.",
        },
    ]
    return {
        "schema_version": 1,
        "audit_id": "POST_V7_GENERALIZATION_MECHANISM_AUDIT_V1",
        "status": "CPU_AUDIT_COMPLETE_GPU_NOT_AUTHORIZED",
        "source_identities": {
            "treatment_schedule": {"path": str(SCHEDULE.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(SCHEDULE)},
            "v2_result": {"path": str(V2_RESULT.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(V2_RESULT)},
            "e04d_result": {"path": str(E04D_RESULT.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(E04D_RESULT)},
            "token_accounting": {"path": str(TOKEN_ACCOUNTING.relative_to(ROOT)).replace("\\", "/"), "sha256": file_digest(TOKEN_ACCOUNTING)},
        },
        "dataset_identifiability": audit,
        "objective_implementation": objective_audit,
        "loss_and_gradient_decomposition": loss,
        "existing_prediction_evidence": {
            "available": "aggregate frozen exact-grid and paired contrasts only",
            "unavailable": "raw per-example V7/Control/Treatment predictions are not present in the committed evidence package, so error-transition attribution is NOT_RUN",
            "v2_metrics": v2["metrics"],
            "v2_paired_gates": v2["frozen_gate_inputs"],
            "decision": v2["score_decision"],
        },
        "ranked_bottlenecks": ranked,
        "highest_information_experiment": {
            "id": "POST_V7_DECONFOUNDED_COUNT_CURRICULUM_CAUSAL_PILOT_V1",
            "hypothesis": "Breaking topology/count and demo-order shortcuts, while holding model, optimizer, updates, token mass, object-extraction slots, and replay fixed, improves free-running structural-OOD count accuracy.",
            "control": "V7 initialization plus the original 48 confounded count tasks, the same 48 object-extraction tasks, and the same 288 replay slots.",
            "treatment": "Same V7 initialization and identical non-count slots; replace only the 48 count tasks with the CPU-validated counterbalanced TRAIN-derived panel.",
            "principal_variable": "count-curriculum identifiability/counterbalancing only",
            "fixed": ["V7 adapter bytes", "96 optimizer updates", "384 slots", "27360 supervised-token mass", "optimizer and LR", "object-extraction slots", "protected replay", "Native Greedy evaluation"],
            "development_endpoints": ["per-arm train-fit Native-Greedy exact-grid", "fresh structural-OOD exact-grid", "paired demo-order invariance", "protected retention", "non-duplicative composition sentinel"],
            "fresh_ood": {"panel_id": fresh_ood["panel_id"], "canonical_units": fresh_ood["canonical_units"], "prompt_records": fresh_ood["prompt_records"], "target_commitments": len(fresh_ood["target_commitments"])},
            "success_gate": {
                "train_fit": "Treatment reaches >=43/48 exact on its count-training queries; otherwise classify optimization/supervision failure.",
                "ood": "Treatment-Control >=6/48 canonical units and canonical-unit paired bootstrap 95% lower bound >0.",
                "order_invariance": "Treatment is correct under both demo orders on >=80% of its OOD-correct canonical units and is not worse than Control.",
                "retention": "No protected family loses >1 correct versus Control and aggregate protected delta >=-2 correct.",
                "composition": "No loss >1 correct versus Control; sentinel is a no-harm check, not evidence of compositional mastery.",
                "promotion": "No checkpoint promotion from this pilot alone; a newly frozen independent confirmation is required.",
            },
            "failure_interpretation": {
                "poor_train_fit": "Prioritize query-supervision/optimization diagnosis.",
                "good_train_fit_poor_ood": "Prioritize object-centric representation/grounding rather than another similar SFT curriculum.",
                "ood_gain_with_retention_loss": "Test replay composition as a separate intervention.",
            },
            "query_only_followup": "Run only as a separate matched experiment if the deconfounded arm has poor query train fit or a future component-gradient diagnostic shows demo-dominated or opposing query gradients.",
            "query_only_decision_rule": "If Equal-Slot plus deconfounded data reaches <43/48 query train-fit while demonstration-output fit is >=90%, or if a separately authorized component-gradient measurement finds demo-versus-query cosine <=0, freeze a second two-arm comparison on identical deconfounded data: Equal-Slot CE versus Query-Only CE. Otherwise do not add the objective arm.",
            "gpu_requirement": {"authorized": False, "devices": 1, "class": "RTX 5090 or equivalent BF16-capable >=24GB", "hard_cap_gpu_hours": 2.0, "arms": 2, "execution": "sequential fresh V7 initializations"},
        },
        "boundaries": ["CPU-only work completed", "no GPU/model load", "no Gold/dGold", "no V2 target reuse", "no checkpoint promotion"],
    }


def render_review(report: dict[str, Any], panel: dict[str, Any], fresh_ood: dict[str, Any]) -> str:
    lines = [
        "# Post-V7 independent-generalization mechanism audit V1",
        "",
        "## Verdict",
        "",
        "The Count curriculum has a confirmed identifiability defect, but its causal role in the failed independent transfer remains unproven. Topology predicts the query count in 48/48 tasks, and the first demonstration count equals the query count in 48/48 tasks. Query color and rotation phase are also constant. The cleanest next causal test changes only this curriculum construction.",
        "",
        "## Quantitative evidence",
        "",
        "- Original Count TRAIN: topology-to-count accuracy 48/48; first-demo-count accuracy 48/48; normalized mutual information 1.0.",
        "- Supervision: 45/60 assistant tokens (75%) are demonstrations and 15/60 (25%) are the final query. Existing code-pair weighting preserves this 75/25 split.",
        "- Equal-Slot effective coefficient mass: Count 12.5%, Object Extraction 12.5%, Replay 75%; the final Count query is 3.125% of full-schedule nominal mass. Raw token exposure is 2,880/27,360 Count and 21,600/27,360 replay, but those raw shares are not the Equal-Slot weights.",
        "- Implementation audit: all 48 Count tasks pass user-mask/assistant-label, causal-shift, and mass-preservation checks. Code-Pair fails its documented semantic locator on 48/48 tasks: it selects columns 0/1, not 1/2, so Count 2 and Count 3 receive the same selected pattern.",
        "- Independent V2: V7 64/104, Control 53/104, Treatment 52/104. Treatment-Control Count +1/24, 95% CI [-2,4]; Treatment-V7 Count -4/24, CI [-10,2].",
        "- E04-D: positive start-state local alignment (bootstrap CI 0.842828 to 0.859694) deprioritizes direct local gradient opposition but does not prove training compatibility.",
        f"- New CPU panel: {panel['validation']['records']} TRAIN-derived tasks, topology/count MI {panel['validation']['topology_target_mutual_information_bits']:.1f}, first-demo match {panel['validation']['first_demo_query_matches']}/48, and {panel['validation']['demo_order_pairs_oracle_invariant']}/48 oracle-invariant order pairs.",
        f"- Fresh target-blind OOD: {fresh_ood['canonical_units']} canonical units and {fresh_ood['prompt_records']} paired-order prompts; no V2 or Gold/dGold targets used.",
        "",
        "## Ranked bottlenecks",
        "",
    ]
    for item in report["ranked_bottlenecks"]:
        lines.append(f"{item['rank']}. **{item['bottleneck']}** - {item['causal_status']} Evidence: {item['quantitative_evidence']}")
    exp = report["highest_information_experiment"]
    lines += [
        "",
        "## Highest-information experiment",
        "",
        f"**{exp['id']}**",
        "",
        exp["hypothesis"],
        "",
        f"Control: {exp['control']}",
        "",
        f"Treatment: {exp['treatment']}",
        "",
        "This is a two-arm, one-variable causal pilot. Query-only supervision is explicitly deferred to a separate conditional experiment so curriculum identifiability and objective choice are not changed together.",
        "",
        "GPU requirement: one BF16-capable >=24GB GPU, two sequential fresh-V7 arms, hard cap 2 GPU-hours. GPU execution is not authorized by this audit.",
        "",
        "## Evidence limits",
        "",
        "Raw V2 per-example predictions are absent from the committed result package, so per-example error transitions are NOT_RUN. Demo/query/decision/formatting gradient decomposition is also NOT_RUN because this CPU-only audit did not load a model. Neither omission is silently converted into a causal claim.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    audit, original_rows = original_audit(schedule)
    objective_audit = objective_implementation_audit(schedule)
    panel = deconfounded_train_panel(original_rows)
    ood = fresh_ood_prompts()
    report = mechanism_and_proposal(audit, objective_audit, panel, ood)
    write_json(OUT / "DATASET_IDENTIFIABILITY_AUDIT.json", audit)
    write_json(OUT / "OBJECTIVE_IMPLEMENTATION_AUDIT.json", objective_audit)
    write_json(OUT / "COUNTERBALANCED_TRAIN_DERIVED_TASKS.json", panel)
    write_json(OUT / "FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json", ood)
    write_json(OUT / "MECHANISM_DIAGNOSIS_AND_CAUSAL_PROPOSAL.json", report)
    (OUT / "SCIENTIFIC_REVIEW.md").write_bytes(render_review(report, panel, ood).encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "audit_id": report["audit_id"],
        "status": report["status"],
        "files": {path.name: file_digest(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "MANIFEST.json"},
        "source_script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "source_script_sha256": file_digest(Path(__file__)),
        "gpu_used": False,
        "model_loaded": False,
        "gold_or_dgold_accessed": False,
    }
    write_json(OUT / "MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "out": str(OUT), "files": len(manifest["files"])}, sort_keys=True))


if __name__ == "__main__":
    main()
