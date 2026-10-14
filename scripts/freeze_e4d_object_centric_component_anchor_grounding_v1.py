"""Freeze new, target-blind E4-D panels without model loading."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARC2_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
BASE = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
E3 = BASE / "deconfounded_count_curriculum_causal_pilot_v1"
OUT = E3 / "e4d_object_centric_component_anchor_grounding_v1"


def canon(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
def digest(value): return hashlib.sha256(canon(value)).hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canon(value) + b"\n")


def normalized(points):
    y0, x0 = min(y for y, _ in points), min(x for _, x in points)
    return tuple(sorted((y - y0, x - x0) for y, x in points))


def rotate(points, turns):
    output = tuple(points)
    for _ in range(turns % 4):
        output = tuple((x, -y) for y, x in output)
    return normalized(output)


# Ten-cell shapes are disjoint from prior 4/5/6/8/9-cell OOD families.
TOPOLOGIES = {
    "I10": ((0,0),(1,0),(2,0),(2,1),(2,2),(3,2),(4,2),(4,3),(5,3),(5,4)),
    "J10": ((0,1),(1,1),(2,0),(2,1),(2,2),(3,0),(3,2),(4,0),(4,1),(4,2)),
    "K10": ((0,0),(0,1),(1,1),(2,0),(2,1),(2,2),(3,1),(4,0),(4,1),(4,2)),
    "L10": ((0,0),(1,0),(1,1),(2,1),(3,0),(3,1),(3,2),(4,2),(5,2),(5,3)),
}
COLORS = (1,2,3,4,6,7,8,9)
ANCHORS = ((0,0),(0,7),(7,0))


def count_task(points, count, turns, color):
    shape = rotate(points, turns)
    grid = [[0] * 14 for _ in range(14)]
    for top, left in ANCHORS[:count]:
        for y, x in shape:
            grid[top + y][left + x] = color
    return {"input": grid, "output": [[0,0,0],[*([color] * count), *([0] * (3-count))],[0,0,0]]}


def role_hash(task): return digest({"train": task["train"], "query_input": task["test"][0]["input"]})
def pair_hash(task): return digest(task), digest(task["test"][0]["input"])


def prior_tasks():
    paths = [E3 / "CONTROL_SCHEDULE.json", E3 / "TREATMENT_SCHEDULE.json"]
    paths += [
        BASE / "post_v7_generalization_mechanism_audit_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        E3 / "batch1_only_independent_measurement_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        E3 / "query_only_count_objective_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        E3 / "e4a_matched_lr_query_only_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        E3 / "e4b_corrected_code_pair_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        E3 / "e4c_qlcp_v1/OOD_INPUTS_AND_COMMITMENTS.json",
        E3 / "e4c_qlcp_v1/MARKER_INPUTS_AND_COMMITMENTS.json",
    ]
    output = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        output += [row["task"] for row in document.get("episodes", [])]
        output += [row["prompt"] for row in document.get("records", [])]
    return output


def overlap_sets():
    tasks = prior_tasks()
    return ({pair_hash(task)[0] for task in tasks}, {pair_hash(task)[1] for task in tasks}, {role_hash(task) for task in tasks})


def freeze_ood(prompt_hashes, query_hashes, role_hashes):
    from scripts.connected_components_object_extraction_semantic_oracle import target_for_family, validate_demonstrations
    orbits = {name: {rotate(points, turn) for turn in range(4)} for name, points in TOPOLOGIES.items()}
    if any(len(orbit) != 4 for orbit in orbits.values()) or any(left & right for left, right in itertools.combinations(orbits.values(), 2)):
        raise RuntimeError("E4D_TOPOLOGY_ORBIT_INVALID")
    rows, commitments, index = [], [], 0
    for topology, count, turn in itertools.product(TOPOLOGIES, (1,2,3), range(4)):
        color = COLORS[index % len(COLORS)]
        demo_counts = [1,2,3]
        shift = (index + 2) % 3
        demo_counts = demo_counts[shift:] + demo_counts[:shift]
        demos = [count_task(TOPOLOGIES[topology], n, turn + j + 1, COLORS[(index+j+3) % len(COLORS)]) for j, n in enumerate(demo_counts)]
        query = count_task(TOPOLOGIES[topology], count, turn, color)
        canonical_id = f"E4D_COMPONENT_ANCHOR_NEW_OOD:{topology}:C{count}:R{turn}"
        for order, shown in (("A", demos), ("B", list(reversed(demos)))):
            task = {"source_id": canonical_id + ":ORDER_" + order, "train": shown, "test": [{"input": query["input"]}]}
            validate_demonstrations(task, "CONNECTED_COMPONENT_COUNT")
            target = target_for_family("CONNECTED_COMPONENT_COUNT", query["input"])
            if target != query["output"]:
                raise RuntimeError("E4D_OOD_ORACLE_MISMATCH")
            prompt_hash, query_hash = pair_hash(task)
            content_role_hash = role_hash(task)
            if prompt_hash in prompt_hashes or query_hash in query_hashes or content_role_hash in role_hashes:
                raise RuntimeError("E4D_OOD_OVERLAP")
            rows.append({"record_id": task["source_id"], "canonical_unit_id": canonical_id, "demo_order": order, "topology": topology, "query_count": count, "turns": turn, "query_color": color, "prompt": task, "prompt_sha256": prompt_hash, "query_input_sha256": query_hash, "content_role_sha256": content_role_hash})
            commitments.append({"record_id": task["source_id"], "target_sha256": digest(target)})
        index += 1
    if len(rows) != 96 or len({row["canonical_unit_id"] for row in rows}) != 48:
        raise RuntimeError("E4D_OOD_GRID_INVALID")
    return {"panel_id": "E4D_COMPONENT_ANCHOR_NEW_STRUCTURAL_OOD_V1", "status": "FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS", "canonical_units": 48, "prompt_records": 96, "records": rows, "target_commitments": commitments, "oracle": "PASS_INDEPENDENT_CONNECTED_COMPONENT_COUNT", "topology_count_mutual_information_bits": 0.0, "target_blind": True}


def marker_base(base, index):
    item = copy.deepcopy(base)
    item["instance_index"] = 20000 + index
    item["seed"] = 9200000 + index * 7919
    # `task_prompt` deliberately rotates layouts through the native marker
    # constructor's fixed vocabulary.  Use a distinct subset of that
    # vocabulary here; instance/seed/role identity still make this panel
    # disjoint from every earlier marker panel.
    item["layout_template"] = ("CENTER", "NW", "NE", "SW", "SE", "MID_WEST", "MID_EAST", "MID_NORTH")[index]
    item["color_role_tuple"] = [((value + 2 * index) % 9) + 1 for value in item["color_role_tuple"]]
    if len(set(item["color_role_tuple"])) < 4:
        item["color_role_tuple"] = [1,4,6,9] if index % 2 == 0 else [2,3,5,8]
    item["canonical_base_id"] = digest({key: value for key, value in item.items() if key != "canonical_base_id"})
    return item


def marker_oracle(base, turn):
    background, output = base["color_role_tuple"][0], base["color_role_tuple"][3]
    return [[output if column == turn else background for column in range(4)]]


def freeze_marker(prompt_hashes, query_hashes, role_hashes):
    from scripts import e04_orientation_marker_counterfactual_v2 as marker
    rows, commitments = [], []
    for index, source in enumerate(marker.canonical_bases("VALIDATION")[:8]):
        base = marker_base(source, index)
        for turn in (0,2):
            task = marker.task_prompt(base, turn, "MARKER_BINDING_CONTROL")
            task["source_id"] = f"E4D_NEW_MARKER_BINDING:{base['canonical_base_id']}:T{turn}"
            target = marker_oracle(base, turn)
            if target != marker.target_grid(base, turn, "MARKER_BINDING_CONTROL"):
                raise RuntimeError("E4D_MARKER_ORACLE_MISMATCH")
            prompt_hash, query_hash = pair_hash(task)
            content_role_hash = role_hash(task)
            if prompt_hash in prompt_hashes or query_hash in query_hashes or content_role_hash in role_hashes:
                raise RuntimeError("E4D_MARKER_OVERLAP")
            rows.append({"record_id": task["source_id"], "canonical_unit_id": base["canonical_base_id"], "marker_turn": turn, "condition": "MARKER_BINDING_CONTROL", "prompt": task, "prompt_sha256": prompt_hash, "query_input_sha256": query_hash, "content_role_sha256": content_role_hash})
            commitments.append({"record_id": task["source_id"], "target_sha256": digest(target)})
    if len(rows) != 16 or len({row["canonical_unit_id"] for row in rows}) != 8:
        raise RuntimeError("E4D_MARKER_GRID_INVALID")
    return {"panel_id": "E4D_NEW_MARKER_BINDING_PRESERVATION_V1", "status": "FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS", "canonical_units": 8, "prompt_records": 16, "records": rows, "target_commitments": commitments, "oracle": "PASS_INDEPENDENT_MARKER_BINDING", "target_blind": True}


def freeze():
    prompt_hashes, query_hashes, role_hashes = overlap_sets()
    ood = freeze_ood(prompt_hashes, query_hashes, role_hashes)
    prompt_hashes |= {row["prompt_sha256"] for row in ood["records"]}
    query_hashes |= {row["query_input_sha256"] for row in ood["records"]}
    role_hashes |= {row["content_role_sha256"] for row in ood["records"]}
    marker = freeze_marker(prompt_hashes, query_hashes, role_hashes)
    write(OUT / "OOD_INPUTS_AND_COMMITMENTS.json", ood)
    write(OUT / "MARKER_INPUTS_AND_COMMITMENTS.json", marker)
    receipt = {"status": "PASS_E4D_PROSPECTIVE_FREEZE_NO_MODEL", "model_loaded": False, "gpu_used": False, "targets_opened": False, "target_sidecar_written": False, "ood_sha256": hashlib.sha256((OUT / "OOD_INPUTS_AND_COMMITMENTS.json").read_bytes()).hexdigest(), "marker_sha256": hashlib.sha256((OUT / "MARKER_INPUTS_AND_COMMITMENTS.json").read_bytes()).hexdigest(), "zero_overlap_prior_prompt_query_content_role": True, "ood": {"canonical_units": 48, "count_balance": {str(value):16 for value in (1,2,3)}, "orders":2, "topology_count_mutual_information_bits":0.0}, "marker":{"canonical_units":8,"records":16}}
    write(OUT / "PROSPECTIVE_FREEZE_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True))
