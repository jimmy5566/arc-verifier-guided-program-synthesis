"""Freeze E4F2's new target-blind structural OOD inputs, CPU-only."""
from __future__ import annotations

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
OUT = E3 / "e4f2_distance_overlap_affinity_v1"


def canon(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
def digest(value): return hashlib.sha256(canon(value)).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canon(value) + b"\n")


# Twelve-cell asymmetric families differ in cell cardinality from the exposed
# E4D (10-cell) and E4E (11-cell) structural panels.
TOPOLOGIES = {
    "B12": ((0,0),(0,1),(1,1),(1,2),(2,0),(2,1),(2,2),(3,2),(4,1),(4,2),(5,2),(5,3)),
    "C12": ((0,1),(1,0),(1,1),(2,1),(2,2),(3,2),(3,3),(4,3),(4,4),(5,2),(5,3),(6,3)),
    "D12": ((0,0),(1,0),(1,1),(2,1),(2,2),(2,3),(3,3),(4,2),(4,3),(5,3),(5,4),(6,4)),
    "E12": ((0,2),(1,1),(1,2),(2,0),(2,1),(2,2),(3,2),(3,3),(4,3),(5,3),(5,4),(6,4)),
}
COLOURS = (1, 2, 3, 4, 6, 7, 8, 9)
ANCHORS = ((1, 1), (1, 9), (13, 1))


def norm(points):
    y = min(a for a, _ in points); x = min(b for _, b in points)
    return tuple(sorted((a-y, b-x) for a, b in points))


def rotate(points, turns):
    out = tuple(points)
    for _ in range(turns % 4):
        out = tuple((x, -y) for y, x in out)
    return norm(out)


def count_components(grid):
    from scripts.e4f2_component_affinity import components
    return len(components(grid))


def make_example(points, count, turns, colour):
    grid = [[0] * 24 for _ in range(24)]
    for top, left in ANCHORS[:count]:
        for y, x in rotate(points, turns):
            grid[top+y][left+x] = colour
    return {"input": grid, "output": [[0,0,0], [*([colour]*count), *([0]*(3-count))], [0,0,0]]}


def prompt_digest(task): return digest(task), digest(task["test"][0]["input"]), digest({"train":task["train"], "query_input":task["test"][0]["input"]})


def prior_tasks():
    files = [E3 / "CONTROL_SCHEDULE.json", E3 / "TREATMENT_SCHEDULE.json", OUT / "SUCCESSOR_COUNTERFACTUAL_SCHEDULE.json"]
    for rel in (
        "batch1_only_independent_measurement_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        "query_only_count_objective_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        "e4a_matched_lr_query_only_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        "e4b_corrected_code_pair_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        "e4c_qlcp_v1/OOD_INPUTS_AND_COMMITMENTS.json",
        "e4d_object_centric_component_anchor_grounding_v1/OOD_INPUTS_AND_COMMITMENTS.json",
        "e4e_late_count_v1/OOD_INPUTS_AND_COMMITMENTS.json",
    ):
        files.append(E3 / rel)
    tasks = []
    for path in files:
        data = json.loads(path.read_text())
        tasks += [row["task"] for row in data.get("episodes", [])]
        tasks += [row["prompt"] for row in data.get("records", [])]
    return tasks


def freeze():
    from scripts.connected_components_object_extraction_semantic_oracle import target_for_family, validate_demonstrations

    old = prior_tasks()
    old_prompts = {prompt_digest(x)[0] for x in old}
    old_queries = {prompt_digest(x)[1] for x in old}
    old_roles = {prompt_digest(x)[2] for x in old}
    if any(len({rotate(points, n) for n in range(4)}) != 4 for points in TOPOLOGIES.values()):
        raise RuntimeError("E4F2_OOD_ROTATION_ORBIT")
    rows, commitments = [], []
    ordinal = 0
    for topology, count, turns in itertools.product(TOPOLOGIES, (1, 2, 3), range(4)):
        colour = COLOURS[ordinal % len(COLOURS)]
        query = make_example(TOPOLOGIES[topology], count, turns, colour)
        demos = [make_example(TOPOLOGIES[topology], c, turns+j+1, COLOURS[(ordinal+j+3) % len(COLOURS)]) for j, c in enumerate((2,3,1))]
        unit = f"E4F2_DISTANCE_OVERLAP_OOD:{topology}:C{count}:R{turns}"
        for order, shown in (("A", demos), ("B", list(reversed(demos)))):
            prompt = {"source_id":unit+":ORDER_"+order, "train":shown, "test":[{"input":query["input"]}]}
            validate_demonstrations(prompt, "CONNECTED_COMPONENT_COUNT")
            target = target_for_family("CONNECTED_COMPONENT_COUNT", query["input"])
            if target != query["output"] or count_components(query["input"]) != count:
                raise RuntimeError("E4F2_OOD_INDEPENDENT_ORACLE")
            p, q, r = prompt_digest(prompt)
            if p in old_prompts or q in old_queries or r in old_roles:
                raise RuntimeError("E4F2_OOD_OVERLAP")
            rows.append({"record_id":prompt["source_id"], "canonical_unit_id":unit, "demo_order":order,
                         "topology":topology, "query_count":count, "turns":turns, "query_colour":colour,
                         "prompt":prompt, "prompt_sha256":p, "query_input_sha256":q, "content_role_sha256":r})
            commitments.append({"record_id":prompt["source_id"], "target_sha256":digest(target)})
        ordinal += 1
    if len(rows) != 96 or len({r["canonical_unit_id"] for r in rows}) != 48:
        raise RuntimeError("E4F2_OOD_EXACT_GRID")
    panel = {"schema_version":1, "panel_id":"E4F2_DISTANCE_OVERLAP_COMPONENT_AFFINITY_NEW_STRUCTURAL_OOD_V1",
             "status":"FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS", "canonical_units":48, "prompt_records":96,
             "records":rows, "target_commitments":commitments, "oracle":"PASS_INDEPENDENT_CONNECTED_COMPONENT_COUNT",
             "target_blind":True, "topology_count_mutual_information_bits":0.0,
             "overlap_policy":"zero full-prompt/query-input/content-role overlap with training and E3-E4E exposed panels"}
    write(OUT / "NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json", panel)
    receipt = {"status":"PASS_E4F2_NEW_OOD_PROSPECTIVE_FREEZE_NO_MODEL", "model_loaded":False, "gpu_used":False,
               "targets_opened":False, "sidecar_opened":False, "target_sidecar_written":False,
               "panel_sha256":sha(OUT / "NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json"),
               "canonical_units":48, "prompt_records":96, "count_balance":{"1":16,"2":16,"3":16},
               "orders":2, "topology_count_mutual_information_bits":0.0,
               "zero_overlap_prior_prompt_query_content_role":True}
    write(OUT / "NEW_OOD_PROSPECTIVE_FREEZE_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True))
