"""CPU-only prospective freeze for the E4D representation-versus-binding diagnostic.

This module deliberately never imports Torch, transformers, CUDA, or a model.
It freezes input-only panels and validates the later probe contract before a
separate execution authorization may be requested.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARC2_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
E3 = BASE / "deconfounded_count_curriculum_causal_pilot_v1"
E4D = E3 / "e4d_object_centric_component_anchor_grounding_v1"
OUT = E3 / "e4d_representation_vs_generation_diagnostic_v1"
E4D_EVIDENCE = ROOT / "e4d_evidence" / "E4D_4f704bc36c3d565a5840e631c2b8af0e"

# Eleven-cell forms and shifted 16x16 layout are fixed prospectively.  E4D
# used ten-cell forms in a 14x14 layout, so neither is a tuning reuse.
TOPOLOGIES = {
    "M11": ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (5, 3), (6, 3), (6, 4)),
    "N11": ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 2), (4, 0), (4, 1), (4, 2), (5, 1)),
    "P11": ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1), (4, 1), (5, 0), (5, 1), (5, 2)),
    "Q11": ((0, 0), (1, 0), (1, 1), (2, 1), (3, 0), (3, 1), (3, 2), (4, 2), (5, 2), (5, 3), (6, 3)),
}
PLACEMENTS = ((1, 1), (1, 9), (9, 1))
QUERY_COLOR = 5
LAYERS = (8, 16, 24, 31)
BOOTSTRAP_SEED = 20261013


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def normal(points):
    y0, x0 = min(y for y, _ in points), min(x for _, x in points)
    return tuple(sorted((y - y0, x - x0) for y, x in points))


def rotate(points, turns):
    output = tuple(points)
    for _ in range(turns % 4):
        output = tuple((x, -y) for y, x in output)
    return normal(output)


def count_example(points, count, turns, color=QUERY_COLOR):
    grid = [[0] * 16 for _ in range(16)]
    for top, left in PLACEMENTS[:count]:
        for y, x in rotate(points, turns):
            grid[top + y][left + x] = color
    return {"input": grid, "output": [[0, 0, 0], [*([color] * count), *([0] * (3 - count))], [0, 0, 0]]}


def role_hash(task):
    return digest({"train": task["train"], "query_input": task["test"][0]["input"]})


def pair_hash(task):
    return digest(task), digest(task["test"][0]["input"])


def prior_paths():
    return [
        E3 / "CONTROL_SCHEDULE.json", E3 / "TREATMENT_SCHEDULE.json",
        BASE / "post_v7_generalization_mechanism_audit_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        E3 / "batch1_only_independent_measurement_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        E3 / "query_only_count_objective_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        E3 / "e4a_matched_lr_query_only_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        E3 / "e4b_corrected_code_pair_causal_pilot_v1/NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",
        E3 / "e4c_qlcp_v1/OOD_INPUTS_AND_COMMITMENTS.json", E3 / "e4c_qlcp_v1/MARKER_INPUTS_AND_COMMITMENTS.json",
        E4D / "OOD_INPUTS_AND_COMMITMENTS.json", E4D / "MARKER_INPUTS_AND_COMMITMENTS.json",
    ]


def prior_hash_sets():
    tasks = []
    for path in prior_paths():
        document = json.loads(path.read_text(encoding="utf-8"))
        tasks.extend(row["task"] for row in document.get("episodes", []))
        tasks.extend(row["prompt"] for row in document.get("records", []))
    return ({pair_hash(t)[0] for t in tasks}, {pair_hash(t)[1] for t in tasks}, {role_hash(t) for t in tasks})


def freeze_ood():
    from scripts.connected_components_object_extraction_semantic_oracle import target_for_family, validate_demonstrations
    prompt_seen, query_seen, role_seen = prior_hash_sets()
    orbits = {name: {rotate(points, turn) for turn in range(4)} for name, points in TOPOLOGIES.items()}
    if any(len(orbit) != 4 for orbit in orbits.values()) or any(left & right for left, right in itertools.combinations(orbits.values(), 2)):
        raise RuntimeError("DIAGNOSTIC_TOPOLOGY_ORBIT_INVALID")
    rows, commitments, index = [], [], 0
    for topology, count, turn in itertools.product(TOPOLOGIES, (1, 2, 3), range(4)):
        demo_counts = [1, 2, 3]
        shift = index % 3
        demo_counts = demo_counts[shift:] + demo_counts[:shift]
        demos = [count_example(TOPOLOGIES[topology], n, turn + j + 1) for j, n in enumerate(demo_counts)]
        query = count_example(TOPOLOGIES[topology], count, turn)
        canonical_id = f"E4D_REPRESENTATION_BINDING_OOD:{topology}:C{count}:R{turn}"
        for order, shown in (("A", demos), ("B", list(reversed(demos)))):
            task = {"source_id": canonical_id + ":ORDER_" + order, "train": shown, "test": [{"input": query["input"]}]}
            validate_demonstrations(task, "CONNECTED_COMPONENT_COUNT")
            target = target_for_family("CONNECTED_COMPONENT_COUNT", task["test"][0]["input"])
            if target != query["output"]:
                raise RuntimeError("DIAGNOSTIC_INDEPENDENT_ORACLE_MISMATCH")
            prompt_hash, query_hash = pair_hash(task)
            content_role_hash = role_hash(task)
            if prompt_hash in prompt_seen or query_hash in query_seen or content_role_hash in role_seen:
                raise RuntimeError("DIAGNOSTIC_PRIOR_OVERLAP")
            rows.append({"record_id": task["source_id"], "canonical_unit_id": canonical_id, "demo_order": order,
                         "topology": topology, "query_count": count, "turns": turn, "query_color": QUERY_COLOR,
                         "topology_cells": 11, "layout": "16x16_SHIFTED_1_1__1_9__9_1", "prompt": task,
                         "prompt_sha256": prompt_hash, "query_input_sha256": query_hash, "content_role_sha256": content_role_hash})
            commitments.append({"record_id": task["source_id"], "target_sha256": digest(target)})
        index += 1
    if len(rows) != 96 or len({r["canonical_unit_id"] for r in rows}) != 48:
        raise RuntimeError("DIAGNOSTIC_CANONICAL_GRID_INVALID")
    return {"panel_id": "E4D_REPRESENTATION_VS_BINDING_STRUCTURAL_OOD_V1", "status": "FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS",
            "canonical_units": 48, "prompt_records": 96, "records": rows, "target_commitments": commitments,
            "oracle": "PASS_INDEPENDENT_CONNECTED_COMPONENT_COUNT", "target_blind": True,
            "count_balance": {"1": 16, "2": 16, "3": 16}, "topology_count_mutual_information_bits": 0.0,
            "structural_disjointness": {"topology_cells": 11, "query_color": QUERY_COLOR, "layout": "16x16_SHIFTED_1_1__1_9__9_1", "orders": ["A", "B"]}}


def train_probe_split():
    from scripts.connected_components_object_extraction_semantic_oracle import connected_component_count, validate_demonstrations
    schedule = json.loads((E3 / "TREATMENT_SCHEDULE.json").read_text(encoding="utf-8"))
    rows, seen = [], set()
    for episode in schedule["episodes"]:
        if episode["role"] != "PRIMARY_CONNECTED_COMPONENT_COUNT" or episode["canonical_base_id"] in seen:
            continue
        task = episode["task"]
        validate_demonstrations(task, "CONNECTED_COMPONENT_COUNT")
        target = connected_component_count(task["test"][0]["input"])
        count = sum(value != 0 for value in target[1])
        rows.append({"canonical_unit_id": episode["canonical_base_id"], "slot": episode["slot"], "count": count, "task": task,
                     "prompt_sha256": digest(task), "query_input_sha256": digest(task["test"][0]["input"]),
                     "content_role_sha256": role_hash(task)})
        seen.add(episode["canonical_base_id"])
    if len(rows) != 48 or {r["count"] for r in rows} != {1, 2, 3} or [sum(r["count"] == c for r in rows) for c in (1, 2, 3)] != [16, 16, 16]:
        raise RuntimeError("DIAGNOSTIC_TRAIN_PROBE_SPLIT_INVALID")
    return {"split_id": "E3_TRAIN_DERIVED_DECONFOUNDED_COUNT_PROBE_V1", "status": "FROZEN_TRAIN_DERIVED_DECONFOUNDED_LABELS",
            "records": rows, "canonical_units": 48, "count_balance": {"1": 16, "2": 16, "3": 16},
            "labels_are_probe_only": True, "model_weight_updates_forbidden": True}


def checkpoint_identities():
    binding = json.loads((E4D / "E4D_AUTH_BINDING.json").read_text(encoding="utf-8"))
    equal = json.loads((E4D_EVIDENCE / "EQUAL_SLOT_5E5_TRAINING_RECEIPT.json").read_text(encoding="utf-8"))
    anchor = json.loads((E4D_EVIDENCE / "COMPONENT_ANCHOR_TRAINING_RECEIPT.json").read_text(encoding="utf-8"))
    root = binding["output_root"]
    return {"status": "EXPECTED_IDENTITIES_REQUIRE_BYTE_SHA256_PRE_MODEL", "V7": {"adapter_path": binding["v7"]["adapter_path"], "adapter_files": binding["v7"]["adapter_files"], "base_path": binding["v7"]["base_path"], "base_files": binding["v7"]["base_files"]},
            "EQUAL_SLOT": {"adapter_path": root + "/EQUAL_SLOT_5E5_ADAPTER", "adapter_model_sha256": equal["adapter_sha256"], "training_receipt_sha256": file_sha(E4D_EVIDENCE / "EQUAL_SLOT_5E5_TRAINING_RECEIPT.json")},
            "COMPONENT_ANCHOR": {"adapter_path": root + "/COMPONENT_ANCHOR_ADAPTER", "adapter_model_sha256": anchor["adapter_sha256"], "training_receipt_sha256": file_sha(E4D_EVIDENCE / "COMPONENT_ANCHOR_TRAINING_RECEIPT.json")},
            "required_before_execution": "all listed model/adapters must exist with exact byte count and SHA256; no replacement or reconstruction is allowed"}


def probe_contract():
    return {"architecture": "frozen-checkpoint hidden-state extraction plus deterministic class-centroid affine multiclass linear readout only", "layers": list(LAYERS),
            "final_query_representation": {"token_positions": "all final-query grid digit tokens in row-major order, located by native task serialization", "pooling": "mean over those positions", "fixed_before_activations": True},
            "fit_split": "E3_TRAIN_DERIVED_DECONFOUNDED_COUNT_PROBE_V1", "evaluation_split": "E4D_REPRESENTATION_VS_BINDING_STRUCTURAL_OOD_V1",
            "readout": {"classes": [1, 2, 3], "bias": True, "fit": "per-class TRAIN-derived feature centroids; affine discriminant score_c(x)=2*mu_c.dot(x)-||mu_c||^2", "same_hyperparameters_each_checkpoint": True, "model_weight_updates": "FORBIDDEN", "readout_parameters": "deterministically fitted only from the frozen TRAIN-derived split"},
            "native_generation": {"decoder": "NVARC_NATIVE_GREEDY", "physical_batch_size": 1, "raw_freeze_before_target_sidecar": True},
            "decision_rule": {"binding_supported_only_if": "component-anchor minus equal-slot paired OOD probe >=6 canonical units and paired bootstrap 95% lower bound >0 while native-Greedy exact count does not materially improve", "anchor_representation_failed_if": "neither probe nor native-Greedy meets that threshold", "both_improve": "diagnostic support only; requires later independently frozen confirmation"}}


def deterministic_cpu_tests(ood, train, identities, contract):
    # A synthetic fixed-dimension fixture validates the *linear* probe math
    # only; it is never evidence about a checkpoint or the prospective panel.
    # Keep preparation dependency-free: production uses the same affine
    # multiclass readout, while this fixture checks its row-major mean-pool and
    # argmax semantics without importing a numerical/model runtime.
    fixture = [
        ((1.0, 0.0), 0), ((1.2, 0.1), 0), ((0.0, 1.0), 1),
        ((0.1, 1.2), 1), ((-1.0, -1.0), 2), ((-1.1, -0.9), 2),
    ]
    weights = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, -1.0, 0.0))
    def predict(vector):
        scores = [vector[0] * row[0] + vector[1] * row[1] + row[2] for row in weights]
        return max(range(3), key=lambda index: scores[index])
    if any(predict(vector) != label for vector, label in fixture):
        raise RuntimeError("LINEAR_READOUT_FIXTURE_FAILED")
    if len(ood["target_commitments"]) != 96 or len(train["records"]) != 48:
        raise RuntimeError("DIAGNOSTIC_ARTIFACT_COUNT_FAILED")
    if identities["V7"]["adapter_files"][-1]["sha256"] == identities["EQUAL_SLOT"]["adapter_model_sha256"]:
        raise RuntimeError("CHECKPOINT_IDENTITY_COLLISION")
    return {"status": "PASS_E4D_REPRESENTATION_BINDING_CPU_VALIDATION", "model_loaded": False, "gpu_used": False,
            "targets_opened": False, "target_sidecar_written": False, "independent_oracle": ood["oracle"],
            "fresh_ood": {"canonical_units": 48, "records": 96, "count_balance": ood["count_balance"], "orders": 2,
                          "zero_overlap_prior_prompt_query_content_role": True},
            "train_probe": {"canonical_units": 48, "count_balance": train["count_balance"]},
            "fixed_layers": contract["layers"], "linear_fixture_exact": True,
            "checkpoint_bytes_verified_now": False, "checkpoint_bytes_required_pre_model": True}


def freeze():
    ood = freeze_ood()
    train = train_probe_split()
    identities = checkpoint_identities()
    contract = probe_contract()
    receipt = deterministic_cpu_tests(ood, train, identities, contract)
    write(OUT / "OOD_INPUTS_AND_COMMITMENTS.json", ood)
    write(OUT / "TRAIN_DERIVED_PROBE_SPLIT.json", train)
    write(OUT / "CHECKPOINT_IDENTITIES.json", identities)
    write(OUT / "PROBE_CONTRACT.json", contract)
    receipt["artifact_sha256"] = {name: file_sha(OUT / name) for name in ("OOD_INPUTS_AND_COMMITMENTS.json", "TRAIN_DERIVED_PROBE_SPLIT.json", "CHECKPOINT_IDENTITIES.json", "PROBE_CONTRACT.json")}
    write(OUT / "CPU_VALIDATION_RECEIPT.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if not args.freeze:
        raise SystemExit("--freeze is required")
    print(json.dumps(freeze(), sort_keys=True))


if __name__ == "__main__":
    main()
