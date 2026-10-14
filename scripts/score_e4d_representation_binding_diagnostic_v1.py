"""CPU scorer for frozen E4D representation-versus-generation diagnostic raws.

The executor writes target-blind feature/native-count records.  This scorer is
the only component allowed to join them to the deterministic post-freeze
sidecar.  Its class-centroid discriminant is an affine, low-capacity linear
three-class readout: score_c(x)=2*mean_c dot x - ||mean_c||^2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def canon(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def rehash_raw_freeze(raw_paths):
    parents = {Path(path).parent.resolve() for path in raw_paths.values()}
    if len(parents) != 1: raise ValueError("RAW_FREEZE_DIRECTORY")
    receipt = read(next(iter(parents)) / "RAW_FREEZE_RECEIPT.json")
    if receipt.get("status") != "COMPLETE_RAW_FROZEN_NO_TARGETS" or receipt.get("targets_accessed") is not False:
        raise ValueError("RAW_FREEZE_STATUS")
    observed = {arm: sha(path) for arm, path in raw_paths.items()}
    if receipt.get("raw_sha256") != observed: raise ValueError("RAW_POST_FREEZE_MUTATION")
    return receipt


def mean(vectors):
    if not vectors: raise ValueError("EMPTY_CLASS")
    width = len(vectors[0])
    if not width or any(len(vector) != width for vector in vectors): raise ValueError("FEATURE_WIDTH")
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(width)]


def classifier(train):
    grouped = defaultdict(list)
    for vector, label in train: grouped[int(label)].append([float(x) for x in vector])
    if set(grouped) != {1, 2, 3}: raise ValueError("TRAIN_COUNT_BALANCE")
    centers = {label: mean(grouped[label]) for label in grouped}
    def predict(vector):
        vector = [float(x) for x in vector]
        return max(centers, key=lambda label: 2 * sum(a*b for a, b in zip(centers[label], vector, strict=True)) - sum(a*a for a in centers[label]))
    return predict, centers


def target_count(grid):
    return sum(value != 0 for value in grid[1])


def bootstrap(values, seed=20261013, replicates=10000):
    rng = random.Random(seed); n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) for _ in range(replicates))
    return samples[int(.025 * replicates)], samples[max(0, int(.975 * replicates) - 1)]


def classify(probe_pass, native_pass):
    table = {
        (True, False): "REPRESENTATION_TO_GENERATION_BINDING_SUPPORTED",
        (False, False): "ANCHOR_REPRESENTATION_INTERVENTION_FAILED",
        (True, True): "BOTH_IMPROVE_DIAGNOSTIC_SUPPORT_ONLY",
        (False, True): "NATIVE_ONLY_DISCORDANT_DIAGNOSTIC_INCONCLUSIVE",
    }
    return table[(bool(probe_pass), bool(native_pass))]


def arm_predictions(train_rows, ood_rows, layer):
    fit = classifier([(row["layer_features"][str(layer)], row["count"]) for row in train_rows])
    predict, centers = fit
    return ({row["record_id"]: int(predict(row["layer_features"][str(layer)])) for row in ood_rows}, centers)


def run(train_path, panel_path, sidecar_path, raw_paths, output_path):
    rehash_raw_freeze(raw_paths)
    train = read(train_path); panel = read(panel_path); sidecar = read(sidecar_path)
    if sidecar.get("opened_after_raw_freeze") is not True or sidecar.get("status") != "DERIVED_AFTER_RAW_FREEZE":
        raise ValueError("SIDECAR_FREEZE_BOUNDARY")
    if len(train["records"]) != 48 or len(panel["records"]) != 96:
        raise ValueError("FROZEN_GRID")
    meta = {row["record_id"]: row for row in panel["records"]}
    targets = sidecar["targets"]
    if set(targets) != set(meta): raise ValueError("SIDECAR_IDENTITY")
    for row in panel["target_commitments"]:
        if hashlib.sha256(canon(targets[row["record_id"]])).hexdigest() != row["target_sha256"]: raise ValueError("TARGET_COMMITMENT")
    actual = {record_id: target_count(grid) for record_id, grid in targets.items()}
    arms = {}
    for arm, path in raw_paths.items():
        raw = rows(path); train_raw = [r for r in raw if r["split"] == "TRAIN_PROBE"]
        ood_raw = [r for r in raw if r["split"] == "OOD"]
        if len(train_raw) != 48 or {r["record_id"] for r in ood_raw} != set(meta): raise ValueError("RAW_GRID:" + arm)
        if any(set(r["layer_features"]) != {"8", "16", "24", "31"} for r in raw): raise ValueError("LAYER_CONTRACT:" + arm)
        layers = {}
        for layer in (8, 16, 24, 31):
            predicted, centers = arm_predictions(train_raw, ood_raw, layer)
            layers[str(layer)] = {"predicted_count": predicted, "centroid_width": len(next(iter(centers.values())))}
        native = {r["record_id"]: r.get("native_count") for r in ood_raw}
        arms[arm] = {"layers": layers, "native_count": native, "raw_sha256": sha(path)}
    def canonical_correct(arm, source):
        by_unit = defaultdict(list)
        for record_id, row in meta.items():
            prediction = arms[arm]["layers"]["31"]["predicted_count"][record_id] if source == "probe" else arms[arm]["native_count"][record_id]
            by_unit[row["canonical_unit_id"]].append(prediction == actual[record_id])
        if len(by_unit) != 48 or any(len(values) != 2 for values in by_unit.values()): raise ValueError("CANONICAL_ORDER_GRID")
        return {unit: int(all(values)) for unit, values in by_unit.items()}
    equal_probe, anchor_probe = canonical_correct("EQUAL_SLOT", "probe"), canonical_correct("COMPONENT_ANCHOR", "probe")
    equal_native, anchor_native = canonical_correct("EQUAL_SLOT", "native"), canonical_correct("COMPONENT_ANCHOR", "native")
    units = sorted(equal_probe)
    probe_diff = [anchor_probe[u] - equal_probe[u] for u in units]
    native_diff = [anchor_native[u] - equal_native[u] for u in units]
    probe_ci, native_ci = bootstrap(probe_diff), bootstrap(native_diff)
    material_probe = sum(probe_diff) >= 6 and probe_ci[0] > 0
    material_native = sum(native_diff) >= 6 and native_ci[0] > 0
    result = {"schema_version": "E4D_REPRESENTATION_VS_BINDING_DIAGNOSTIC_SCORE_V1", "status": "COMPLETE_AFTER_RAW_FREEZE",
              "raw_freeze_before_sidecar": True, "primary_layer": 31, "layers_descriptive": [8, 16, 24, 31],
              "primary_probe": {"component_anchor_minus_equal_slot": sum(probe_diff), "bootstrap_95_ci": probe_ci, "canonical_units": 48},
              "native_greedy": {"component_anchor_minus_equal_slot": sum(native_diff), "bootstrap_95_ci": native_ci, "canonical_units": 48},
              "classification": classify(material_probe, material_native),
              "arms": arms, "inputs": {"train": sha(train_path), "panel": sha(panel_path), "sidecar": sha(sidecar_path)}}
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_bytes(canon(result) + b"\n")
    return result


def self_test():
    # Fixed tiny fixture: three balanced train classes and two canonical units.
    train = [{"record_id": f"t{i}", "count": i // 2 + 1, "layer_features": {str(layer): [float(i // 2 == 0), float(i // 2 == 1)] for layer in (8,16,24,31)}} for i in range(6)]
    pred, _ = arm_predictions(train, [{"record_id":"x", "layer_features":{str(layer):[1.,0.] for layer in (8,16,24,31)}}], 31)
    if pred["x"] != 1: raise RuntimeError("LINEAR_FIXTURE")
    return {"status":"PASS_E4D_REPRESENTATION_BINDING_SCORER_CPU_TEST","model_loaded":False,"gpu_used":False,"sidecar_opened":False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--train", type=Path); parser.add_argument("--panel", type=Path); parser.add_argument("--sidecar", type=Path); parser.add_argument("--raw-dir", type=Path); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test: result = self_test()
    else:
        if not all((args.train,args.panel,args.sidecar,args.raw_dir,args.output)): raise SystemExit("all scorer paths required")
        result = run(args.train,args.panel,args.sidecar,{arm:args.raw_dir/(arm+"_RAW.jsonl") for arm in ("V7","EQUAL_SLOT","COMPONENT_ANCHOR")},args.output)
    print(json.dumps(result,sort_keys=True))

if __name__ == "__main__": main()
