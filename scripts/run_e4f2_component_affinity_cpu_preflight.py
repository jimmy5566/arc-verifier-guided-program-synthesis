"""Production-shaped CPU closure for the E4F2 affinity package.

This runner deliberately has no model, CUDA, or target-sidecar path.  It
validates the frozen serializer, pair loss contract, raw-freeze ordering, and
the deterministic promotion gate with synthetic non-scientific fixture rows.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
import sys

ROOT = Path(os.environ.get("ARC2_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
E3 = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1"
OUT = E3 / "e4f2_distance_overlap_affinity_v1"


def canon(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(canon(value)+b"\n")


def protocol():
    return {
        "schema_version":"E4F2_DISTANCE_OVERLAP_COMPONENT_AFFINITY_CAUSAL_PILOT_V1",
        "status":"FROZEN_CPU_VALIDATED_NO_EXECUTION_AUTHORIZED",
        "question":"Does balanced, target-blind multicomponent affinity supervision improve native Count OOD output generation over matched Equal-Slot CE?",
        "arms":{"control":"fresh exact V7 with unchanged full Equal-Slot CE", "treatment":"same counterfactual schedule and full Equal-Slot CE plus 0.25 balanced affinity BCE on Count=2/3 only"},
        "fixed":{"v7_initialization":"exact", "seed":20261013, "learning_rate":5e-5, "optimizer":"PagedAdamW8bit", "updates":96, "slots_per_update":4, "slot_coefficient":0.25, "physical_batch_size":1, "decoder":"NVARC_NATIVE_GREEDY", "protected_replay":"byte-identical Equal-Slot CE", "answer_ce_coefficient":1.0},
        "auxiliary":{"coefficient":0.25,"normalization":"0.25 times one binary BCE per treated slot; no auxiliary term for Count=1","pair_domain":"visible foreground final-query cells","labels":"same colour-separated 4-connected component", "treated_count_strata":[2,3],"count1_coefficient":0.0,"pair_balance":"one deterministic unordered pair per Count=2/3 treated slot; 32 slots total, 16 SAME_COMPONENT and 16 DIFFERENT_COMPONENT overall, with exact 8/8 label balance within each Count=2 and Count=3 stratum","matching":"positive and negative examples are matched across prospectively constructed P/N counterfactual task pairs, not within one query","metadata_boundary":"variant and generator metadata are audit-only and never serialized into model input or auxiliary-head features","permutation_invariant":True,"head_saved_for_inference":False},
        "panels":{"ood_path":str((OUT/'NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json').relative_to(ROOT)).replace('\\\\','/'),"canonical_units":48,"orders":2,"count_balance":{"1":16,"2":16,"3":16},"topology_count_mutual_information_bits":0.0,"target_blind":True},
        "gates":{"native_ood_delta_correct_min":6,"native_ood_bootstrap_95_lower_strict_gt":0,"train_fit":"diagnostic predeclared","auxiliary_fit":"diagnostic predeclared","order_invariance":"required","marker_binding_treatment_minus_control_min":0,"retention_each_family_treatment_minus_control_min":0,"retention_aggregate_treatment_minus_control_min":0,"composition_treatment_minus_control_min":0,"promotion":"all gates plus later independent confirmation"},
        "boundaries":{"raw_freeze_before_sidecar":True,"target_blind_inference":True,"forbidden":["Gold","dGold","FINAL_AUDIT","GPU","model_load","training","generation","target_sidecar_open","checkpoint_promotion","official_submission"]},
    }


def validate_freeze():
    from training_data.pipeline import task_to_sample
    from scripts.e4f2_component_affinity import pair_layout, scalar_mass
    schedule = load(OUT / "SUCCESSOR_COUNTERFACTUAL_SCHEDULE.json")["episodes"]
    spec = load(OUT / "PAIR_AFFINITY_SPEC.json")
    panel = load(OUT / "NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json")
    if len(schedule) != 384 or [x["slot"] for x in schedule] != list(range(384)) or [x["optimizer_step"] for x in schedule] != [i//4 for i in range(384)]:
        raise RuntimeError("E4F2_PREFLIGHT_SCHEDULE")
    roles = {name:sum(x["role"] == name for x in schedule) for name in ("PRIMARY_CONNECTED_COMPONENT_COUNT", "PRIMARY_OBJECT_EXTRACTION", "PROTECTED_REPLAY")}
    if roles != {"PRIMARY_CONNECTED_COMPONENT_COUNT":48,"PRIMARY_OBJECT_EXTRACTION":48,"PROTECTED_REPLAY":288}:
        raise RuntimeError("E4F2_PREFLIGHT_ROLES")
    if sum(task_to_sample(x["task"])["assistant_token_count"] for x in schedule) != 27360:
        raise RuntimeError("E4F2_PREFLIGHT_SUPERVISED_TOKEN_BUDGET")
    by_slot = {x["slot"]:x["task"] for x in schedule}; treated = 0
    for record in spec["records"]:
        if record["count"] == 1:
            if record["selected_pairs"] or scalar_mass(1.0, 1.0, False, 1) != scalar_mass(1.0, 1.0, True, 1):
                raise RuntimeError("E4F2_PREFLIGHT_COUNT1")
        else:
            layout = pair_layout(by_slot[record["slot"]], record["selected_pairs"])
            if layout.component_count != record["count"] or layout.labels[0] not in (0,1):
                raise RuntimeError("E4F2_PREFLIGHT_PAIR_LAYOUT")
            treated += 1
    if treated != 32 or not panel["target_blind"] or len(panel["records"]) != 96 or len(panel["target_commitments"]) != 96:
        raise RuntimeError("E4F2_PREFLIGHT_PANEL")
    if any("output" in row["prompt"]["test"][0] for row in panel["records"]):
        raise RuntimeError("E4F2_PREFLIGHT_TARGET_EXPOSED")
    return {"status":"PASS_E4F2_PRODUCTION_SHAPED_PREFLIGHT_NO_MODEL", "slots":384,"treated_slots":treated,"assistant_supervised_tokens":27360,"ood_records":96,"model_loaded":False,"gpu_used":False,"targets_opened":False,"sidecar_opened":False}


def frozen_raw(paths: dict[str, Path], directory: Path):
    manifest = {arm:{"path":path.name,"sha256":sha(path),"rows":len(path.read_text().splitlines())} for arm,path in paths.items()}
    write(directory / "RAW_FREEZE_FIXTURE.json", {"status":"FROZEN_BEFORE_SIDECAR_FIXTURE","arms":manifest})
    return manifest


def promotion(score: dict):
    gates = {
        "native_ood":score["native_delta"] >= 6 and score["bootstrap_lower"] > 0,
        "marker":all(v >= 0 for v in score["marker"].values()),
        "retention":all(v >= 0 for v in score["retention"].values()) and score["retention_aggregate"] >= 0,
        "composition":score["composition"] >= 0,
        "order_invariance":bool(score["order_invariance"]),
    }
    return {"gates":gates,"decision":"PASS_REQUIRES_LATER_INDEPENDENT_CONFIRMATION" if all(gates.values()) else "REJECT","promotion":"FORBIDDEN"}


def cpu_integration():
    pre = validate_freeze(); directory = Path(tempfile.mkdtemp(prefix="e4f2-cpu-"))
    try:
        panel = load(OUT / "NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json")
        paths = {}
        for arm in ("V7","EQUAL_SLOT","COMPONENT_AFFINITY"):
            path = directory / f"{arm}_RAW.jsonl"
            path.write_bytes(b"".join(canon({"record_id":row["record_id"],"generated_token_ids":[],"parsed_grid":None,"effective_batch_size":1})+b"\n" for row in panel["records"]))
            paths[arm] = path
        manifest = frozen_raw(paths, directory)
        # Deliberately synthetic fixture, never a panel target: proves score routing
        # and blocks promotion on a protected regression.
        fixture = promotion({"native_delta":6,"bootstrap_lower":0.1,"marker":{"endpoint_0":0,"endpoint_2":0},"retention":{"protected":0},"retention_aggregate":0,"composition":0,"order_invariance":True})
        negative = promotion({"native_delta":6,"bootstrap_lower":0.1,"marker":{"endpoint_0":0},"retention":{"protected":-1},"retention_aggregate":-1,"composition":0,"order_invariance":True})
        if fixture["decision"].startswith("REJECT") or negative["decision"] != "REJECT":
            raise RuntimeError("E4F2_PREFLIGHT_PROMOTION_CONTRACT")
        return {"status":"PASS_E4F2_LAUNCHER_WORKER_RAW_FREEZE_SCORER_CPU_INTEGRATION_NO_MODEL","preflight":pre["status"],"raw_freeze_before_sidecar":True,"raw_fixture":manifest,"promotion_contract":fixture,"protected_regression_fixture":negative,"model_loaded":False,"gpu_used":False,"targets_opened":False,"sidecar_opened":False}
    finally:
        shutil.rmtree(directory)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--integration", action="store_true"); args = parser.parse_args()
    result = cpu_integration() if args.integration else validate_freeze()
    write(OUT / ("CPU_PRODUCTION_PATH_RECEIPT.json" if args.integration else "CPU_PREFLIGHT_RECEIPT.json"), result)
    print(json.dumps(result, sort_keys=True))
