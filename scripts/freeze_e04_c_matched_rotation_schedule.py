"""Freeze a matched, target-blind E04-C pilot schedule without model imports.

The two arms contain 384 aligned slots.  They differ only at 96 predeclared
intervention slots: fixed-turn rotation supervision replaces the matched
no-transform target.  The remaining 288 slots are identical protected replay.
Tokenizer-derived token equality is intentionally a separate prelaunch gate.
"""
from __future__ import annotations
import argparse, hashlib, json, tempfile
from collections import Counter
from pathlib import Path
from scripts import freeze_e04_c_matched_rotation_pilot as cohort

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "experiments/capability_repair_baseline_v1/e04_c_matched_fixed_turn_rotation_repair_pilot_v1"
DEFAULT_OUT = DEFAULT_IN / "schedule_freeze_v1"
SEED = "E04_C_MATCHED_FIXED_TURN_ROTATION_REPAIR_PILOT_V1:selection:v1"
SLOTS = 384
INTERVENTION_SLOTS = 96
GRADIENT_ACCUMULATION = 4

class FreezeFailure(RuntimeError): pass

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def canon(x: object) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+".", delete=False) as f:
        f.write(canon(value)+b"\n"); temp=Path(f.name)
    temp.replace(path)

def rank(row: dict) -> str:
    return hashlib.sha256((SEED + ":" + row["canonical_base_id"]).encode()).hexdigest()

def load_rows(path: Path, key: str) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def select_interventions(rows: list[dict]) -> set[str]:
    by_cell: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for row in rows:
        c=row["train_cell"]; cell=(c["shape_orientation"], c["reference_relation"])
        by_cell.setdefault(cell, {}).setdefault(row["canonical_base_id"], []).append(row)
    if len(by_cell) != 12: raise FreezeFailure("TRAIN_CELL_COUNT")
    chosen: set[str] = set()
    for cell, bases in sorted(by_cell.items()):
        if len(bases) != 24 or any(len(v) != 4 for v in bases.values()):
            raise FreezeFailure("TRAIN_BASE_OR_TURN_COVERAGE")
        # Two deterministic bases per train cell, all four turns: 8 slots/cell.
        for base_id in sorted(bases, key=lambda x: rank({"canonical_base_id": x}))[:2]:
            chosen.update(row["pair_id"] for row in bases[base_id])
    if len(chosen) != INTERVENTION_SLOTS: raise FreezeFailure("INTERVENTION_COUNT")
    return chosen

def build(input_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists(): raise FreezeFailure("REFUSE_OVERWRITE")
    manifest=json.loads((input_dir/"MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["training_split_only"] is not True or manifest["validation_rows_read"] != 0:
        raise FreezeFailure("TRAIN_ONLY_PROVENANCE")
    trows=load_rows(input_dir/"TREATMENT_FIXED_TURN_ROTATION_TRAIN.jsonl", "treatment")
    crows=load_rows(input_dir/"CONTROL_NO_TRANSFORM_TRAIN.jsonl", "control")
    by_t={r["pair_id"]:r for r in trows}; by_c={r["pair_id"]:r for r in crows}
    if set(by_t)!=set(by_c) or len(by_t)!=1152: raise FreezeFailure("PAIR_ALIGNMENT")
    for pair_id in by_t:
        if by_t[pair_id]["canonical_base_id"] != by_c[pair_id]["canonical_base_id"] or by_t[pair_id]["turn"] != by_c[pair_id]["turn"]:
            raise FreezeFailure("PAIR_METADATA_MISMATCH")
    selected=select_interventions(trows)
    replay_ids=[p for p in sorted(by_c, key=lambda p: hashlib.sha256((SEED+":replay:"+p).encode()).hexdigest()) if p not in selected][:SLOTS-INTERVENTION_SLOTS]
    if len(replay_ids) != SLOTS-INTERVENTION_SLOTS or set(replay_ids)&selected: raise FreezeFailure("REPLAY_SELECTION")
    ordered=list(sorted(selected, key=lambda p: hashlib.sha256((SEED+":intervention:"+p).encode()).hexdigest()))+replay_ids
    control=[]; treatment=[]
    for slot,pair_id in enumerate(ordered):
        intervention=slot<INTERVENTION_SLOTS
        base={"slot":slot,"pair_id":pair_id,"intervention_slot":intervention,
              "role":"E04_C_INTERVENTION" if intervention else "PROTECTED_REPLAY",
              "canonical_base_id":by_c[pair_id]["canonical_base_id"],"turn":by_c[pair_id]["turn"],
              "train_cell":by_c[pair_id]["train_cell"],"control_task_sha256":by_c[pair_id]["control_task_sha256"],
              "control_task":by_c[pair_id]["control"]}
        c=dict(base)
        t=dict(base)
        if intervention:
            t["treatment_task_sha256"]=by_t[pair_id]["treatment_task_sha256"]
            t["treatment_task"]=by_t[pair_id]["treatment"]
        else:
            t["treatment_task_sha256"]=by_c[pair_id]["control_task_sha256"]
            t["treatment_task"]=by_c[pair_id]["control"]
        control.append(c); treatment.append(t)
    if len(control) != SLOTS or len(control)%GRADIENT_ACCUMULATION: raise FreezeFailure("STEP_ALIGNMENT")
    diff=[i for i,(c,t) in enumerate(zip(control,treatment)) if c["control_task_sha256"] != t["treatment_task_sha256"]]
    if diff != list(range(INTERVENTION_SLOTS)): raise FreezeFailure("ARM_DIFFERENCE")
    if len({x["pair_id"] for x in control}) != SLOTS: raise FreezeFailure("DUPLICATE_SCHEDULE_PAIR")
    counts=Counter(x["role"] for x in control)
    turn_counts=Counter(x["turn"] for x in control[:INTERVENTION_SLOTS])
    cell_counts=Counter((x["train_cell"]["shape_orientation"],x["train_cell"]["reference_relation"]) for x in control[:INTERVENTION_SLOTS])
    if counts != Counter({"PROTECTED_REPLAY":288,"E04_C_INTERVENTION":96}): raise FreezeFailure("ROLE_BALANCE")
    if turn_counts != Counter({0:24,1:24,2:24,3:24}) or any(v != 8 for v in cell_counts.values()): raise FreezeFailure("INTERVENTION_BALANCE")
    output_dir.mkdir(parents=True)
    common={"schema_version":1,"protocol_id":manifest["protocol_id"],"status":"CPU_ONLY_STRUCTURAL_SCHEDULE_FROZEN",
            "cohort_manifest_sha256":sha(input_dir/"MANIFEST.json"),"treatment_cohort_sha256":sha(input_dir/"TREATMENT_FIXED_TURN_ROTATION_TRAIN.jsonl"),
            "control_cohort_sha256":sha(input_dir/"CONTROL_NO_TRANSFORM_TRAIN.jsonl"),"schedule_seed":SEED,
            "slots":SLOTS,"intervention_slots":INTERVENTION_SLOTS,"protected_replay_slots":SLOTS-INTERVENTION_SLOTS,
            "gradient_accumulation":GRADIENT_ACCUMULATION,"optimizer_steps":SLOTS//GRADIENT_ACCUMULATION,
            "model_loaded":False,"tokenizer_loaded":False,"gpu_used":False,"optimizer_constructed":False,"execution_authorized":False,
            "token_equality":"NOT_RUN_REQUIRES_FROZEN_V7_TOKENIZER_STATIC_PREFLIGHT",
            "scientific_training_started":False}
    cp=output_dir/"CONTROL_SCHEDULE.json"; tp=output_dir/"TREATMENT_SCHEDULE.json"
    atomic(cp,{**common,"arm":"CONTROL","episodes":control})
    atomic(tp,{**common,"arm":"TREATMENT","episodes":treatment})
    binding={**common,"control_schedule_sha256":sha(cp),"treatment_schedule_sha256":sha(tp),
             "allowed_difference":"only slots 0..95 replace matched no-transform supervision with fixed-turn rotation; slots 96..383 are byte-identical task replay",
             "intervention_turn_counts":dict(sorted(turn_counts.items())),"intervention_train_cell_counts":{"%s|%s"%k:v for k,v in sorted(cell_counts.items())},
             "prelaunch_gates_remaining":["exact_token_and_supervised_token_equality_under_frozen_v7_tokenizer","freeze_hyperparameters_and_runtime_budget","freeze_paired_uncertainty_minimum_effect_retention_runtime_and_cost_rules","new_director_prelaunch_review"]}
    atomic(output_dir/"MATCHED_SCHEDULE_BINDING_V1.json",binding)
    return binding

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--in",dest="input_dir",type=Path,default=DEFAULT_IN); ap.add_argument("--out",type=Path,default=DEFAULT_OUT);a=ap.parse_args()
    print(json.dumps(build(a.input_dir,a.out),sort_keys=True))
if __name__=="__main__": main()
