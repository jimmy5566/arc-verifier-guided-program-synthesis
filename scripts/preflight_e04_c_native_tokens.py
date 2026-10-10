"""CPU-only native-serialization preflight for the frozen E04-C schedule.

This does not load a Hugging Face tokenizer or a model.  It verifies the
vendored native runtime files pinned by E04-V3, then uses the repository's
pure-Python 16-token serializer to count every scheduled task and assistant
label.  Its conclusion is scoped to that native runtime contract.
"""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scripts import freeze_e04_c_matched_rotation_schedule as schedule
from training_data import pipeline

DEFAULT_SCHEDULE=ROOT/"experiments/capability_repair_baseline_v1/e04_c_matched_fixed_turn_rotation_repair_pilot_v1/schedule_freeze_v1"
DEFAULT_OUT=DEFAULT_SCHEDULE/"NATIVE_TOKEN_STATIC_PREFLIGHT_V1.json"
IDENTITY=ROOT/"experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json"
CHECKPOINT=ROOT/"experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"
CONFIG=ROOT/"configs/nvarc_native_846d0198"

class PreflightFailure(RuntimeError): pass
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def atomic(p:Path,x:object)->None:
 p.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=p.parent,prefix=p.name+".",suffix=".tmp")
 try:
  with os.fdopen(fd,"w",encoding="utf-8",newline="\n") as f: json.dump(x,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def read_schedule(p:Path)->dict:
 return json.loads(p.read_text(encoding="utf-8"))
def main(schedule_dir:Path=DEFAULT_SCHEDULE,out:Path=DEFAULT_OUT)->dict:
 if out.exists():raise PreflightFailure("REFUSE_OVERWRITE")
 binding=read_schedule(schedule_dir/"MATCHED_SCHEDULE_BINDING_V1.json")
 if binding["token_equality"]!="NOT_RUN_REQUIRES_FROZEN_V7_TOKENIZER_STATIC_PREFLIGHT":raise PreflightFailure("SCHEDULE_GATE_UNEXPECTED")
 runtime=json.loads(IDENTITY.read_text(encoding="utf-8"))
 if runtime["kind"]!="E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY":raise PreflightFailure("RUNTIME_IDENTITY_KIND")
 verified={}
 for item in runtime["files"]:
  path=CONFIG/item["runtime_relative_path"]; actual=sha(path)
  if actual!=item["sha256"]:raise PreflightFailure("E04_RUNTIME_FILE_HASH:"+item["runtime_relative_path"])
  verified[item["runtime_relative_path"]]=actual
 contract=pipeline._verify_native_contract(CONFIG)
 if contract["status"]!="PASS":raise PreflightFailure("NATIVE_SERIALIZATION_CONTRACT")
 checkpoint=json.loads(CHECKPOINT.read_text(encoding="utf-8"))
 base={x["name"]:x["sha256"] for x in checkpoint["base_files"]}
 full_hf_tokenizer_json={"expected_v7_base_sha256":base["tokenizer.json"],"vendored_sha256":sha(CONFIG/"tokenizer.json")}
 full_hf_tokenizer_json["byte_equal"]=full_hf_tokenizer_json["expected_v7_base_sha256"]==full_hf_tokenizer_json["vendored_sha256"]
 control=read_schedule(schedule_dir/"CONTROL_SCHEDULE.json")["episodes"]
 treatment=read_schedule(schedule_dir/"TREATMENT_SCHEDULE.json")["episodes"]
 if len(control)!=len(treatment)!=binding["slots"]:raise PreflightFailure("SCHEDULE_LENGTH")
 rows=[]; bad=[]
 for c,t in zip(control,treatment):
  if c["slot"]!=t["slot"] or c["pair_id"]!=t["pair_id"] or c["intervention_slot"]!=t["intervention_slot"]:raise PreflightFailure("SCHEDULE_ALIGNMENT")
  cs=pipeline.task_to_sample({"source_id":"e04c-control-"+c["pair_id"],**c["control_task"]})
  ts=pipeline.task_to_sample({"source_id":"e04c-treatment-"+t["pair_id"],**t["treatment_task"]})
  equal=(cs["sequence_length"]==ts["sequence_length"] and cs["assistant_token_count"]==ts["assistant_token_count"])
  record={"slot":c["slot"],"pair_id":c["pair_id"],"intervention_slot":c["intervention_slot"],"control_sequence_length":cs["sequence_length"],"treatment_sequence_length":ts["sequence_length"],"control_supervised_token_count":cs["assistant_token_count"],"treatment_supervised_token_count":ts["assistant_token_count"],"equal":equal}
  rows.append(record)
  if not equal:bad.append(record)
 if bad:raise PreflightFailure("PER_SLOT_TOKEN_MISMATCH:"+str(len(bad)))
 total_c=sum(x["control_sequence_length"] for x in rows);total_t=sum(x["treatment_sequence_length"] for x in rows)
 sup_c=sum(x["control_supervised_token_count"] for x in rows);sup_t=sum(x["treatment_supervised_token_count"] for x in rows)
 if total_c!=total_t or sup_c!=sup_t:raise PreflightFailure("ARM_TOTAL_MISMATCH")
 result={"schema_version":1,"protocol_id":binding["protocol_id"],"status":"PASS_NATIVE_RUNTIME_SERIALIZATION_CONTRACT",
  "scope":"Exact counts under the E04-V3 byte-verified native runtime serializer; not a claim that vendored tokenizer.json equals the frozen V7 base tokenizer.json.",
  "model_loaded":False,"hf_tokenizer_loaded":False,"transformers_imported":False,"gpu_used":False,"optimizer_constructed":False,
  "schedule_binding_sha256":sha(schedule_dir/"MATCHED_SCHEDULE_BINDING_V1.json"),"runtime_identity_sha256":sha(IDENTITY),
  "verified_runtime_files":verified,"native_contract":contract,"full_hf_tokenizer_json":full_hf_tokenizer_json,
  "rows":len(rows),"intervention_rows":sum(x["intervention_slot"] for x in rows),
  "control_total_tokens":total_c,"treatment_total_tokens":total_t,"control_supervised_tokens":sup_c,"treatment_supervised_tokens":sup_t,
  "per_slot_equal_token_and_supervision":True,"per_slot_rows":rows,
  "remaining_prelaunch_gates":["freeze_hyperparameters_and_runtime_budget","freeze_paired_uncertainty_minimum_effect_retention_runtime_and_cost_rules","new_director_prelaunch_review"]}
 atomic(out,result);return result
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--schedule-dir",type=Path,default=DEFAULT_SCHEDULE);p.add_argument("--out",type=Path,default=DEFAULT_OUT);a=p.parse_args();print(json.dumps(main(a.schedule_dir,a.out),sort_keys=True))
