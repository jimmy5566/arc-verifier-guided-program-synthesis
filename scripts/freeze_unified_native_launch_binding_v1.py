#!/usr/bin/env python3
"""Freeze a no-update launch binding for the unified native baseline."""
from __future__ import annotations
import argparse, hashlib, json, secrets
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/capability_repair_baseline_v1/unified_native_model_capability_baseline_v1"
def sha(p: Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p:Path,v:object)->None:p.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
def main()->int:
 a=argparse.ArgumentParser();a.add_argument("--out",type=Path,default=OUT/"LAUNCH_BINDING_V1.json");a.add_argument("--nonce",default=None);z=a.parse_args()
 manifest=OUT/"SYNTHETIC_BENCHMARK_INPUT_MANIFEST_V1.json";discovery=OUT/"CHECKPOINT_PROVENANCE_DISCOVERY_V1.json";contract=OUT/"EXECUTION_CONTRACT_DRAFT_V1.json";worker=ROOT/"scripts/run_unified_native_model_capability_baseline_v1.py";preflight=ROOT/"scripts/preflight_unified_native_model_capability_baseline_v1.py"
 for path in (manifest,discovery,contract,worker,preflight):
  if not path.is_file():raise RuntimeError(f"REQUIRED_FILE_MISSING:{path}")
 nonce=z.nonce or secrets.token_hex(16)
 if len(nonce)<16 or any(c not in "0123456789abcdef" for c in nonce):raise RuntimeError("NONCE_INVALID")
 root=f"/workspace/arc2/unified_native_model_capability_baseline_v1/run_001_{nonce}"
 v={"schema_version":1,"protocol_id":"UNIFIED_NATIVE_MODEL_CAPABILITY_BASELINE_V1","status":"FROZEN_PENDING_RUNTIME_IDENTITY_PREFLIGHT","worker_source_commit":"7db7c0b36effe20cfcfe003ed7f20c11dc6b0b6d","branch":"infra/arc2-dual-agent-runpod-orchestrator-v1","nonce":nonce,"output_root":root,"output_path":root+"/RAW_UNSCORED.jsonl","receipt_path":root+"/TERMINAL_RECEIPT.json","runtime_cap_seconds":9000,"input_manifest":{"path":str(manifest.relative_to(ROOT)).replace('\\','/'),"sha256":sha(manifest)},"checkpoint_discovery":{"path":str(discovery.relative_to(ROOT)).replace('\\','/'),"sha256":sha(discovery)},"execution_contract":{"path":str(contract.relative_to(ROOT)).replace('\\','/'),"sha256":sha(contract)},"worker":{"path":str(worker.relative_to(ROOT)).replace('\\','/'),"sha256":sha(worker)},"cpu_preflight":{"path":str(preflight.relative_to(ROOT)).replace('\\','/'),"sha256":sha(preflight)},"machine_gates":["exact_runtime_checkout_and_worker_sha256","all_available_adapter_sha256","base_identity","input_manifest_sha256","fresh_output_root","no_duplicate_live_job","final_audit_closed"],"forbidden":["training","optimizer","backward","checkpoint_mutation","TTT","augmentation","DFS","Beam-4","external_selector","Gold","dGold","FINAL_AUDIT"],"authorization":{"gpu_inference_authorized":True,"model_loading_authorized":True,"scientific_training_authorized":False}}
 write(z.out,v);print(json.dumps({"binding":str(z.out),"sha256":sha(z.out),"nonce":nonce},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
