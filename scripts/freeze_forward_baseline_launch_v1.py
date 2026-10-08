#!/usr/bin/env python3
"""Freeze and verify the one-shot forward-baseline evaluation launch contract.

This utility is deliberately read-only with respect to models and datasets.  It
captures a complete direct-file checkpoint manifest, creates an immutable launch
contract, and validates that contract before any model library import.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

TARGET_SHA = "e68b6d389fb51fe7f87311e13bd2630af2ce32c2e09b6314005d89983f924a97"
RETENTION_SHA = "b41c25a05e798d6b5f238b013add54a75aefa3cd0ddfbc74fbbc9f112de80526"
BASELINE_SHA = "d1ca26d0a5f1651d2987e4a0555a4751caa1ca1785f5d46e77ab83fc3dacaf3b"
INFERENCE_SHA = "ff10ac5fc7291a667e2d23864a080ddb16bb719c31abd2337ebfe824f9e6361e"

def canon(x: Any) -> bytes: return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
def digest_bytes(x: bytes) -> str: return hashlib.sha256(x).hexdigest()
def digest(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2)+"\n",encoding="utf-8",newline="\n"); os.replace(tmp,path)
def files(root: Path) -> list[dict[str,Any]]:
    if not root.is_dir(): raise RuntimeError(f"CHECKPOINT_ROOT_MISSING:{root}")
    out=[]
    for p in sorted(root.iterdir()):
        if not p.is_file(): raise RuntimeError(f"CHECKPOINT_NONFILE_ENTRY:{p.name}")
        out.append({"name":p.name,"bytes":p.stat().st_size,"sha256":digest(p)})
    if not out: raise RuntimeError("CHECKPOINT_EMPTY")
    return out
def git_head(root: Path) -> str:
    return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
def data_meta(path: Path, roles: set[str], expected: str, count: int) -> dict[str,Any]:
    if not path.is_file() or digest(path)!=expected: raise RuntimeError(f"DATASET_HASH_MISMATCH:{path}")
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows)!=count or any(x.get("role") not in roles for x in rows): raise RuntimeError(f"DATASET_ROLE_OR_COUNT_MISMATCH:{path}")
    ids=[str(x.get("episode_id", "")) for x in rows]
    if len(ids)!=len(set(ids)) or any(not x for x in ids): raise RuntimeError(f"DATASET_IDENTITY_MISMATCH:{path}")
    return {"path":str(path),"sha256":expected,"bytes":path.stat().st_size,"rows":count,"roles":sorted(roles),"episode_ids_sha256":digest_bytes(canon(ids))}
def capture(a: argparse.Namespace) -> int:
    base=Path(a.base); adapter=Path(a.adapter); source=Path(a.source)
    payload={"schema_version":1,"kind":"FORWARD_BASELINE_CHECKPOINT_MANIFEST","captured_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"source_commit":git_head(source),"base_path":str(base),"base_files":files(base),"adapter_path":str(adapter),"adapter_files":files(adapter)}
    payload["manifest_sha256"]=digest_bytes(canon({k:v for k,v in payload.items() if k!="manifest_sha256"}))
    atomic(Path(a.output),payload); print(json.dumps({"status":"PASS","manifest_sha256":payload["manifest_sha256"]},sort_keys=True)); return 0
def create(a: argparse.Namespace) -> int:
    manifest=json.loads(Path(a.manifest).read_text(encoding="utf-8")); mp=Path(a.manifest); bi=Path(a.baseline_identity)
    if not bi.is_file() or digest(bi)!=BASELINE_SHA: raise RuntimeError("BASELINE_IDENTITY_MISMATCH")
    if manifest.get("kind")!="FORWARD_BASELINE_CHECKPOINT_MANIFEST": raise RuntimeError("MANIFEST_KIND_INVALID")
    td=data_meta(Path(a.target_dev), {"TARGETED_EVALUATION","TARGETED_COMPOSITION"}, TARGET_SHA, 192)
    rt=data_meta(Path(a.retention), {"RETENTION_SENTINEL"}, RETENTION_SHA, 96)
    source=Path(a.source); collector=source/"scripts/collect_capability_repair_baseline_v1.py"; prompt=source/"configs/nvarc_native_846d0198/chat_template.j2"; parser=source/"src/inference/arc_native_io.py"; contract=source/"experiments/base_only_targeted_repair_remote_first_v2/execution_contract_v7/BASELINE_INFERENCE_CONTRACT_V2.json"
    for p in (collector,prompt,parser,contract):
        if not p.is_file(): raise RuntimeError(f"REQUIRED_SOURCE_MISSING:{p}")
    root=Path(a.output_root)
    if any(p.exists() for p in (root, Path(a.receipt))): raise RuntimeError("FRESH_OUTPUT_COLLISION")
    if not Path(a.nonce).is_file(): raise RuntimeError("FRESH_NONCE_MISSING")
    review_path=Path(a.governor_review).resolve()
    payload={"schema_version":1,"contract_id":"FORWARD_BASELINE_EVALUATION_LAUNCH_V3","directive_id":"DIRECTOR_DIRECTIVE_028","directive_sha256":a.directive_sha256,"reviewed_brief_sha256":a.brief_sha256,"baseline_identity":{"path":str(bi),"sha256":BASELINE_SHA},"source_commit":git_head(source),"source_root":str(source),"checkpoint_manifest_path":str(mp),"checkpoint_manifest_sha256":digest(mp),"checkpoint_manifest_identity":manifest["manifest_sha256"],"datasets":{"TARGET_DEV":td,"RETENTION_SENTINEL":rt},"inference_contract":{"path":str(contract),"sha256":digest(contract),"expected_sha256":INFERENCE_SHA},"executable":{"path":str(collector),"sha256":digest(collector)},"dependencies":{"prompt_template":{"path":str(prompt),"sha256":digest(prompt)},"parser":{"path":str(parser),"sha256":digest(parser)},"baseline_identity":{"path":str(bi),"sha256":BASELINE_SHA}},"interpreter":a.interpreter,"contract_path":str(Path(a.output).resolve()),"governor_review_path":str(review_path),"argv":[a.interpreter,str(collector),"--launch-contract",str(Path(a.output).resolve()),"--governor-review",str(review_path),"--output",str(root/"CAPABILITY_REPAIR_BASELINE_V1_RESULTS.json"),"--receipt",str(Path(a.receipt).resolve()),"--runtime-limit-seconds",str(int(a.runtime_cap_seconds))],"environment":{"PYTHONHASHSEED":"0","TOKENIZERS_PARALLELISM":"false","CUDA_VISIBLE_DEVICES":"0"},"runtime_cap_seconds":int(a.runtime_cap_seconds),"nonce_path":str(Path(a.nonce).resolve()),"nonce_sha256":a.nonce_sha256,"output_root":str(root.resolve()),"receipt_path":str(Path(a.receipt).resolve()),"forbidden":{"training":True,"optimizer":True,"lora_construction":True,"final_audit":True,"diagnostic_gold_training":True,"automatic_follow_on_training":True},"authorization":{"mode":"EXTERNAL_GOVERNOR_REVIEW_REQUIRED","embedded_authorization_forbidden":True},"expected":{"target_dev_rows":192,"retention_rows":96,"total_rows":288},"status":"FROZEN_PENDING_LAUNCH_AUTHORIZATION"}
    payload["contract_sha256"]=digest_bytes(canon({k:v for k,v in payload.items() if k!="contract_sha256"}))
    atomic(Path(a.output),payload); print(json.dumps({"status":"PASS","contract_sha256":payload["contract_sha256"]},sort_keys=True)); return 0
def verify(a: argparse.Namespace) -> int:
    c=json.loads(Path(a.contract).read_text(encoding="utf-8")); source=Path(c["source_root"])
    if c.get("contract_id") not in {"FORWARD_BASELINE_EVALUATION_LAUNCH_V1","FORWARD_BASELINE_EVALUATION_LAUNCH_V2","FORWARD_BASELINE_EVALUATION_LAUNCH_V3"}: raise RuntimeError("CONTRACT_ID_INVALID")
    if digest_bytes(canon({k:v for k,v in c.items() if k!='contract_sha256'})) != c.get('contract_sha256'): raise RuntimeError("CONTRACT_INTERNAL_IDENTITY_MISMATCH")
    if digest(Path(c['baseline_identity']['path'])) != c['baseline_identity']['sha256']: raise RuntimeError("BASELINE_IDENTITY_MISMATCH")
    if c.get("inference_contract",{}).get("sha256") != INFERENCE_SHA: raise RuntimeError("INFERENCE_CONTRACT_IDENTITY_INVALID")
    if git_head(source)!=c["source_commit"]: raise RuntimeError("SOURCE_COMMIT_MISMATCH")
    if digest(Path(c["executable"]["path"])) != c["executable"]["sha256"]: raise RuntimeError("EXECUTABLE_HASH_MISMATCH")
    for v in c["dependencies"].values():
        if digest(Path(v["path"])) != v["sha256"]: raise RuntimeError("DEPENDENCY_HASH_MISMATCH")
    m=json.loads(Path(c["checkpoint_manifest_path"]).read_text(encoding="utf-8"))
    if digest(Path(c["checkpoint_manifest_path"]))!=c["checkpoint_manifest_sha256"]: raise RuntimeError("CHECKPOINT_MANIFEST_HASH_MISMATCH")
    for key, rootkey, filekey in (("base_files","base_path","base_files"),("adapter_files","adapter_path","adapter_files")):
        actual=files(Path(m[rootkey])); expected=m[filekey]
        if actual!=expected: raise RuntimeError(f"CHECKPOINT_CONTENT_MISMATCH:{key}")
    for key, roles, expected, count in (("TARGET_DEV",{"TARGETED_EVALUATION","TARGETED_COMPOSITION"},TARGET_SHA,192),("RETENTION_SENTINEL",{"RETENTION_SENTINEL"},RETENTION_SHA,96)):
        got=data_meta(Path(c["datasets"][key]["path"]),roles,expected,count)
        if got != c["datasets"][key]: raise RuntimeError(f"DATASET_BINDING_MISMATCH:{key}")
    if any(Path(x).exists() for x in (c["output_root"],c["receipt_path"])): raise RuntimeError("OUTPUT_COLLISION")
    nonce=Path(c["nonce_path"])
    if not nonce.is_file() or digest(nonce)!=c["nonce_sha256"]: raise RuntimeError("NONCE_INVALID")
    print(json.dumps({"status":"PASS_NO_MODEL_IMPORT","contract_sha256":c["contract_sha256"],"source_commit":c["source_commit"]},sort_keys=True)); return 0
def main() -> int:
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="mode",required=True)
    x=sub.add_parser("capture"); x.add_argument("--source",required=True); x.add_argument("--base",required=True); x.add_argument("--adapter",required=True); x.add_argument("--output",required=True)
    x=sub.add_parser("create"); x.add_argument("--source",required=True); x.add_argument("--baseline-identity",required=True); x.add_argument("--manifest",required=True); x.add_argument("--target-dev",required=True); x.add_argument("--retention",required=True); x.add_argument("--output",required=True); x.add_argument("--output-root",required=True); x.add_argument("--receipt",required=True); x.add_argument("--nonce",required=True); x.add_argument("--nonce-sha256",required=True); x.add_argument("--interpreter",required=True); x.add_argument("--directive-sha256",required=True); x.add_argument("--brief-sha256",required=True); x.add_argument("--governor-review",required=True); x.add_argument("--runtime-cap-seconds",default="7200")
    x=sub.add_parser("verify"); x.add_argument("--contract",required=True)
    a=p.parse_args(); return {"capture":capture,"create":create,"verify":verify}[a.mode](a)
if __name__=="__main__": raise SystemExit(main())
