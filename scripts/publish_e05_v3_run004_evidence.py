#!/usr/bin/env python3
"""Publish the already-terminal E05 V3 RUN_004 evidence through GitHub.

This is transport only: it validates the five frozen remote output bytes, copies
no targets, and performs one ordinary non-force commit to the approved branch.
"""
import argparse, hashlib, json, shutil, subprocess, tempfile
from pathlib import Path

PROTOCOL = "E05_V3_SEMANTICALLY_IDENTIFIABLE_DEMONSTRATIONS_AND_ADAPTER_STORAGE_RECOVERY"
RUN_ID = "run_004_33d7c84be6e74b7bbe76001001fe0aad"
DEST_REL = Path("experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery") / "RUN_004_REMOTE_EVIDENCE"
EXPECTED = {
    "TERMINAL_RECEIPT.json": "09e3ec729a1bf21ed259f47b49342db533a8490496efc8499642fb5d88fadc9c",
    "LAUNCH_CAP_RECEIPT.json": "91fafada998b2d0901bb8eb04a7c644e2e60a6848e419de6ea690314086e920c",
    "RAW_V7_NO_E04E_UPDATE_RAW_GENERATIONS.jsonl": "c951c870fb090d573c97bc27e7f389e686cea44cc6df29585586a9a695cb188c",
    "E04E_RUN003_CONTROL_EQUAL_SLOT_RAW_GENERATIONS.jsonl": "47edee3d176f5fc72d8f2c519d17f95db8ed56f188bdb66bda28abb272be67df",
    "E04E_RUN003_TREATMENT_EQUAL_SLOT_RAW_GENERATIONS.jsonl": "30745fc53fe6dd762a428515abc874985d0ff0011402e338b4187296519f7c9e",
}

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def run(args, cwd=None):
    return subprocess.run(args,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True).stdout

def validate_source(root: Path):
    root=root.resolve()
    got={}
    for name, expected in EXPECTED.items():
        p=root/name
        if not p.is_file(): raise RuntimeError("MISSING_FROZEN_E05_EVIDENCE:"+name)
        actual=sha(p)
        if actual != expected: raise RuntimeError("E05_EVIDENCE_SHA256_MISMATCH:"+name)
        got[name]={"sha256":actual,"bytes":p.stat().st_size}
    terminal=json.loads((root/'TERMINAL_RECEIPT.json').read_text(encoding='utf-8'))
    if terminal.get('protocol_id') != PROTOCOL or terminal.get('status') != 'COMPLETE_NO_UPDATE':
        raise RuntimeError('E05_TERMINAL_RECEIPT_PROTOCOL_OR_STATUS_INVALID')
    if any(terminal.get(k) not in (0,False) for k in ('optimizer_steps','parameter_updates','backward_calls','target_sidecar_accessed','final_audit_opened')):
        raise RuntimeError('E05_TERMINAL_RECEIPT_NO_UPDATE_GATE_FAILED')
    return got

def manifest(got):
    return {"schema_version":1,"transport":"GITHUB_NON_FORCE_COMMIT","protocol_id":PROTOCOL,"run_id":RUN_ID,"target_sidecar_accessed_by_transport":False,"files":got}

def publish(source: Path, repo: Path, branch: str):
    got=validate_source(source)
    repo=repo.resolve(); run(['git','fetch','origin',branch],repo)
    with tempfile.TemporaryDirectory(prefix='arc2-e05-run004-publish-') as td:
        tree=Path(td)/'tree'
        run(['git','worktree','add','--detach',str(tree),'origin/'+branch],repo)
        try:
            dest=tree/DEST_REL
            if dest.exists(): raise RuntimeError('REFUSE_OVERWRITE_EXISTING_E05_RUN004_EVIDENCE')
            dest.mkdir(parents=True)
            for name in EXPECTED: shutil.copyfile(source/name,dest/name)
            (dest/'REMOTE_TRANSPORT_MANIFEST.json').write_text(json.dumps(manifest(got),sort_keys=True,indent=2)+'\n',encoding='utf-8')
            run(['git','add',str(DEST_REL)],tree)
            run(['git','-c','user.name=ARC2 Evidence Publisher','-c','user.email=arc2-evidence@localhost','commit','-m','Preserve E05 V3 RUN_004 raw evidence'],tree)
            run(['git','push','origin','HEAD:refs/heads/'+branch],tree)
            return run(['git','rev-parse','HEAD'],tree).strip()
        finally:
            run(['git','worktree','remove','--force',str(tree)],repo)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--repo',type=Path)
    ap.add_argument('--branch')
    ap.add_argument('--validate-only',action='store_true')
    a=ap.parse_args(); got=validate_source(a.source_root)
    if a.validate_only:
        print(json.dumps(manifest(got),sort_keys=True)); return
    if not a.repo or not a.branch: raise SystemExit('--repo and --branch required unless --validate-only')
    print(json.dumps({"published_commit":publish(a.source_root,a.repo,a.branch)},sort_keys=True))
if __name__=='__main__': main()
