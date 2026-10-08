#!/usr/bin/env python3
"""Target-blind post-evaluation paired analysis for frozen ARC2 result files."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf-8');temp.replace(path)
def paired(base: list[dict], candidate: list[dict]) -> dict:
    b={x['episode_id']:x for x in base}; c={x['episode_id']:x for x in candidate}
    if set(b)!=set(c): raise RuntimeError('EPISODE_SET_MISMATCH')
    n01=sum(not bool(b[k]['exact_grid_match']) and bool(c[k]['exact_grid_match']) for k in b)
    n10=sum(bool(b[k]['exact_grid_match']) and not bool(c[k]['exact_grid_match']) for k in b)
    return {'episodes':len(b),'baseline_exact':sum(bool(x['exact_grid_match']) for x in b.values()),'candidate_exact':sum(bool(x['exact_grid_match']) for x in c.values()),'paired_fixes_n01':n01,'paired_harms_n10':n10,'net_paired_fixes':n01-n10}
def rows_for(rows: list[dict], label: str) -> list[dict]:
    if label=='TARGET_DEV': return [x for x in rows if x['surface']=='TARGET_DEV']
    if label=='RETENTION_SENTINEL': return [x for x in rows if x['surface']=='RETENTION_SENTINEL']
    return [x for x in rows if f':{label}:' in x['episode_id']]
def analyze(base: dict, candidate: dict) -> dict:
    if base.get('status')!='COLLECTED_PASS' or candidate.get('status')!='COLLECTED_PASS': raise RuntimeError('RESULT_STATUS_INVALID')
    b=base['predictions'];c=candidate['predictions']
    surfaces={label:paired(rows_for(b,label),rows_for(c,label)) for label in ('TARGET_DEV','RETENTION_SENTINEL')}
    roles={label:paired(rows_for(b,label),rows_for(c,label)) for label in ('TARGETED_EVALUATION','TARGETED_COMPOSITION','RETENTION_SENTINEL')}
    retention=surfaces['RETENTION_SENTINEL']; composition=roles['TARGETED_COMPOSITION']
    gates={'retention_recovery':retention['candidate_exact']>=70 and retention['net_paired_fixes']>=0,'composition_preservation':composition['candidate_exact']>=34 and composition['net_paired_fixes']>=3}
    return {'schema_version':1,'status':'COMPLETE','baseline':{'result_sha256':None},'candidate':{'result_sha256':None},'paired_by_surface':surfaces,'paired_by_role':roles,'prospective_r2_gates':gates,'prospective_r2_gate_status':'PASS' if all(gates.values()) else 'FAIL','final_audit_opened':False}
def main() -> int:
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 result=analyze(json.loads(args.baseline.read_text()),json.loads(args.candidate.read_text()));result['baseline']['result_sha256']=sha(args.baseline);result['candidate']['result_sha256']=sha(args.candidate);dump(args.output,result);return 0
if __name__=='__main__':raise SystemExit(main())
