#!/usr/bin/env python3
"""Target-blind B1/Batch32 divergence audit for frozen RUN_006 bytes."""
import argparse, hashlib, json
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def first_difference(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y: return i
    return min(len(a),len(b)) if len(a)!=len(b) else None
def main(a):
 raw=Path(a.raw_dir); freeze=json.loads(Path(a.raw_freeze_manifest).read_text())
 if freeze.get('target_sidecar_accessed_before_freeze') is not False: raise SystemExit('FREEZE_TARGET_BOUNDARY')
 receipt=raw/'TERMINAL_RECEIPT.json'
 if sha(receipt)!=freeze['terminal_receipt_sha256']: raise SystemExit('RECEIPT_SHA')
 term=json.loads(receipt.read_text())
 if term.get('status')!='COMPLETE_NO_UPDATE_TARGET_BLIND' or term.get('target_sidecar_accessed') is not False: raise SystemExit('TERMINAL_BOUNDARY')
 report=[]
 for item in term['raw_evidence']:
  arm=item['arm_id']; path=raw/item['path']
  if sha(path)!=item['sha256'] or freeze['raw_target_blind_sha256'].get(item['path'])!=item['sha256']: raise SystemExit('RAW_SHA:'+arm)
  data=read(path); primary={x['episode_id']:x for x in data if x['phase']=='PRIMARY'}; b1=[x for x in data if x['phase']=='B1_SENSITIVITY']
  if len(primary)!=192 or len(b1)!=12: raise SystemExit('CARDINALITY:'+arm)
  for b in b1:
   p=primary[b['episode_id']]
   same_tokens=p['generated_token_ids']==b['generated_token_ids']; same_grid=p.get('parsed_grid')==b.get('parsed_grid')
   if not same_tokens or not same_grid:
    report.append({'arm_id':arm,'episode_id':b['episode_id'],'first_differing_token_position':first_difference(p['generated_token_ids'],b['generated_token_ids']),'primary_completion_length':len(p['generated_token_ids']),'b1_completion_length':len(b['generated_token_ids']),'primary_parser_valid':bool(p.get('parser_valid')),'b1_parser_valid':bool(b.get('parser_valid')),'primary_canonical_grid_sha256':p.get('canonical_grid_sha256'),'b1_canonical_grid_sha256':b.get('canonical_grid_sha256'),'primary_requested_batch_size':32,'primary_effective_batch_size':p['effective_batch_size'],'b1_requested_batch_size':1,'b1_effective_batch_size':b['effective_batch_size'],'target_correctness_confinement':'NOT_EVALUATED_TARGET_BLIND'})
 if len(report)!=3: raise SystemExit('EXPECTED_EXACTLY_THREE_DISCORDANCES:'+str(len(report)))
 out={'schema_version':1,'run_id':freeze['run_id'],'protocol_id':freeze['protocol_id'],'status':'COMPLETE_TARGET_BLIND_CPU_DIVERGENCE_REPORT','raw_freeze_manifest_sha256':sha(a.raw_freeze_manifest),'terminal_receipt_sha256':sha(receipt),'target_sidecar_accessed':False,'target_content_released':False,'discordant_fixed_b1_episodes':report,'discordant_count_by_arm':{arm:sum(x['arm_id']==arm for x in report) for arm in ('V7','E04_E_CONTROL','E04_E_TREATMENT')},'interpretation':'Target-blind descriptive mechanism audit only. It neither changes the B1 validity gate nor assigns target correctness to a divergence.'}
 Path(a.output).write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--raw-dir',required=True); p.add_argument('--raw-freeze-manifest',required=True); p.add_argument('--output',required=True); main(p.parse_args())
