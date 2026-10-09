#!/usr/bin/env python3
"""Authorized no-update teacher-forced NLL diagnostic for three frozen ARC2 adapters."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from training_data.pipeline import IGNORE_INDEX, task_to_sample
WEAK=('connected components','inside/contains','difference','width','orientation')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def token_hash(ids,labels): return hashlib.sha256(json.dumps({'input_ids':ids,'labels':labels},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load_rows(train,schedule):
 data={x['episode_id']:x for x in (json.loads(z) for z in Path(train).read_text(encoding='utf8').splitlines() if z.strip())}
 out=[]
 for entry in sorted(schedule['episodes'],key=lambda x:x['slot']):
  row=data.get(entry['episode_id'])
  if not row or row.get('split')!='TRAIN': raise RuntimeError('SCHEDULE_TRAIN_BINDING_FAIL:'+entry['episode_id'])
  x=task_to_sample({'source_id':row['episode_id'],**row['task']}); labels=x['labels']
  if token_hash(x['input_ids'],labels)!=entry['token_label_sha256']: raise RuntimeError('TOKEN_LABEL_BINDING_FAIL:'+entry['episode_id'])
  if sum(v!=IGNORE_INDEX for v in labels)!=entry['supervised_token_count']: raise RuntimeError('SUPERVISED_TOKEN_BINDING_FAIL:'+entry['episode_id'])
  out.append({'episode_id':entry['episode_id'],'family':entry['family'],'ids':x['input_ids'],'labels':labels,'supervised':entry['supervised_token_count']})
 if len(out)!=400 or {x['family'] for x in out if x['family'] in WEAK}!=set(WEAK): raise RuntimeError('SCHEDULE_SHAPE_FAIL')
 return out
def batches(rows,n):
 for i in range(0,len(rows),n): yield rows[i:i+n]
def collate(rows,torch):
 w=max(len(x['ids']) for x in rows); ids=[]; labels=[]; mask=[]
 for x in rows:
  pad=w-len(x['ids']);ids.append(x['ids']+[13]*pad);labels.append(x['labels']+[IGNORE_INDEX]*pad);mask.append([1]*len(x['ids'])+[0]*pad)
 return torch.tensor(ids,device='cuda:0'),torch.tensor(mask,device='cuda:0'),torch.tensor(labels,device='cuda:0')
def score(model,rows,n,torch,F):
 totals=defaultdict(lambda:[0.0,0,[],0])
 with torch.no_grad():
  for group in batches(rows,n):
   ids,mask,labels=collate(group,torch); logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float(); y=labels[:,1:]; z=logits[:,:-1,:]
   losses=F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1),ignore_index=IGNORE_INDEX,reduction='none').view_as(y)
   for i,row in enumerate(group):
    valid=y[i]!=IGNORE_INDEX; count=int(valid.sum().item())
    if count!=row['supervised']: raise RuntimeError('SHIFTED_SUPERVISION_MISMATCH:'+row['episode_id'])
    val=float(losses[i][valid].sum().item()); t=totals[row['family']];t[0]+=val;t[1]+=count;t[2].append(val/count);t[3]+=1
 return {k:{'nll_sum':v[0],'supervised_tokens':v[1],'token_micro_mean_nll':v[0]/v[1],'row_macro_mean_nll':sum(v[2])/len(v[2]),'rows':v[3]} for k,v in sorted(totals.items())}
def classify(v7,candidates,threshold):
 result={}
 for name,cur in candidates.items():
  fam={}
  for f in WEAK:
   d=cur[f]['token_micro_mean_nll']-v7[f]['token_micro_mean_nll']
   fam[f]='FIT_IMPROVED' if d<=-threshold else ('FIT_WORSE' if d>=threshold else 'NO_MATERIAL_CHANGE')
  result[name]=fam
 return result
def decision(classes,dev):
 fit={name:sum(v=='FIT_IMPROVED' for v in fs.values()) for name,fs in classes.items()}
 if max(fit.values())==0:return 'H1_SUPPORTED'
 for name,fs in classes.items():
  no=sum(fs[f]=='FIT_IMPROVED' and dev[name][f]<=0 for f in WEAK); yes=sum(fs[f]=='FIT_IMPROVED' and dev[name][f]>=1 for f in WEAK)
  if no>=3 and yes==0:return 'H3_SUPPORTED'
 return 'INCONCLUSIVE'
def self_test(cfg):
 schedule=json.loads(Path(cfg['schedule_path']).read_text()); rows=load_rows(cfg['train_path'],schedule)
 assert len(rows)==400 and all(sum(x['family']==f for x in rows)==48 for f in WEAK)
 fake={f:{'token_micro_mean_nll':1.0} for f in WEAK}; worse={f:{'token_micro_mean_nll':1.0} for f in WEAK}; improved={f:{'token_micro_mean_nll':.98} for f in WEAK}; dev0={f:0 for f in WEAK}; dev1={f:1 for f in WEAK}
 assert decision(classify(fake,{'a':worse,'b':worse},.01),{'a':dev0,'b':dev0})=='H1_SUPPORTED'
 assert decision(classify(fake,{'a':improved,'b':worse},.01),{'a':dev0,'b':dev0})=='H3_SUPPORTED'
 assert decision(classify(fake,{'a':improved,'b':worse},.01),{'a':dev1,'b':dev0})=='INCONCLUSIVE'
 return {'status':'PASS','schedule_rows':len(rows),'weak_family_counts':{f:sum(x['family']==f for x in rows) for f in WEAK}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path);ap.add_argument('--self-test',action='store_true');a=ap.parse_args();cfg=json.loads(a.config.read_text())
 for path,key in [('binding_path','binding_sha256'),('train_path','train_sha256'),('schedule_path','schedule_sha256')]:
  if sha(cfg[path])!=cfg[key]: raise RuntimeError('IDENTITY_FAIL:'+path)
 if a.self_test: print(json.dumps(self_test(cfg),sort_keys=True));return
 if not a.out: raise RuntimeError('OUT_REQUIRED')
 if a.out.exists():raise RuntimeError('FRESH_OUTPUT_REQUIRED')
 terminal={'schema_version':1,'stage_id':cfg['stage_id'],'status':'PRE_MODEL_FAILURE','optimizer_steps':0,'generation_performed':False,'checkpoint_mutated':False,'final_audit_opened':False,'scientific_gpu_training_seconds':0.0}
 try:
  binding=json.loads(Path(cfg['binding_path']).read_text());schedule=json.loads(Path(cfg['schedule_path']).read_text()); rows=load_rows(cfg['train_path'],schedule)
  import torch, torch.nn.functional as F
  from transformers import AutoModelForCausalLM
  from peft import PeftModel
  if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA_BF16_UNAVAILABLE')
  os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false'
  cross=rows[:0]
  for f in WEAK: cross += [x for x in rows if x['family']==f][:6]
  outputs={};t0=time.monotonic()
  for name,c in cfg['conditions'].items():
   if sha(c['manifest_path'])!=c['manifest_sha256'] or sha(c['result_path'])!=c['result_sha256']:raise RuntimeError('CONDITION_IDENTITY_FAIL:'+name)
   manifest=json.loads(Path(c['manifest_path']).read_text()); adapter=Path(manifest['adapter_path'])/'adapter_model.safetensors'
   if sha(adapter)!=c['adapter_sha256']:raise RuntimeError('ADAPTER_IDENTITY_FAIL:'+name)
   model=AutoModelForCausalLM.from_pretrained(manifest['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0');model=PeftModel.from_pretrained(model,manifest['adapter_path'],is_trainable=False);model.eval()
   full=score(model,rows,cfg['requested_batch_size'],torch,F); b1=score(model,cross,1,torch,F); eff=score(model,cross,cfg['requested_batch_size'],torch,F); rep=score(model,cross,cfg['requested_batch_size'],torch,F)
   drift=max(abs(b1[f]['token_micro_mean_nll']-eff[f]['token_micro_mean_nll']) for f in WEAK); drift=max(drift,max(abs(eff[f]['token_micro_mean_nll']-rep[f]['token_micro_mean_nll']) for f in WEAK))
   outputs[name]={'full':full,'batch_validation':{'rows':len(cross),'batch1':b1,'effective_batch':eff,'repeat_effective_batch':rep,'max_absolute_mean_nll_drift':drift,'status':'PASS' if drift<=cfg['drift_tolerance'] else 'NUMERICAL_DRIFT_FAIL'}}
   del model;torch.cuda.empty_cache()
  if any(x['batch_validation']['status']!='PASS' for x in outputs.values()): outcome='INCONCLUSIVE'
  else:
   classes=classify(outputs['V7']['full'],{k:v['full'] for k,v in outputs.items() if k!='V7'},cfg['fit_threshold']);dev=cfg['frozen_dev_paired_net_fixes'];outcome=decision(classes,dev)
  result={'schema_version':1,'stage_id':cfg['stage_id'],'status':'COMPLETE','binding_sha256':cfg['binding_sha256'],'conditions':outputs,'fit_classes':classes if 'classes' in locals() else None,'decision':outcome,'generation_performed':False,'optimizer_steps':0,'checkpoint_mutated':False,'final_audit_opened':False,'wall_seconds':time.monotonic()-t0}
  atomic(a.out/'ATOMIC_PREREQUISITE_FIT_GENERALIZATION_DIAGNOSTIC_RESULT.json',result);terminal.update({'status':'COMPLETE','result_sha256':sha(a.out/'ATOMIC_PREREQUISITE_FIT_GENERALIZATION_DIAGNOSTIC_RESULT.json'),'decision':outcome})
 except Exception as e: terminal.update({'status':'PRE_MODEL_FAILURE' if terminal['status']=='PRE_MODEL_FAILURE' else 'FAILURE','error':type(e).__name__+':'+str(e),'traceback':traceback.format_exc(limit=5)})
 finally:
  atomic(a.out/'TERMINAL_RECEIPT.json',terminal)
 if terminal['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()
