#!/usr/bin/env python3
"""Audit the nominal optimization signal of a frozen ARC2 training schedule.

This CPU-only audit reconstructs every scheduled example from TRAIN rows and
matches the worker's token-label bindings. It reports loss coefficients implied
by the worker's per-step supervised-token normalization; it does not infer
actual gradient direction or learning efficacy.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from training_data.pipeline import IGNORE_INDEX, task_to_sample

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(v): return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
def token_hash(ids,labels): return hashlib.sha256(canonical({'input_ids':ids,'labels':labels})).hexdigest()
def dump(path:Path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--schedule',type=Path,required=True);p.add_argument('--worker',type=Path,required=True);p.add_argument('--out-json',type=Path,required=True);p.add_argument('--out-md',type=Path,required=True);a=p.parse_args()
 schedule=json.loads(a.schedule.read_text(encoding='utf-8')); rows={r['episode_id']:r for r in (json.loads(x) for x in a.train.read_text(encoding='utf-8').splitlines() if x)}
 eps=schedule['episodes']; assert len(eps)%4==0 and len(eps)==400
 worker=a.worker.read_text(encoding='utf-8')
 required=['denom = sum(x["supervised"] for x in group)','(loss * (row["supervised"] / denom)).backward()','optimizer.step(); steps += 1']
 if any(x not in worker for x in required): raise RuntimeError('WORKER_LOSS_CONTRACT_NOT_RECOGNIZED')
 per=defaultdict(lambda: {'episodes':0,'transformer_tokens':0,'supervised_tokens':0,'nominal_loss_coefficient':0.0,'steps_present':set(),'within_step_weight_min':1.0,'within_step_weight_max':0.0})
 role=defaultdict(lambda: {'episodes':0,'transformer_tokens':0,'supervised_tokens':0,'nominal_loss_coefficient':0.0,'steps_present':set()})
 steps=[]
 for step,group in enumerate([eps[i:i+4] for i in range(0,len(eps),4)]):
  verified=[]
  for e in group:
   source=rows.get(e['episode_id'])
   if source is None or source.get('split')!='TRAIN': raise RuntimeError('SCHEDULE_NOT_TRAIN_ONLY')
   sample=task_to_sample({'source_id':source['episode_id'],**source['task']});labels=sample['labels']
   observed={'sequence_length':len(sample['input_ids']),'supervised_token_count':sample['assistant_token_count'],'token_label_sha256':token_hash(sample['input_ids'],labels)}
   if observed!={k:e[k] for k in observed}: raise RuntimeError('TOKENIZER_LABEL_BINDING_MISMATCH')
   if not any(x!=IGNORE_INDEX for x in labels): raise RuntimeError('ZERO_SUPERVISION')
   verified.append((e,observed))
  denom=sum(o['supervised_token_count'] for _,o in verified)
  mixes=[]
  for e,o in verified:
   fam=e['family']; r=e['objective_role']; w=o['supervised_token_count']/denom; coeff=w/(len(eps)//4)
   for bucket,key in ((per[fam],fam),(role[r],r)):
    bucket['episodes']+=1;bucket['transformer_tokens']+=o['sequence_length'];bucket['supervised_tokens']+=o['supervised_token_count'];bucket['nominal_loss_coefficient']+=coeff;bucket['steps_present'].add(step)
   per[fam]['within_step_weight_min']=min(per[fam]['within_step_weight_min'],w);per[fam]['within_step_weight_max']=max(per[fam]['within_step_weight_max'],w)
   mixes.append({'family':fam,'objective_role':r,'supervised_tokens':o['supervised_token_count'],'within_step_loss_weight':w,'global_nominal_loss_coefficient':coeff})
  steps.append({'optimizer_step':step+1,'supervised_token_denominator':denom,'family_mix':mixes})
 def finish(d):
  total_sup=sum(x['supervised_tokens'] for x in d.values())
  for x in d.values():
   x['steps_present']=len(x['steps_present']);x['supervised_token_share']=x['supervised_tokens']/total_sup
   x['nominal_loss_percent']=100*x['nominal_loss_coefficient']
  return dict(sorted(d.items()))
 fam=finish(per); roles=finish(role)
 nominal=sum(x['nominal_loss_coefficient'] for x in fam.values())
 doc={'schema_version':1,'audit_id':'ATOMIC_PREREQUISITE_REPAIR_V1_TRAINING_SIGNAL_AUDIT','status':'CPU_AUDIT_COMPLETE','scope':'FROZEN_SCHEDULE_AND_WORKER_LOSS_IMPLEMENTATION','schedule_sha256':sha(a.schedule),'train_sha256':sha(a.train),'worker_sha256':sha(a.worker),'reconstruction_checks':{'episodes':len(eps),'optimizer_steps':len(steps),'gradient_accumulation':4,'all_sources_train_only':True,'tokenizer_label_binding_verified':True,'worker_loss_contract_verified':True,'nominal_coefficient_sum':nominal},'loss_contract':{'per_example_loss':'The Transformers causal-LM loss is mean-reduced over non-ignored labels for each row.','within_optimizer_step':'The worker multiplies each row loss by its supervised-token count divided by the four-row step denominator. Therefore every supervised token in a step has equal nominal scalar-loss weight.','across_optimizer_steps':'Each of 100 optimizer steps contributes one normalized scalar loss; families receive the sum of their within-step weights divided by 100. Different step denominators make this differ from global supervised-token share.','not_measured':['gradient magnitude','gradient direction or cosine','parameter interference','effective learning rate by family','train-to-DEV generalization']},'by_objective_role':roles,'by_family':fam,'per_optimizer_step':steps,'mechanism_assessment':{'ESTABLISHED':['All five weak families have 48 episodes, but equal episode count does not imply equal nominal loss coefficient because output supervision lengths differ and losses are token-weighted within each accumulation group.','The reported coefficients are exact properties of this frozen schedule and worker implementation, not gradient measurements.'],'SUPPORTED_HYPOTHESIS':['Families with much lower nominal loss coefficient may receive less optimization emphasis under this fixed schedule; whether that explains DEV outcomes requires controlled evidence.'],'UNKNOWN':['H2 negative gradient interference: no gradient-cosine or equivalent measurement exists.','H3 poor task/generalization despite supervised fitting: per-family train loss and post-round DEV results are not yet available.']},'candidate_successor_only_if_stage_review_justifies':{'intervention':'Family-balanced scalar-loss aggregation while retaining exactly the V7 start, same TRAIN rows/order, total tokens, optimizer, LR, LoRA, seed, duration and evaluation.','normalization':'Normalize family weights so each optimizer step retains total scalar loss scale 1.0; do not conflate family reweighting with a global LR change.','not_authorized_now':True}}
 dump(a.out_json,doc)
 lines=['# Atomic-prerequisite training-signal audit','','## What the frozen worker optimizes','','Each four-example accumulation group uses each row\'s mean token loss multiplied by `supervised_tokens / group_supervised_tokens`. Each optimizer step is equally weighted. Thus equal episode counts do **not** create equal nominal objective weight.','','| Family | Episodes | Transformer tokens | Supervised tokens | Nominal loss share | Steps present |','|---|---:|---:|---:|---:|---:|']
 for name,x in fam.items(): lines.append(f"| {name} | {x['episodes']} | {x['transformer_tokens']} | {x['supervised_tokens']} | {x['nominal_loss_percent']:.2f}% | {x['steps_present']} |")
 lines += ['','## Interpretation','','- **Established:** the table is a deterministic property of the frozen schedule, tokenizer labels, and worker scaling.','- **Not established:** gradient conflict, gradient size, and learning efficacy were not measured. Nominal loss share is not a gradient measurement.','- **Current round:** remains unchanged. A family-balanced loss comparison is only a candidate for a separately reviewed successor after this round is evaluated.']
 a.out_md.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
 print(json.dumps({'status':'PASS','families':len(fam),'nominal_coefficient_sum':nominal,'out_json':str(a.out_json)},sort_keys=True))
if __name__=='__main__':main()
