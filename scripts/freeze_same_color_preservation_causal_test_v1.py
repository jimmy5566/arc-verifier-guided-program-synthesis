#!/usr/bin/env python3
"""Freeze matched CPU-only schedules for SAME_COLOR_PRESERVATION_CAUSAL_TEST_V1.

No Torch, Transformers, CUDA, model, optimizer, or evaluation imports. The two
schedules are identical except designated replay slots, where an equal-length,
equal-supervision same-color row replaces a non-same-color replay row.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, tempfile
from collections import Counter, defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from training_data.pipeline import IGNORE_INDEX, task_to_sample

def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix='.tmp-',suffix='.json',dir=path.parent)
 try:
  with os.fdopen(fd,'w',encoding='utf8',newline='\n') as f: json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def weighted(weights,seed):
 names=sorted(weights);return random.Random(seed).choices(names,weights=[float(weights[n]) for n in names],k=1)[0]
def load(train):
 rows=[json.loads(line) for line in Path(train).read_text(encoding='utf8').splitlines() if line.strip()]
 pool=defaultdict(lambda:defaultdict(list))
 for r in rows:
  s=task_to_sample({'source_id':r['episode_id'],**r['task']})
  if not any(x!=IGNORE_INDEX for x in s['labels']):raise RuntimeError('ZERO_SUPERVISION')
  pool[r['role']][r['family']].append({'episode_id':r['episode_id'],'role':r['role'],'family':r['family'],'sequence_length':len(s['input_ids']),'supervised_token_count':s['assistant_token_count'],'token_label_sha256':digest({'input_ids':s['input_ids'],'labels':s['labels']})})
 for byfamily in pool.values():
  for values in byfamily.values():values.sort(key=lambda x:x['episode_id'])
 return pool
def totals(entries):
 role=Counter(x['role'] for x in entries);fam=Counter(x['family'] for x in entries)
 return {'episodes':len(entries),'transformer_tokens':sum(x['sequence_length'] for x in entries),'supervised_tokens':sum(x['supervised_token_count'] for x in entries),'by_role':dict(sorted(role.items())),'by_family':dict(sorted(fam.items()))}
def build(config,pool):
 shared=config['shared_training'];seed=int(shared['seed']);cycle=shared['role_cycle'];target=int(shared['nominal_transformer_tokens_per_arm']);acc=int(shared['gradient_accumulation'])
 if len(cycle)!=20 or cycle.count('RETENTION_TRAIN')!=4:raise RuntimeError('ROLE_CYCLE_INVALID')
 common_weights={'ATOMIC_REPAIR':shared['atomic_family_weights'],'COMPOSITION_REPAIR':shared['composition_family_weights']}
 replay_control=config['control_replay_family_weights'];non_same=[]
 for family in replay_control:
  non_same.extend(pool['RETENTION_TRAIN'].get(family,[]))
 if not pool['ATOMIC_REPAIR'] or not pool['COMPOSITION_REPAIR'] or not non_same or not pool['ATOMIC_REPAIR'].get('same color'):raise RuntimeError('REQUIRED_FAMILY_UNAVAILABLE')
 # Map every same-color candidate to all valid equal-token, equal-supervision controls.
 match={}
 for t in pool['ATOMIC_REPAIR']['same color']:
  valid=[c for c in non_same if c['sequence_length']==t['sequence_length'] and c['supervised_token_count']==t['supervised_token_count']]
  if valid:match[t['episode_id']]=valid
 if not match:raise RuntimeError('NO_EQUAL_TOKEN_CONTROL_MATCH')
 cursor=Counter();pair_cursor=Counter();control=[];treatment=[];common_total=0;slot=0;replay_slot=0
 alignment=int(shared.get('schedule_alignment_slots', acc))
 if alignment<=0 or alignment%acc: raise RuntimeError('SCHEDULE_ALIGNMENT_INVALID')
 while common_total<target or slot%acc or slot%alignment:
  role=cycle[slot%len(cycle)]
  if role!='RETENTION_TRAIN':
   weights=common_weights[role];family=weighted(weights,f'{seed}:{role}:{slot}');items=pool[role].get(family)
   if not items:raise RuntimeError('FAMILY_UNAVAILABLE:'+family)
   x=items[cursor[(role,family)]%len(items)];cursor[(role,family)]+=1
   c=t=dict(x);c['slot']=t['slot']=slot;c['intervention_slot']=t['intervention_slot']=False
  else:
   # Exact 20% of replay slots, established before any schedule materialization.
   intervention=(replay_slot%5==0)
   if intervention:
    candidates=sorted(match)
    tid=candidates[pair_cursor['same']%len(candidates)];pair_cursor['same']+=1;t=next(x for x in pool['ATOMIC_REPAIR']['same color'] if x['episode_id']==tid)
    options=match[tid];c0=options[pair_cursor[tid]%len(options)];pair_cursor[tid]+=1
    c=dict(c0);t=dict(t);t['source_role']=t['role'];t['role']='RETENTION_TRAIN';c['slot']=t['slot']=slot;c['intervention_slot']=t['intervention_slot']=True;c['paired_replacement_for']=t['episode_id'];t['paired_replacement_for']=c['episode_id']
   else:
    family=weighted(replay_control,f'{seed}:RETENTION_TRAIN:{replay_slot}');items=pool['RETENTION_TRAIN'].get(family)
    if not items:raise RuntimeError('REPLAY_FAMILY_UNAVAILABLE:'+family)
    x=items[cursor[('RETENTION_TRAIN',family)]%len(items)];cursor[('RETENTION_TRAIN',family)]+=1;c=t=dict(x);c['slot']=t['slot']=slot;c['intervention_slot']=t['intervention_slot']=False
   replay_slot+=1
  if c['sequence_length']!=t['sequence_length'] or c['supervised_token_count']!=t['supervised_token_count']:raise RuntimeError('UNMATCHED_TOKEN_OR_SUPERVISION')
  control.append(c);treatment.append(t);common_total+=c['sequence_length'];slot+=1
 if totals(control)['transformer_tokens']!=totals(treatment)['transformer_tokens'] or totals(control)['supervised_tokens']!=totals(treatment)['supervised_tokens']:raise RuntimeError('ARM_TOTAL_MISMATCH')
 if len(control)%acc:raise RuntimeError('ACCUMULATION_ALIGNMENT_FAIL')
 if len(control)%alignment:raise RuntimeError('SCHEDULE_ALIGNMENT_FAIL')
 if replay_slot%5:raise RuntimeError('REPLAY_ALLOCATION_ALIGNMENT_FAIL')
 diff=[i for i,(c,t) in enumerate(zip(control,treatment)) if c['episode_id']!=t['episode_id']]
 expected=[i for i,x in enumerate(treatment) if x['intervention_slot']]
 if diff!=expected:raise RuntimeError('UNPLANNED_ARM_DIFFERENCE')
 return control,treatment,{'slots':len(control),'replay_slots':replay_slot,'intervention_slots':len(expected),'intervention_fraction':len(expected)/replay_slot,'optimizer_steps':len(control)//acc,'token_total':common_total}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 c=json.loads(args.config.read_text(encoding='utf8'))
 if c.get('scientific_training_authorized') is not False or c.get('model_loading_authorized') is not False:raise RuntimeError('PREPARATION_ONLY_REQUIRED')
 train=Path(c['train']['path']);
 if sha(train)!=c['train']['sha256']:raise RuntimeError('TRAIN_HASH_MISMATCH')
 pool=load(train);control,treatment,shape=build(c,pool)
 common={'protocol_id':c['protocol_id'],'config_sha256':sha(args.config),'train_sha256':sha(train),'checkpoint_manifest':c['checkpoint_manifest'],'shared_training':c['shared_training'],'shape':shape,'cpu_only':True,'model_loaded':False,'optimizer_constructed':False,'scientific_training_started':False}
 args.out.mkdir(parents=True,exist_ok=False)
 cp=args.out/'CONTROL_TRAINING_SCHEDULE.json';tp=args.out/'TREATMENT_TRAINING_SCHEDULE.json'
 atomic(cp,{**common,'arm':'CONTROL','episodes':control,'totals':totals(control)})
 atomic(tp,{**common,'arm':'TREATMENT','episodes':treatment,'totals':totals(treatment)})
 binding={'schema_version':1,**common,'control_schedule_sha256':sha(cp),'treatment_schedule_sha256':sha(tp),'allowed_difference':'designated intervention_slot replay positions only; each replacement preserves sequence_length and supervised_token_count','control_totals':totals(control),'treatment_totals':totals(treatment),'status':'FROZEN_CPU_ONLY_MATCHED_SCHEDULES'}
 atomic(args.out/'MATCHED_SCHEDULE_BINDING_V1.json',binding)
 print(json.dumps({'status':binding['status'],'optimizer_steps':shape['optimizer_steps'],'tokens':shape['token_total'],'intervention_slots':shape['intervention_slots']},sort_keys=True))
if __name__=='__main__':main()
