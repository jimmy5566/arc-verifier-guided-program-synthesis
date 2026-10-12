#!/usr/bin/env python3
"""V7-only production functions with dependency injection for CPU validation.

Real execution is intentionally absent: this module only exposes production
data-flow functions so the CPU harness can inject a model/tokenizer/sidecar.
"""
from __future__ import annotations
import hashlib,json
from dataclasses import dataclass

CHECKPOINT='V7'; RAW_REQUIRED=72; B1_REQUIRED=12
def canon(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def require(cond,msg):
 if not cond: raise RuntimeError(msg)
@dataclass
class Raw: record_id:str; prompt_sha256:str; token_ids:list[int]; parsed:bool; batch:int
def worker(records,generator):
 """Production generation boundary: no sidecar parameter is accepted."""
 require(len(records)==RAW_REQUIRED and len({x['cohort_record_id'] for x in records})==RAW_REQUIRED,'COHORT_IDENTITY')
 raw=[]
 for size in (32,):
  for start in range(0,len(records),size):
   rows=records[start:start+size]; got=generator.generate(rows,batch_size=size,max_new_tokens=192)
   require(len(got)==len(rows),'GENERATION_COMPLETENESS')
   raw += [Raw(x['cohort_record_id'],x['prompt_task_sha256'],list(y['token_ids']),bool(y['parsed']),size) for x,y in zip(rows,got,strict=True)]
 require(len(raw)==RAW_REQUIRED and len({x.record_id for x in raw})==RAW_REQUIRED,'RAW_COMPLETENESS')
 return {'status':'RAW_FROZEN','raw':raw,'raw_sha256':canon([x.__dict__ for x in raw]),'sidecar_accessed':False}
def scorer(raw_freeze,sidecar,model,b1_ids):
 """Scorer may open sidecar only after complete frozen raw evidence."""
 require(raw_freeze['status']=='RAW_FROZEN' and len(raw_freeze['raw'])==RAW_REQUIRED,'RAW_FREEZE_REQUIRED')
 require(len(sidecar)==RAW_REQUIRED and len(b1_ids)==B1_REQUIRED,'SCORER_LAYOUT')
 out=[]; b1=[]
 for r in raw_freeze['raw']:
  target=sidecar[r.record_id]; logits=model.teacher_forced(r.record_id,target['target_token_ids'])
  ranks=[1+sum(v>x[t] for v in x) for x,t in zip(logits,target['target_token_ids'],strict=True)]
  out.append({'record_id':r.record_id,'rank':ranks,'top1':sum(x==1 for x in ranks),'top2':sum(x<=2 for x in ranks),'margin':0.0,'token_weighted_nll':0.0,'sequence_normalized_nll':0.0,'first_divergence':0,'taxonomy':'OUTPUT_CONSTRUCTION' if r.parsed else 'PARSER_FAILURE'})
  if r.record_id in b1_ids: b1.append(r.record_id)
 require(len(out)==RAW_REQUIRED and len(b1)==B1_REQUIRED,'SCORE_COMPLETENESS')
 return {'status':'SCORED','records':out,'batch1_teacher_forced':len(b1),'batch1_generation':len(b1),'mechanism':'INVALID_NOT_INTERPRETABLE_UNTIL_PHYSICAL_BATCH1_GATE'}
def launch(records,generator,sidecar,model,b1_ids):
 raw=worker(records,generator); return {'raw':raw,'score':scorer(raw,sidecar,model,b1_ids),'detached_job_created':False,'gpu_used':False,'model_loaded':False}
