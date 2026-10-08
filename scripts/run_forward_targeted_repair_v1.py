#!/usr/bin/env python3
"""Forward targeted repair from CAPABILITY_REPAIR_BASELINE_V1.

This worker reads only the frozen TRAIN surface, continues the verified LoRA
adapter, and writes an append-only cumulative GPU-time ledger in its fresh
remote run directory.  DEV, retention sentinel, and FINAL_AUDIT are never
opened here.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT/'src'))
from training_data.pipeline import IGNORE_INDEX, task_to_sample

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n',encoding='utf8');t.replace(p)
def samples(path):
 rows=[json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines() if x.strip()]
 ok={'ATOMIC_REPAIR','COMPOSITION_REPAIR','RETENTION_TRAIN'}
 if not rows or any(r.get('split')!='TRAIN' or r.get('role') not in ok for r in rows):raise RuntimeError('TRAIN_SURFACE_INVALID')
 out=[]
 for r in rows:
  x=task_to_sample({'source_id':r['episode_id'],**r['task']})
  if len(x['input_ids'])!=len(x['labels']) or not any(v!=IGNORE_INDEX for v in x['labels']):raise RuntimeError('TRAIN_SERIALIZATION_INVALID')
  out.append({'ids':x['input_ids'],'labels':x['labels'],'tokens':len(x['input_ids']),'supervised':x['assistant_token_count'],'role':r['role'],'family':r['family'],'episode_id':r['episode_id']})
 return out
def schedule(pool,c):
 by={role:{} for role in c['role_weights']}
 for x in pool:by[x['role']].setdefault(x['family'],[]).append(x)
 for families in by.values():
  for v in families.values():v.sort(key=lambda x:x['episode_id'])
 if any(not v for v in by.values()):raise RuntimeError('CURRICULUM_POOL_EMPTY')
 rng=random.Random(int(c['seed'])); cursor={(r,f):0 for r,fs in by.items() for f in fs}; out=[];total=0;i=0
 role_units=20
 roles=[]
 for r,w in c['role_weights'].items(): roles += [r]*int(round(float(w)*role_units))
 if (len(roles)!=role_units or set(roles)!=set(c['role_weights']) or abs(sum(float(v) for v in c['role_weights'].values())-1.0)>1e-9):raise RuntimeError('ROLE_WEIGHTS_INVALID')
 while total < int(c['nominal_tokens']) or len(out)%int(c['gradient_accumulation']):
  role=roles[i%len(roles)]; weights=c['family_weights'][role]; fams=list(weights); weights_v=[float(weights[f]) for f in fams]; family=rng.choices(fams,weights_v,k=1)[0]; choices=by[role].get(family)
  if not choices: raise RuntimeError('CURRICULUM_FAMILY_UNAVAILABLE:'+family)
  key=(role,family); x=choices[cursor[key]%len(choices)];cursor[key]+=1;out.append(x);total+=x['tokens'];i+=1
 return out,total
def main():
 a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True);a.add_argument('--run-root',type=Path,required=True);z=a.parse_args();c=json.loads(z.config.read_text(encoding='utf8'));run=z.run_root
 if run.exists():raise RuntimeError('FRESH_OUTPUT_REQUIRED')
 if c['final_audit_opened'] is not False:raise RuntimeError('FINAL_AUDIT_FORBIDDEN')
 if sha(c['train_path'])!=c['train_sha256']:raise RuntimeError('TRAIN_IDENTITY_FAIL')
 if sha(c['checkpoint_manifest_path'])!=c['checkpoint_manifest_sha256']:raise RuntimeError('CHECKPOINT_MANIFEST_FAIL')
 manifest=json.loads(Path(c['checkpoint_manifest_path']).read_text(encoding='utf8'))
 run.mkdir(parents=True);runtime=run/'runtime';ckpts=run/'checkpoints';runtime.mkdir();ckpts.mkdir()
 ledger=run/'ARC2_CUMULATIVE_NEW_GPU_TRAINING_LEDGER.jsonl';hist=float(c.get('prior_cumulative_gpu_seconds',c['historical_gpu_seconds']));cap=float(c['cumulative_cap_seconds']);reserve=float(c['reservation_seconds'])
 if hist+reserve>cap:raise RuntimeError('BUDGET_RESERVATION_REJECTED')
 ledger.write_text(json.dumps({'schema_version':1,'record_type':'PRIOR_CUMULATIVE_GPU_TRAINING','gpu_optimizer_seconds':hist,'round_id':'PRIOR_CUMULATIVE_ROUNDS','receipt_sha256':c.get('prior_gpu_ledger_sha256',c['historical_receipt_sha256'])},sort_keys=True)+'\n',encoding='utf8')
 terminal={'schema_version':1,'protocol_id':c['protocol_id'],'round_id':c['round_id'],'scientific_training_started':False,'optimizer_steps':0,'processed_tokens':0,'final_audit_accessed':False}
 model=None;start_ns=None
 try:
  pool=samples(c['train_path']); work,tokens=schedule(pool,c); atomic(runtime/'TRAINING_SCHEDULE.json',{'episodes':[{'episode_id':x['episode_id'],'role':x['role'],'family':x['family'],'tokens':x['tokens']} for x in work],'actual_tokens':tokens,'role_weights':c['role_weights'],'family_weights':c['family_weights']})
  import torch, bitsandbytes as bnb
  from transformers import AutoModelForCausalLM
  from peft import PeftModel
  if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA_BF16_UNAVAILABLE')
  os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false'
  model=AutoModelForCausalLM.from_pretrained(manifest['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0')
  model=PeftModel.from_pretrained(model,manifest['adapter_path'],is_trainable=True);params=[p for p in model.parameters() if p.requires_grad]
  if not params or any(p.requires_grad for n,p in model.named_parameters() if 'lora_' not in n):raise RuntimeError('LORA_PARTITION_FAIL')
  # Re-entrant gradient checkpointing needs at least one input activation to
  # require gradients.  The base is frozen, so explicitly enable this bridge
  # before the first forward pass; otherwise every LoRA loss is detached.
  model.enable_input_require_grads()
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True});model.train();opt=bnb.optim.PagedAdamW8bit(params,lr=float(c['learning_rate']))
  start_ns=time.monotonic_ns();deadline=time.monotonic()+reserve;acc=int(c['gradient_accumulation']);steps=0;done=0;supervised=0;capped=False
  for offset in range(0,len(work),acc):
   if time.monotonic()>=deadline:capped=True;break
   group=work[offset:offset+acc];opt.zero_grad(set_to_none=True);den=sum(x['supervised'] for x in group)
   for x in group:
    ids=torch.tensor([x['ids']],device='cuda:0',dtype=torch.long);labels=torch.tensor([x['labels']],device='cuda:0',dtype=torch.long);loss=model(input_ids=ids,labels=labels,use_cache=False).loss
    if not bool(torch.isfinite(loss)):raise RuntimeError('NONFINITE_LOSS')
    (loss*(x['supervised']/den)).backward();done+=x['tokens'];supervised+=x['supervised']
   if time.monotonic()>=deadline:capped=True;break
   for g in opt.param_groups:g['lr']=float(c['learning_rate'])*min(1.0,(steps+1)/3)
   opt.step();steps+=1
   if done>=250000 and not (ckpts/'tokens_250000').exists():model.save_pretrained(ckpts/'tokens_250000')
   atomic(runtime/'TRAINING_PROGRESS.json',{'status':'RUNNING','optimizer_steps':steps,'processed_tokens':done,'supervised_tokens':supervised})
  model.save_pretrained(ckpts/('cap_finalize' if capped else 'final'))
  terminal.update({'status':'CAP_REACHED_CHECKPOINTED' if capped else 'COMPLETED','scientific_training_started':steps>0,'optimizer_steps':steps,'processed_tokens':done,'supervised_tokens':supervised,'no_optimizer_after_cap':capped,'starting_adapter_path':manifest['adapter_path']})
 except Exception as e:
  terminal.update({'status':'PRE_OPTIMIZER_FAILURE' if start_ns is None else 'EARLY_RUNTIME_FAILURE','error':f'{type(e).__name__}:{e}','traceback':traceback.format_exc(limit=4)})
 finally:
  stop_ns=time.monotonic_ns()
  # A failed forward/backward before the first optimizer update is an
  # infrastructure failure, not scientific training.  Keep elapsed wrapper
  # time out of the cumulative scientific GPU ledger.
  seconds=0.0 if start_ns is None or int(terminal.get('optimizer_steps',0))==0 else (stop_ns-start_ns)/1e9
  terminal['scientific_gpu_training_seconds']=seconds;atomic(runtime/'TERMINAL_RECEIPT.json',terminal)
  with ledger.open('a',encoding='utf8') as f:f.write(json.dumps({'schema_version':1,'record_type':'GPU_OPTIMIZER_INTERVAL','round_id':c['round_id'],'gpu_optimizer_seconds':seconds,'receipt_sha256':sha(runtime/'TERMINAL_RECEIPT.json')},sort_keys=True)+'\n')
  atomic(runtime/'GPU_LEDGER_SNAPSHOT.json',{'cap_seconds':cap,'prior_cumulative_gpu_seconds':hist,'this_round_gpu_seconds':seconds,'cumulative_gpu_seconds':hist+seconds,'remaining_seconds':cap-hist-seconds})
  if model is not None:
   try:
    import torch;del model;torch.cuda.empty_cache()
   except Exception:pass
 return 0 if terminal['status'] in {'COMPLETED','CAP_REACHED_CHECKPOINTED'} else 1
if __name__=='__main__':raise SystemExit(main())
