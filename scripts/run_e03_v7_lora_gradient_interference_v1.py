#!/usr/bin/env python3
"""Bounded, authorized-only E03 LoRA gradient collection worker.

The worker is not invoked by the CPU freeze.  An immutable future binding must
set execution_authorized=true and bind this exact source SHA before model import.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
IGNORE=-100

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n');os.replace(t,p)
def resolve(x):
 p=Path(x);return p if p.is_absolute() else ROOT/p
def die(x):raise RuntimeError(x)
def preflight(config, config_path, output):
 required=('protocol_id','execution_authorized','manifest_path','manifest_sha256','checkpoint_manifest_path','checkpoint_manifest_sha256','runtime','parameters','numerics','seed')
 if any(k not in config for k in required):die('E03_CONFIG_FIELD_MISSING')
 if config['execution_authorized'] is not True:die('E03_DIRECTOR_AUTHORIZATION_REQUIRED')
 if config['protocol_id']!='E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V1':die('E03_PROTOCOL_MISMATCH')
 if output.exists() and any(output.iterdir()):die('E03_FRESH_OUTPUT_REQUIRED')
 manifest=resolve(config['manifest_path']); checkpoint=resolve(config['checkpoint_manifest_path'])
 if sha(manifest)!=config['manifest_sha256'] or sha(checkpoint)!=config['checkpoint_manifest_sha256']:die('E03_IDENTITY_HASH_MISMATCH')
 data=json.loads(manifest.read_text(encoding='utf-8'))
 if data.get('rows')!=288 or len(data.get('families',[]))!=9:die('E03_MANIFEST_SHAPE_INVALID')
 ids=[m['episode_id'] for f in data['families'] for b in f['microbatches'] for m in b['members']]
 if len(ids)!=288 or len(set(ids))!=288 or any(not x.startswith('TRAIN:') for x in ids):die('E03_TRAIN_ONLY_INVALID')
 p=config['parameters']
 if p.get('trainable')!='LoRA adapters only' or p.get('optimizer_constructed') is not False or p.get('optimizer_steps')!=0 or p.get('parameter_updates')!=0 or p.get('base_weight_gradients_persisted') is not False:die('E03_NO_UPDATE_CONTRACT_INVALID')
 if config['runtime'].get('hard_cap_seconds')!=1500:die('E03_RUNTIME_CAP_INVALID')
 return data,json.loads(checkpoint.read_text(encoding='utf-8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);a=ap.parse_args();started=time.monotonic();receipt=a.output_root/'TERMINAL_RECEIPT.json'
 try:
  config=json.loads(a.config.read_text(encoding='utf-8'));manifest,checkpoint=preflight(config,a.config,a.output_root)
  # No model-related import is reachable until all identity/isolation/freshness
  # checks above pass.
  import torch
  import torch.nn.functional as F
  from peft import PeftModel
  from transformers import AutoModelForCausalLM
  torch.manual_seed(int(config['seed']));torch.cuda.manual_seed_all(int(config['seed']))
  model=AutoModelForCausalLM.from_pretrained(checkpoint['base_path'],torch_dtype=torch.bfloat16,device_map='cuda:0')
  model=PeftModel.from_pretrained(model,checkpoint['adapter_path'],is_trainable=True);model.eval()
  named=list(model.named_parameters()); allow=tuple(config['lora_parameter_name_allowlist'])
  for name,p in named:p.requires_grad_(name in allow)
  lora=[(n,p) for n,p in named if n in allow]
  if not lora or any(p.requires_grad for n,p in named if n not in allow):die('E03_LORA_ALLOWLIST_REQUIRES_GRAD_FAIL')
  observed=[]
  for family in manifest['families']:
   for batch in family['microbatches']:
    if time.monotonic()-started>1500:die('E03_RUNTIME_CAP_EXCEEDED')
    members=batch['members'];width=max(len(m['input_ids']) for m in members)
    ids=torch.full((8,width),13,dtype=torch.long,device='cuda');labels=torch.full((8,width),IGNORE,dtype=torch.long,device='cuda');att=torch.zeros((8,width),dtype=torch.long,device='cuda')
    for i,m in enumerate(members):
     ids[i,:len(m['input_ids'])]=torch.tensor(m['input_ids'],device='cuda');labels[i,:len(m['labels'])]=torch.tensor(m['labels'],device='cuda');att[i,:len(m['input_ids'])]=1
    positions=(att.cumsum(-1)-1).clamp_min(0);model.zero_grad(set_to_none=True)
    logits=model(input_ids=ids,attention_mask=att,position_ids=positions,use_cache=False).logits[:,:-1].float();targets=labels[:,1:]
    count=int((targets!=IGNORE).sum().item());
    if count<1:die('E03_ZERO_SUPERVISED_TOKENS')
    total=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=IGNORE,reduction='sum');(total/count).backward()
    pieces=[];module_sq={}
    for name,p in lora:
     if p.grad is None or not torch.isfinite(p.grad).all():die('E03_NONFINITE_LORA_GRADIENT')
     g=p.grad.detach().float();pieces.append(g.flatten().cpu());module=name.rsplit('.',1)[0];module_sq[module]=module_sq.get(module,0.0)+float(g.square().sum().item())
    vector=torch.cat(pieces);n=float(torch.linalg.vector_norm(vector).item())
    if not math.isfinite(n) or n==0:die('E03_ZERO_NORM_GRADIENT_INVALID_NOT_INTERPRETABLE')
    # Full LoRA vectors are retained only in process memory for the exact Gram
    # reducer; no base vector/gradient is materialized or persisted.
    observed.append({'canonical_family':family['canonical_family'],'microbatch_index':batch['microbatch_index'],'supervised_token_count':count,'scalar_loss_sum':float(total.item()),'gradient':vector.tolist(),'adapter_module_squared_norms':module_sq})
    model.zero_grad(set_to_none=True)
  b1=[]
  for family in manifest['families']:
   for member in family['microbatches'][0]['members']:
    if time.monotonic()-started>1500:die('E03_RUNTIME_CAP_EXCEEDED')
    ids=torch.tensor([member['input_ids']],dtype=torch.long,device='cuda');labels=torch.tensor([member['labels']],dtype=torch.long,device='cuda')
    att=torch.ones_like(ids);positions=(att.cumsum(-1)-1).clamp_min(0);model.zero_grad(set_to_none=True)
    logits=model(input_ids=ids,attention_mask=att,position_ids=positions,use_cache=False).logits[:,:-1].float();targets=labels[:,1:]
    count=int((targets!=IGNORE).sum().item())
    if count<1:die('E03_B1_ZERO_SUPERVISED_TOKENS')
    total=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=IGNORE,reduction='sum');(total/count).backward()
    pieces=[]
    for _,p in lora:
     if p.grad is None or not torch.isfinite(p.grad).all():die('E03_B1_NONFINITE_LORA_GRADIENT')
     pieces.append(p.grad.detach().float().flatten().cpu())
    vector=torch.cat(pieces);n=float(torch.linalg.vector_norm(vector).item())
    if not math.isfinite(n) or n==0:die('E03_B1_ZERO_NORM_GRADIENT_INVALID_NOT_INTERPRETABLE')
    b1.append({'canonical_family':family['canonical_family'],'episode_id':member['episode_id'],'supervised_token_count':count,'scalar_loss_sum':float(total.item()),'gradient':vector.tolist()})
    model.zero_grad(set_to_none=True)
  raw={'status':'COMPLETE_NO_UPDATE_E03_SUFFICIENT_STATISTICS','manifest_rows':288,'batch1_rows':72,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False,'primary_b8_microbatch_gradients':observed,'batch1_per_example_gradients':b1}
  atomic(a.output_root/'RAW_LORA_GRADIENTS.json',raw)
  atomic(receipt,{'protocol_id':'E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V1','status':'COMPLETE_NO_UPDATE','elapsed_seconds':time.monotonic()-started,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False,'raw_evidence_sha256':sha(a.output_root/'RAW_LORA_GRADIENTS.json')})
 except Exception as e:
  atomic(receipt,{'protocol_id':'E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V1','status':'FAILED_OR_PARTIAL_NO_UPDATE','error_class':str(e),'elapsed_seconds':time.monotonic()-started,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False,'traceback':traceback.format_exc(limit=4)})
  raise
if __name__=='__main__':main()
