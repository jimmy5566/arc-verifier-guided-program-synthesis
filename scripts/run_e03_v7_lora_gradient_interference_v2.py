#!/usr/bin/env python3
"""Authorized E03 worker: transient vectors, durable Gram statistics only."""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, shutil, subprocess, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; IGNORE=-100; FAMILIES=('connected_components','inside_contains','width','difference','orientation','same_color','color_mapping','object_selection_then_action','novel_two_operation_order')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n');os.replace(t,p)
def resolve(s):p=Path(s);return p if p.is_absolute() else ROOT/p
def fail(s):raise RuntimeError(s)
def check_deadline(t):
 if time.monotonic()>=t:fail('E03_RUNTIME_CAP_EXCEEDED')
def verify_files(manifest):
 for root_key,files_key in (('base_path','base_files'),('adapter_path','adapter_files')):
  folder=Path(manifest[root_key])
  for item in manifest[files_key]:
   p=folder/item['name']
   if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:fail('E03_CHECKPOINT_FILE_HASH_MISMATCH:'+item['name'])
def preflight(config_path,binding_path,out):
 c=json.loads(config_path.read_text(encoding='utf-8'));b=json.loads(binding_path.read_text(encoding='utf-8'))
 if os.environ.get('E03_EXTERNAL_CAP_ENFORCED')!='1':fail('E03_EXTERNAL_CAP_LAUNCHER_REQUIRED')
 if c.get('execution_authorized') is not False:fail('E03_CONFIG_MUST_REMAIN_IMMUTABLE_UNAUTHORIZED')
 if b.get('execution_authorized') is not True or b.get('status')!='EXECUTION_AUTHORIZED_AFTER_DIRECTOR_REVIEW':fail('E03_IMMUTABLE_BINDING_AUTHORIZATION_REQUIRED')
 if str(out)!=b.get('output_root') or str(out/'TERMINAL_RECEIPT.json')!=b.get('terminal_receipt_path'):fail('E03_BOUND_OUTPUT_OR_RECEIPT_MISMATCH')
 if out.exists() and any(out.iterdir()):fail('E03_FRESH_OUTPUT_REQUIRED')
 if b.get('runtime_cap_seconds')!=1500 or b.get('jobs')!=1 or b.get('retry') is not False:fail('E03_BOUND_RUNTIME_INVALID')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=b.get('source_commit'):fail('E03_SOURCE_COMMIT_MISMATCH')
 if sha(config_path)!=b['bound_files'].get(str(config_path.relative_to(ROOT)).replace('\\','/')):fail('E03_CONFIG_HASH_MISMATCH')
 for rp,expected in b['bound_files'].items():
  if not (ROOT/rp).is_file() or sha(ROOT/rp)!=expected:fail('E03_BOUND_FILE_HASH_MISMATCH:'+rp)
 response=resolve(b['director_response_path'])
 if not response.is_file() or sha(response)!=b['director_response_sha256']:fail('E03_DIRECTOR_RESPONSE_HASH_MISMATCH')
 if json.loads(response.read_text(encoding='utf-8')).get('decision')!=b.get('required_director_decision','CONTINUE_CONTROLLER'):fail('E03_DIRECTOR_DECISION_NOT_AUTHORIZING')
 manifest=resolve(c['manifest_path']);checkpoint=resolve(c['checkpoint_manifest_path'])
 if sha(manifest)!=c['manifest_sha256'] or sha(checkpoint)!=c['checkpoint_manifest_sha256']:fail('E03_MANIFEST_OR_CHECKPOINT_MANIFEST_MISMATCH')
 m=json.loads(manifest.read_text(encoding='utf-8')); ck=json.loads(checkpoint.read_text(encoding='utf-8'))
 ids=[x['episode_id'] for f in m.get('families',[]) for mb in f.get('microbatches',[]) for x in mb.get('members',[])]
 if len(ids)!=288 or len(set(ids))!=288 or any(not x.startswith('TRAIN:') for x in ids):fail('E03_TRAIN_ONLY_MANIFEST_INVALID')
 verify_files(ck) # before model import
 return c,b,m,ck
def write_progress(out,done,phase):atomic(out/'PROGRESS.json',{'status':'PARTIAL_NO_UPDATE','phase':phase,'completed_basis_rows':done,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False})
def target_module(name):
 m=re.search(r'\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.',name)
 if not m:fail('E03_UNGROUPABLE_LORA_PARAMETER')
 return m.group(1)
def copy_gradient(dest,row,params,offsets,torch,scale=1.0,accumulate=False):
 for _,p in params:
  off,n=offsets[_]; g=p.grad
  if g is None or not bool(torch.isfinite(g).all()):fail('E03_NONFINITE_LORA_GRADIENT')
  value=g.detach().float().flatten().cpu().numpy()*scale
  if accumulate: dest[row,off:off+n]+=value
  else: dest[row,off:off+n]=value
def gram(mm,groups,offsets,chunk=262144):
 import numpy as np
 n=mm.shape[0]; out=np.zeros((n,n),dtype=np.float64); by={k:np.zeros((n,n),dtype=np.float64) for k in groups}
 for name,(off,length) in offsets.items():
  key=target_module(name)
  for start in range(off,off+length,chunk):
   block=np.asarray(mm[:,start:min(start+chunk,off+length)],dtype=np.float64); prod=block@block.T;out+=prod;by[key]+=prod
 return out,by
def main():
 a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True);a.add_argument('--binding',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True);x=a.parse_args();started=time.monotonic();receipt=x.output_root/'TERMINAL_RECEIPT.json';deadline=started+1500
 try:
  c,b,m,ck=preflight(x.config,x.binding,x.output_root);x.output_root.mkdir(parents=True,exist_ok=False);write_progress(x.output_root,0,'PREFLIGHT_COMPLETE')
  check_deadline(deadline)
  import numpy as np, torch
  import torch.nn.functional as F
  from peft import PeftModel
  from transformers import AutoModelForCausalLM
  torch.manual_seed(int(c['seed']));torch.cuda.manual_seed_all(int(c['seed']))
  model=AutoModelForCausalLM.from_pretrained(ck['base_path'],torch_dtype=torch.bfloat16,device_map='cuda:0');model=PeftModel.from_pretrained(model,ck['adapter_path'],is_trainable=True);model.eval()
  named=dict(model.named_parameters());allow=tuple(c['lora_parameter_name_allowlist']);actual=tuple(sorted(n for n in named if '.lora_A.' in n or '.lora_B.' in n))
  if actual!=tuple(sorted(allow)):fail('E03_LORA_ALLOWLIST_EXACT_SET_MISMATCH')
  for n,p in named.items():p.requires_grad_(n in allow)
  if any(p.requires_grad!=(n in allow) for n,p in named.items()):fail('E03_REQUIRES_GRAD_ASSIGNMENT_MISMATCH')
  params=[(n,named[n]) for n in allow];offsets={};size=0
  for n,p in params: offsets[n]=(size,p.numel());size+=p.numel()
  need=(36+9)*size*4;free=shutil.disk_usage(x.output_root.parent).free
  if free<need*2:fail('E03_INSUFFICIENT_BOUNDED_SCRATCH_SPACE')
  pmm_path=x.output_root/'PRIMARY_FLOAT32.memmap';bmm_path=x.output_root/'B1_FAMILY_SUM_FLOAT32.memmap';pmm=np.memmap(pmm_path,mode='w+',dtype='float32',shape=(36,size));bmm=np.memmap(bmm_path,mode='w+',dtype='float32',shape=(9,size));bmm[:]=0
  records=[];done=0
  for fam in m['families']:
   for mb in fam['microbatches']:
    check_deadline(deadline); members=mb['members'];width=max(len(z['input_ids']) for z in members);ids=torch.full((8,width),13,dtype=torch.long,device='cuda');lab=torch.full((8,width),IGNORE,dtype=torch.long,device='cuda');att=torch.zeros((8,width),dtype=torch.long,device='cuda')
    for i,z in enumerate(members):ids[i,:len(z['input_ids'])]=torch.tensor(z['input_ids'],device='cuda');lab[i,:len(z['labels'])]=torch.tensor(z['labels'],device='cuda');att[i,:len(z['input_ids'])]=1
    model.zero_grad(set_to_none=True);logits=model(input_ids=ids,attention_mask=att,position_ids=(att.cumsum(-1)-1).clamp_min(0),use_cache=False).logits[:,:-1].float();tar=lab[:,1:];count=int((tar!=IGNORE).sum());
    if count<1:fail('E03_ZERO_SUPERVISED_TOKENS')
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),tar.reshape(-1),ignore_index=IGNORE,reduction='sum');(loss/count).backward();copy_gradient(pmm,done,params,offsets,torch);pmm.flush();records.append({'canonical_family':fam['canonical_family'],'microbatch_index':mb['microbatch_index'],'basis_index':done,'supervised_token_count':count,'scalar_loss_sum':float(loss.item()),'episode_ids':[z['episode_id'] for z in members]});done+=1;model.zero_grad(set_to_none=True);write_progress(x.output_root,done,'PRIMARY_B8')
  bmeta=[]
  for fi,fam in enumerate(m['families']):
   total=0; episodes=[];scalar=0.
   for z in fam['microbatches'][0]['members']:
    check_deadline(deadline);ids=torch.tensor([z['input_ids']],dtype=torch.long,device='cuda');lab=torch.tensor([z['labels']],dtype=torch.long,device='cuda');att=torch.ones_like(ids);model.zero_grad(set_to_none=True);logits=model(input_ids=ids,attention_mask=att,position_ids=(att.cumsum(-1)-1).clamp_min(0),use_cache=False).logits[:,:-1].float();tar=lab[:,1:];count=int((tar!=IGNORE).sum());loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),tar.reshape(-1),ignore_index=IGNORE,reduction='sum');(loss/count).backward();copy_gradient(bmm,fi,params,offsets,torch,count,True);total+=count;scalar+=float(loss.item());episodes.append({'episode_id':z['episode_id'],'supervised_token_count':count,'scalar_loss_sum':float(loss.item())});model.zero_grad(set_to_none=True)
   if total<1:fail('E03_B1_ZERO_SUPERVISED_TOKENS')
   bmm[fi,:]/=total;bmm.flush();bmeta.append({'canonical_family':fam['canonical_family'],'microbatch_index':0,'basis_index':fi,'supervised_token_count':total,'scalar_loss_sum':scalar,'episodes':episodes});write_progress(x.output_root,done+fi+1,'B1_RECONSTRUCTION')
  check_deadline(deadline);pg,pm=gram(pmm,{target_module(n) for n,_ in params},offsets);bg,bgm=gram(bmm,{target_module(n) for n,_ in params},offsets);pmm.flush();bmm.flush()
  raw={'status':'COMPLETE_NO_UPDATE_E03_GRAM_SUFFICIENT_STATISTICS','manifest_rows':288,'batch1_rows':72,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False,'primary_records':records,'batch1_family_records':bmeta,'primary_global_gram':pg.tolist(),'batch1_global_gram':bg.tolist(),'primary_module_grams':{k:v.tolist() for k,v in pm.items()},'batch1_module_grams':{k:v.tolist() for k,v in bgm.items()},'observed_lora_allowlist_sha256':hashlib.sha256(('\n'.join(actual)).encode()).hexdigest(),'bounded_scratch_bytes':need,'complete_vectors_persisted':False}
  raw_path=x.output_root/'RAW_GRAM_SUFFICIENT_STATISTICS.json';atomic(raw_path,raw)
  del pmm,bmm
  pmm_path.unlink();bmm_path.unlink()
  atomic(receipt,{'protocol_id':c['protocol_id'],'status':'COMPLETE_NO_UPDATE','elapsed_seconds':time.monotonic()-started,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False,'raw_evidence_sha256':sha(raw_path),'complete_vectors_persisted':False})
 except Exception as e:
  x.output_root.mkdir(parents=True,exist_ok=True);atomic(receipt,{'protocol_id':'E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V1','status':'FAILED_OR_PARTIAL_NO_UPDATE','error_class':str(e),'elapsed_seconds':time.monotonic()-started,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False,'progress_exists':(x.output_root/'PROGRESS.json').exists(),'traceback':traceback.format_exc(limit=4)});raise
if __name__=='__main__':main()
