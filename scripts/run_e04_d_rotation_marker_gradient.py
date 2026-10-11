#!/usr/bin/env python3
"""E04-D B1 LoRA-gradient worker. Runtime imports occur only after preflight."""
from __future__ import annotations
import argparse,hashlib,json,math,os,re,subprocess,time,traceback,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; IGNORE=-100
if str(ROOT / 'src') not in sys.path: sys.path.insert(0, str(ROOT / 'src'))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n');os.replace(t,p)
def resolve(x):
 p=Path(x);return (p if p.is_absolute() else ROOT/p).resolve()
def load_config(path):
 c=json.loads(path.read_text());co=resolve(c['cohort_path']);pr=resolve(c['prelaunch_protocol_path']);cp=resolve(c['checkpoint_manifest_path'])
 if sha(co)!=c['cohort_sha256'] or sha(pr)!=c['prelaunch_protocol_sha256'] or sha(cp)!=c['checkpoint_manifest_sha256']:raise RuntimeError('E04D_CONFIG_HASH_MISMATCH')
 rows=[json.loads(x) for x in co.read_text().splitlines() if x]
 if len(rows)!=48 or len({x['pair_id'] for x in rows})!=48 or any(set(x['tasks'])!=set(c['conditions']) for x in rows):raise RuntimeError('E04D_COHORT_INVALID')
 if c['physical_batch_size']!=1 or c['precision']!='BF16' or any(c[k] for k in ('optimizer_construction','optimizer_steps','parameter_updates','training','generation_calls','final_audit_opened')):raise RuntimeError('E04D_ZERO_UPDATE_CONTRACT')
 return c,rows,json.loads(cp.read_text())
def static(config):
 c,rows,_=load_config(config)
 from training_data.pipeline import task_to_sample
 counts=[]
 for row in rows:
  for cond,task in row['tasks'].items():
   sample=task_to_sample({'source_id':'e04d-'+row['pair_id']+'-'+cond,**task});n=sum(v!=IGNORE for v in sample['labels'])
   if n<1 or n!=sample['assistant_token_count'] or len(sample['input_ids'])!=len(sample['labels']):raise RuntimeError('E04D_SERIALIZER_MASK_INVALID')
   counts.append(n)
 return {'status':'PASS_CPU_STATIC_NO_MODEL','rows':48,'condition_rows':144,'supervised_tokens_min':min(counts),'supervised_tokens_max':max(counts),'model_loaded':False,'gpu_used':False,'optimizer_steps':0}
def validate_runtime_identity(binding,root=ROOT):
 """Validate the immutable approved source and a clean runtime checkout.

 The approved snapshot is a lower bound: an exact checkout or a clean
 descendant may run only when every scientific executable is separately
 hash-bound in the launch binding.  This is intentionally identical to the
 CPU preflight contract and does not depend on a moving branch name.
 """
 approved=str(binding.get('approved_source_commit','')).lower()
 if not re.fullmatch(r'[0-9a-f]{40}',approved):raise RuntimeError('E04D_APPROVED_SOURCE_INVALID')
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip().lower()
 if subprocess.call(['git','merge-base','--is-ancestor',approved,head],cwd=root,stderr=subprocess.DEVNULL)!=0:raise RuntimeError('E04D_APPROVED_SOURCE_NOT_ANCESTOR')
 if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip():raise RuntimeError('E04D_TRACKED_SOURCE_DIRTY')
 return head
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--binding',type=Path);ap.add_argument('--output-root',type=Path);ap.add_argument('--static',action='store_true');ap.add_argument('--identity-check',action='store_true');a=ap.parse_args(); config=resolve(str(a.config))
 if a.static: print(json.dumps(static(config),sort_keys=True));return
 if a.identity_check:
  if not a.binding:raise SystemExit('E04D_RUNTIME_ARGS')
  c,_,_=load_config(config);b=json.loads(resolve(str(a.binding)).read_text());head=validate_runtime_identity(b)
  print(json.dumps({'status':'PASS_CPU_IDENTITY_NO_MODEL','runtime_head':head,'approved_source_commit':b['approved_source_commit'],'model_loaded':False,'gpu_used':False,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False},sort_keys=True));return
 if not a.binding or not a.output_root:raise SystemExit('E04D_RUNTIME_ARGS')
 start=time.monotonic();out=resolve(str(a.output_root));receipt=out/'TERMINAL_RECEIPT.json'
 try:
  c,rows,checkpoint=load_config(config);b=json.loads(resolve(str(a.binding)).read_text())
  if os.environ.get('E04D_EXTERNAL_CAP_ENFORCED')!='1' or b.get('execution_authorized') is not True or b.get('runtime_cap_seconds')!=1800 or b.get('config_sha256')!=sha(config) or b.get('output_root')!=str(out):raise RuntimeError('E04D_BINDING_INVALID')
  if out.exists():raise RuntimeError('E04D_FRESH_OUTPUT_REQUIRED')
  head=validate_runtime_identity(b)
  for group,pathkey in (('base_files','base_path'),('adapter_files','adapter_path')):
   for f in checkpoint[group]:
    p=Path(checkpoint[pathkey])/f['name']
    if not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:raise RuntimeError('E04D_CHECKPOINT_IDENTITY')
  out.mkdir(parents=True);atom(out/'PROGRESS.json',{'status':'PARTIAL_NO_UPDATE','completed_pairs':0,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False})
  import torch,torch.nn.functional as F
  from peft import PeftModel
  from transformers import AutoModelForCausalLM
  from training_data.pipeline import task_to_sample
  model=AutoModelForCausalLM.from_pretrained(checkpoint['base_path'],torch_dtype=torch.bfloat16,device_map='cuda:0');model=PeftModel.from_pretrained(model,checkpoint['adapter_path'],is_trainable=True);model.eval();model.config.use_cache=False
  if hasattr(model,'enable_input_require_grads'):model.enable_input_require_grads()
  named=dict(model.named_parameters());allow=tuple(c['lora_parameter_name_allowlist'])
  for n,p in named.items():p.requires_grad_(n in allow)
  if tuple(sorted(n for n,p in named.items() if p.requires_grad))!=tuple(allow):raise RuntimeError('E04D_ALLOWLIST_RUNTIME')
  params=[(n,named[n]) for n in allow]
  def grad(task,source):
   model.zero_grad(set_to_none=True);sample=task_to_sample({'source_id':source,**task});ids=torch.tensor([sample['input_ids']],device='cuda');labels=torch.tensor([sample['labels']],device='cuda');att=torch.ones_like(ids);pos=(att.cumsum(-1)-1).clamp_min(0);logits=model(input_ids=ids,attention_mask=att,position_ids=pos,use_cache=False).logits;sel=labels[:,1:].ne(IGNORE);n=int(sel.sum());loss=F.cross_entropy(logits[:,:-1][sel].float(),labels[:,1:][sel],reduction='sum')/n;loss.backward();vec=torch.cat([p.grad.detach().float().flatten() for _,p in params]);
   if not torch.isfinite(vec).all() or float(vec.norm())==0:raise RuntimeError('E04D_NONFINITE_OR_ZERO_GRADIENT')
   mods={m:0. for m in c['target_modules']}
   for name,p in params:
    m=next((x for x in c['target_modules'] if '.'+x+'.' in name),None)
    if m is None:raise RuntimeError('E04D_MODULE_GROUP')
    mods[m]+=float((p.grad.detach().float()**2).sum())
   return vec,{'loss':float(loss.detach()),'supervised_token_count':n,'norm':float(vec.norm()),'module_squared_norms':mods}
  pairs=[];repeat_cos={}; repeat_id=c['repeat_pair_id']
  for i,row in enumerate(rows):
   vectors={};stats={}
   for cond in c['conditions']:vectors[cond],stats[cond]=grad(row['tasks'][cond],row['pair_id']+'-'+cond)
   cos=lambda x,y:float(torch.nn.functional.cosine_similarity(vectors[x],vectors[y],dim=0))
   pairs.append({'pair_id':row['pair_id'],'canonical_base_id':row['canonical_base_id'],'turn':row['turn'],'rotation_marker_cosine':cos(c['conditions'][0],c['conditions'][1]),'rotation_no_transform_cosine':cos(c['conditions'][0],c['conditions'][2]),'marker_no_transform_cosine':cos(c['conditions'][1],c['conditions'][2]),'condition_statistics':stats})
   if row['pair_id']==repeat_id:
    for cond in c['conditions']:
     again,_=grad(row['tasks'][cond],row['pair_id']+'-'+cond+'-repeat');repeat_cos[cond]=float(torch.nn.functional.cosine_similarity(vectors[cond],again,dim=0))
   atom(out/'PROGRESS.json',{'status':'PARTIAL_NO_UPDATE','completed_pairs':i+1,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False})
  raw={'status':'COMPLETE_NO_UPDATE','protocol_id':c['protocol_id'],'config_sha256':sha(config),'pairs':pairs,'repeat_cosines':repeat_cos,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False,'runtime_head':head};atom(out/'RAW_GRADIENT_STATISTICS.json',raw);atom(receipt,{'status':'COMPLETE_NO_UPDATE','protocol_id':c['protocol_id'],'raw_sha256':sha(out/'RAW_GRADIENT_STATISTICS.json'),'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False,'elapsed_seconds':time.monotonic()-start})
 except Exception as e:
  out.mkdir(parents=True,exist_ok=True);atom(receipt,{'status':'FAILED_OR_PARTIAL_NO_UPDATE','error_class':str(e),'traceback':traceback.format_exc(),'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False,'elapsed_seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
