#!/usr/bin/env python3
"""Bounded, no-update ARC2 NLL measurement foundations worker."""
from __future__ import annotations
import argparse, hashlib, json, os, time, traceback
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'src'))
WEAK=('connected components','inside/contains','difference','width','orientation')
IGNORE=-100

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n',encoding='utf-8'); os.replace(tmp,path)
def rp(value):
 p=Path(value); return p if p.is_absolute() else ROOT/p
def loadj(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def check(condition,tag):
 if not condition: raise RuntimeError(tag)
def explicit_position(mask,torch):
 p=mask.long().cumsum(-1)-1
 return p.masked_fill(mask==0,0)
def validate(cfg,auth):
 reviewed=auth.get('reviewed_artifacts',{})
 package_path=auth.get('package_path') or reviewed.get('execution_package_path')
 package_sha=auth.get('package_sha256') or reviewed.get('execution_package_sha256')
 check(package_path is not None and package_sha is not None,'AUTHORIZATION_PACKAGE_BINDING_MISSING')
 check(sha(rp(package_path))==package_sha,'PACKAGE_SHA_MISMATCH')
 check(sha(rp(cfg['inputs']['cohort_path']))==cfg['inputs']['cohort_sha256'],'COHORT_SHA_MISMATCH')
 cohort=loadj(rp(cfg['inputs']['cohort_path'])); check(len(cohort['rows'])==30,'COHORT_ROWS_MISMATCH')
 check([m['mode_id'] for m in cfg['modes']]==['REFERENCE_B1_IMPLICIT_POSITION','REFERENCE_B1_EXPLICIT_POSITION_CONTROL','BATCHED_IMPLICIT_POSITION','BATCHED_EXPLICIT_POSITION_CONTROL'],'MODE_CONTRACT_MISMATCH')
 check(cfg['numeric_acceptance']['admissible_batched_effective_sizes']==[32,16,8,4,2],'BATCH_LADDER_MISMATCH')
 check(cfg['execution_order']['repeat_count_per_checkpoint_row_mode']==3,'REPEAT_CONTRACT_MISMATCH')
 check(cfg['numeric_acceptance']['absolute_token_micro_nll_stability_threshold']==0.0005,'THRESHOLD_MISMATCH')
 return cohort
def rows_from_train(cohort,train):
 from training_data.pipeline import task_to_sample, IGNORE_INDEX
 data={x['episode_id']:x for x in (json.loads(z) for z in Path(train).read_text(encoding='utf-8').splitlines() if z.strip())}
 out=[]
 for c in cohort['rows']:
  row=data.get(c['episode_id']); check(row is not None and row.get('split')=='TRAIN','TRAIN_ROW_MISSING:'+c['episode_id'])
  x=task_to_sample({'source_id':row['episode_id'],**row['task']}); ids=x['input_ids']; labels=x['labels']
  h=hashlib.sha256(json.dumps({'input_ids':ids,'labels':labels},sort_keys=True,separators=(',',':')).encode()).hexdigest()
  check(h==c['token_label_sha256'],'TOKEN_LABEL_MISMATCH:'+c['episode_id'])
  check(sum(v!=IGNORE_INDEX for v in labels)==c['supervised_token_count'],'SUPERVISION_MISMATCH:'+c['episode_id'])
  out.append({'episode_id':c['episode_id'],'family':c['family'],'ids':ids,'labels':labels,'supervised':c['supervised_token_count'],'token_label_sha256':h})
 return out
def collate(group,torch):
 width=max(len(r['ids']) for r in group); ids=[]; mask=[]; labels=[]
 for r in group:
  pad=width-len(r['ids']); ids.append(r['ids']+[13]*pad); mask.append([1]*len(r['ids'])+[0]*pad); labels.append(r['labels']+[IGNORE]*pad)
 return torch.tensor(ids,device='cuda:0'),torch.tensor(mask,device='cuda:0'),torch.tensor(labels,device='cuda:0')
def score(model,rows,n,position_mode,torch,F):
 evidence=[]; family=defaultdict(lambda:[0.0,0,[]])
 with torch.no_grad():
  for start in range(0,len(rows),n):
   group=rows[start:start+n]; ids,mask,labels=collate(group,torch); kw={'input_ids':ids,'attention_mask':mask,'use_cache':False}
   if position_mode=='EXPLICIT': kw['position_ids']=explicit_position(mask,torch)
   logits=model(**kw).logits.float(); y=labels[:,1:]; z=logits[:,:-1,:]
   loss=F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1),ignore_index=IGNORE,reduction='none').view_as(y)
   for i,r in enumerate(group):
    valid=y[i]!=IGNORE; values=[float(v) for v in loss[i][valid].tolist()]; check(len(values)==r['supervised'],'SHIFTED_SUPERVISION_MISMATCH:'+r['episode_id'])
    total=sum(values); family[r['family']][0]+=total;family[r['family']][1]+=len(values);family[r['family']][2].append(total/len(values))
    evidence.append({'episode_id':r['episode_id'],'family':r['family'],'token_label_sha256':r['token_label_sha256'],'supervised_token_count':len(values),'per_token_negative_log_likelihood':values,'row_nll_sum':total,'row_token_micro_mean_nll':total/len(values)})
 return {'families':{f:{'nll_sum':v[0],'supervised_tokens':v[1],'token_micro_mean_nll':v[0]/v[1],'row_macro_mean_nll':sum(v[2])/len(v[2])} for f,v in family.items()},'raw':evidence}
def mean(vals): return sum(vals)/len(vals)
def max_pair(vals): return max(abs(a-b) for i,a in enumerate(vals) for b in vals[i+1:]) if len(vals)>1 else 0.0
def decide(allm,thr):
 repeats_ok=True; implicit_ok=True; explicit_ok=True; semantic_ok=True
 report={}
 for ckpt,modes in allm.items():
  report[ckpt]={}
  for mode,reps in modes.items():
   report[ckpt][mode]={}
   for fam in WEAK:
    vals=[x['families'][fam]['token_micro_mean_nll'] for x in reps]; d=max_pair(vals); report[ckpt][mode][fam]={'repeat_drift':d}; repeats_ok &= d<=thr
  for fam in WEAK:
   bi=mean([x['families'][fam]['token_micro_mean_nll'] for x in modes['REFERENCE_B1_IMPLICIT_POSITION']]); be=mean([x['families'][fam]['token_micro_mean_nll'] for x in modes['REFERENCE_B1_EXPLICIT_POSITION_CONTROL']]); mi=mean([x['families'][fam]['token_micro_mean_nll'] for x in modes['BATCHED_IMPLICIT_POSITION']]); me=mean([x['families'][fam]['token_micro_mean_nll'] for x in modes['BATCHED_EXPLICIT_POSITION_CONTROL']]);
   di=abs(bi-mi); de=abs(be-me); ds=abs(be-bi); report[ckpt].setdefault('comparisons',{})[fam]={'implicit_batch_drift':di,'explicit_batch_drift':de,'b1_semantic_drift':ds}; implicit_ok &= di<=thr; explicit_ok &= de<=thr; semantic_ok &= ds<=thr
 if repeats_ok and implicit_ok:return 'MEASUREMENT_VALID',report
 if repeats_ok and (not implicit_ok) and explicit_ok and semantic_ok:return 'CONTROLLABLE_IMPLEMENTATION_DEFECT',report
 return 'NO_STABLE_MODE',report
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--authorization',type=Path,required=True);ap.add_argument('--train',type=Path,required=True);ap.add_argument('--out',type=Path);ap.add_argument('--self-test',action='store_true');a=ap.parse_args();cfg=loadj(a.config);auth=loadj(a.authorization); cohort=validate(cfg,auth)
 if a.self_test:
  check(a.train.is_file(),'TRAIN_MISSING'); rows_from_train(cohort,a.train); print(json.dumps({'status':'PASS_CPU_ONLY_NO_MODEL_LOAD','rows':30,'modes':4,'repeats':3},sort_keys=True)); return
 check(a.out is not None and not a.out.exists(),'FRESH_OUTPUT_REQUIRED'); out=a.out; terminal={'schema_version':1,'protocol_id':cfg['protocol_id'],'status':'PRE_MODEL_FAILURE','optimizer_steps':0,'scientific_gpu_training_seconds':0.0,'generation_performed':False,'checkpoint_mutated':False,'final_audit_opened':False,'config_sha256':sha(a.config),'authorization_sha256':sha(a.authorization)}
 try:
  check(sha(a.train)==cfg['inputs']['train_sha256'],'TRAIN_SHA_MISMATCH'); rows=rows_from_train(cohort,a.train); import torch,torch.nn.functional as F; from transformers import AutoModelForCausalLM; from peft import PeftModel
  check(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA_BF16_UNAVAILABLE'); torch.manual_seed(20261009);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False; os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; os.environ['TOKENIZERS_PARALLELISM']='false'
  allm={}; raw=[]; selected=None; started=time.monotonic()
  for name,cond in cfg['inputs']['conditions'].items():
   mp=rp(cond['checkpoint_manifest_path']); check(sha(mp)==cond['checkpoint_manifest_sha256'],'MANIFEST_SHA_MISMATCH:'+name); result_path=rp(cond['result_path']); check(sha(result_path)==cond['result_sha256'],'RESULT_SHA_MISMATCH:'+name); manifest=loadj(mp); adapter=Path(manifest['adapter_path'])/'adapter_model.safetensors'; check(sha(adapter)==cond['adapter_sha256'],'ADAPTER_SHA_MISMATCH:'+name)
   model=AutoModelForCausalLM.from_pretrained(manifest['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0'); model=PeftModel.from_pretrained(model,manifest['adapter_path'],is_trainable=False);model.eval(); modes={}
   if name=='CAPABILITY_REPAIR_BASELINE_V1_V7':
    for candidate in [32,16,8,4,2]:
     try: score(model,rows,candidate,'IMPLICIT',torch,F);selected=candidate;break
     except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache()
    check(selected is not None,'NO_BATCH_GT_ONE')
   check(selected is not None,'COMMON_BATCH_UNBOUND')
   for mode,pos,n in [('REFERENCE_B1_IMPLICIT_POSITION','IMPLICIT',1),('REFERENCE_B1_EXPLICIT_POSITION_CONTROL','EXPLICIT',1),('BATCHED_IMPLICIT_POSITION','IMPLICIT',selected),('BATCHED_EXPLICIT_POSITION_CONTROL','EXPLICIT',selected)]:
    reps=[]
    for repeat in range(3):
     check(time.monotonic()-started<=cfg['output_contract']['runtime_cap_seconds'],'RUNTIME_CAP_REACHED');x=score(model,rows,n,pos,torch,F); reps.append(x);raw.append({'checkpoint_id':name,'mode_id':mode,'repeat_index':repeat,'effective_batch_size':n,'rows':x['raw']})
    modes[mode]=reps
   allm[name]=modes;del model;torch.cuda.empty_cache()
  decision,comparisons=decide(allm,cfg['numeric_acceptance']['absolute_token_micro_nll_stability_threshold']);out.mkdir(parents=True); rawpath=out/'RAW_PER_TOKEN_NLL_EVIDENCE.jsonl';rawpath.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in raw),encoding='utf-8');result={'schema_version':1,'protocol_id':cfg['protocol_id'],'status':'COMPLETE','decision':decision,'common_effective_batched_size':selected,'comparisons':comparisons,'raw_evidence_sha256':sha(rawpath),'raw_evidence_rows':len(raw),'optimizer_steps':0,'scientific_gpu_training_seconds':0.0,'generation_performed':False,'checkpoint_mutated':False,'final_audit_opened':False};atomic(out/'MEASUREMENT_FOUNDATIONS_RESULT.json',result);terminal.update({'status':'COMPLETE','decision':decision,'result_sha256':sha(out/'MEASUREMENT_FOUNDATIONS_RESULT.json'),'raw_evidence_sha256':sha(rawpath),'common_effective_batched_size':selected})
 except Exception as e: terminal.update({'status':'PRE_MODEL_FAILURE' if terminal['status']=='PRE_MODEL_FAILURE' else 'FAILURE','error':type(e).__name__+':'+str(e),'traceback':traceback.format_exc(limit=6)})
 finally:
  if a.out: atomic(Path(a.out)/'TERMINAL_RECEIPT.json',terminal)
 if terminal['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()