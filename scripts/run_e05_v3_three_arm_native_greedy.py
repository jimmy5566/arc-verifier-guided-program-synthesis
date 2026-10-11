"""E05 V3 three-arm target-blind native Greedy worker; imports models only after preflight."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.e05_v3_three_arm_native_greedy import E05Failure, OUT, PROMPTS, sha, rows, validate_preimport
from scripts.e04_orientation_v3_no_update import atomic_json, atomic_jsonl, canon, prompt_messages
CAP=10800

def fail(x): raise E05Failure(x)
def check(deadline):
 if time.monotonic()>=deadline: fail('E05_ACTIVE_CAP_EXCEEDED')
def load(path:Path): return json.loads(path.read_text(encoding='utf-8-sig'))
def runtime_preflight(config:Path,binding:Path,out:Path):
 c=load(config); b=load(binding); v=validate_preimport()
 req={'schema_version','protocol_id','source_commit','worker_sha256','config_sha256','execution_binding_sha256','output_root','nonce','active_hard_cap_seconds','authorization','retry'}
 if set(b)!=req or b['protocol_id']!=c['protocol_id'] or b['active_hard_cap_seconds']!=CAP or b['authorization']!='DIRECTOR_CONDITIONAL_ONCE' or b['retry'] is not False: fail('E05_RUNTIME_BINDING_SCHEMA')
 if b['output_root']!=str(out) or not b['nonce'] or sha(Path(__file__))!=b['worker_sha256'] or sha(config)!=b['config_sha256']: fail('E05_RUNTIME_BINDING_IDENTITY')
 eb=OUT/'E05_V3_THREE_ARM_EXECUTION_BINDING_REPAIR.json'
 if sha(eb)!=b['execution_binding_sha256'] or c['execution_binding_sha256']!=b['execution_binding_sha256']: fail('E05_EXECUTION_BINDING_IDENTITY')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=b['source_commit']: fail('E05_SOURCE_COMMIT')
 if out.exists(): fail('E05_FRESH_OUTPUT_REQUIRED')
 if c['physical_batch_size']!=1 or c['decoder']!='native_greedy' or c['max_new_tokens']!=256 or c['context_window']!=8192: fail('E05_RUNTIME_CONFIG')
 return c,b,v

def adapter_dir(arm:dict[str,Any],temp:Path):
 if arm['arm_id']=='RAW_V7_NO_E04E_UPDATE': return Path(arm['adapter_path'])
 from huggingface_hub import hf_hub_download
 token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
 if not token: fail('AUTH_MISSING_HUGGINGFACE')
 d=temp/arm['arm_id'];d.mkdir(parents=True)
 for name in ('adapter_model.safetensors','adapter_config.json'):
  path=Path(hf_hub_download(arm['repository']['repo_id'],filename=arm['filename'].rsplit('/',1)[0]+'/'+name,repo_type='model',revision=arm['repository']['immutable_revision'],token=token,local_dir=str(d),force_download=True))
  if name=='adapter_model.safetensors' and sha(path)!=arm['adapter_sha256']: fail('E05_ADAPTER_SHA256:'+arm['arm_id'])
  shutil.copyfile(path,d/name)
 return d

def raw(arm_id,index,row,g):
 from inference.nvarc_native import parse_native_grid
 x=parse_native_grid(g.text)
 return {'arm_id':arm_id,'row_index':index,'episode_id':row['episode_id'],'input_sha256':hashlib.sha256(canon(row)).hexdigest(),'text':g.text,'parser_valid':x is not None,'parsed_grid':[list(z) for z in x] if x is not None else None,'prompt_tokens':int(g.prompt_tokens),'completion_tokens':int(g.completion_tokens),'elapsed_seconds':float(g.elapsed_seconds),'effective_batch_size':1}

def run(config:Path,binding:Path,out:Path):
 started=time.monotonic(); terminal=out/'TERMINAL_RECEIPT.json'; partial=[]
 try:
  c,b,_=runtime_preflight(config,binding,out); out.mkdir(parents=True); deadline=started+CAP
  eb=load(OUT/'E05_V3_THREE_ARM_EXECUTION_BINDING_REPAIR.json'); checkpoint=load(ROOT/'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json'); prompts=rows(PROMPTS)
  # Target sidecar is deliberately never opened by this worker.
  for arm in eb['arms']:
   check(deadline); from transformers import AutoModelForCausalLM; from peft import PeftModel; import torch
   base=Path(checkpoint['base_path'])
   for entry in checkpoint['base_files']:
    f=base/entry['name']
    if not f.is_file() or f.stat().st_size!=entry['bytes'] or sha(f)!=entry['sha256']: fail('E05_BASE_IDENTITY:'+entry['name'])
   if arm['arm_id']=='RAW_V7_NO_E04E_UPDATE':
    for entry in checkpoint['adapter_files']:
     f=Path(arm['adapter_path'])/entry['name']
     if not f.is_file() or f.stat().st_size!=entry['bytes'] or sha(f)!=entry['sha256']: fail('E05_RAW_V7_IDENTITY:'+entry['name'])
   ad=adapter_dir(arm,out/'_adapter_cache')
   model=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0')
   model=PeftModel.from_pretrained(model,str(ad),is_trainable=False).eval()
   from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer
   tok,_=checkpoint_native_tokenizer(base,Path('/workspace/arc2/native_config')); provider=NVARCNativeProvider(model_path=base,tokenizer_config_dir=Path('/workspace/arc2/native_config'),device='cuda:0'); provider.model=model;provider.tokenizer=tok
   result=[]
   for i,row in enumerate(prompts):
    check(deadline); g=provider.generate_many([prompt_messages(row['prompt'])],max_new_tokens=c['max_new_tokens'],context_window=c['context_window'],seeds=[c['seed']+i])[0];result.append(raw(arm['arm_id'],i,row,g))
   name=arm['arm_id']+'_RAW_GENERATIONS.jsonl'; atomic_jsonl(out/name,result); partial.append({'arm_id':arm['arm_id'],'path':name,'sha256':sha(out/name),'rows':len(result)})
   del provider,model; torch.cuda.empty_cache()
  receipt={'schema_version':1,'status':'COMPLETE_NO_UPDATE','protocol_id':c['protocol_id'],'binding_sha256':sha(binding),'source_commit':b['source_commit'],'arms':partial,'target_sidecar_accessed':False,'optimizer_steps':0,'parameter_updates':0,'backward_calls':0,'active_cap_seconds':CAP,'wall_seconds':time.monotonic()-started,'final_audit_opened':False}; atomic_json(terminal,receipt)
 except Exception as e:
  out.mkdir(parents=True,exist_ok=True);atomic_json(terminal,{'schema_version':1,'status':'FAILED_NO_UPDATE','error':str(e),'error_type':type(e).__name__,'partial_evidence':partial,'target_sidecar_accessed':False,'optimizer_steps':0,'parameter_updates':0,'backward_calls':0,'final_audit_opened':False,'wall_seconds':time.monotonic()-started});raise

def main():
 a=argparse.ArgumentParser();a.add_argument('--config',type=Path);a.add_argument('--binding',type=Path);a.add_argument('--output-root',type=Path);a.add_argument('--self-test',action='store_true');x=a.parse_args()
 if x.self_test: print(json.dumps(validate_preimport(),sort_keys=True));return
 if not(x.config and x.binding and x.output_root):raise SystemExit('CONFIG_BINDING_OUTPUT_REQUIRED')
 run(x.config,x.binding,x.output_root)
if __name__=='__main__':main()
