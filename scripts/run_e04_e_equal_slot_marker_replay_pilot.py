"""One detached two-arm E04-E V2 training and target-blind generation worker."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.e04_e_equal_slot_execution import (ARMS,BASE,E04EFailure,JOINT_CAP_SECONDS,PER_ARM_CAP_SECONDS,PROTOCOL_ID,atomic_json,fail,read_json,relative,sha_path,static_schedule_preflight,build_arm_samples)
from scripts.e04_e_equal_slot_loss import backward_equal_slot_term

def cuda_memory(torch=None)->dict[str,object]:
 if torch is None:return {'status':'UNAVAILABLE'}
 return {'status':'OBSERVED','allocated_bytes':int(torch.cuda.memory_allocated()),'reserved_bytes':int(torch.cuda.memory_reserved()),'peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),'peak_reserved_bytes':int(torch.cuda.max_memory_reserved())}

def _verify_file(rel:str,expected:str,code:str)->Path:
 p=ROOT/rel
 if not p.is_file() or sha_path(p)!=expected:fail('E04E_'+code+'_HASH')
 return p

def load_binding(path:Path,output:Path,launch_commit:str,*,child:bool=False)->dict[str,Any]:
 b=read_json(path); req={'schema_version','protocol_id','authorization_id','director_response_path','director_response_sha256','execution_authorized','worker_source_commit','worker_path','worker_sha256','evaluator_path','evaluator_sha256','launcher_path','launcher_sha256','loss_path','loss_sha256','cpu_loss_test_path','cpu_loss_test_sha256','protocol_path','protocol_sha256','control_schedule_path','control_schedule_sha256','treatment_schedule_path','treatment_schedule_sha256','token_accounting_path','token_accounting_sha256','checkpoint_manifest_path','checkpoint_manifest_sha256','runtime_config_path','runtime_config_sha256','v7_reference_raw_path','v7_reference_raw_sha256','arm_order','seed','arms','jobs','retry','nonce','output_root','arm_output_roots','per_arm_runtime_cap_seconds','joint_runtime_cap_seconds','static_preflight'}
 if set(b)!=req or b['protocol_id']!=PROTOCOL_ID:fail('E04E_BINDING_SCHEMA')
 if not(b['execution_authorized'] is True and b['jobs']==1 and b['arms']==2 and b['retry'] is False and b['arm_order']==list(ARMS)):fail('E04E_BINDING_AUTH')
 if b['per_arm_runtime_cap_seconds']!=PER_ARM_CAP_SECONDS or b['joint_runtime_cap_seconds']!=JOINT_CAP_SECONDS:fail('E04E_BINDING_CAP')
 if str(output).replace('\\','/')!=b['output_root'] or not b['nonce']:fail('E04E_BINDING_OUTPUT')
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 if head!=launch_commit:fail('E04E_LAUNCH_COMMIT')
 if subprocess.call(['git','merge-base','--is-ancestor',b['worker_source_commit'],launch_commit],cwd=ROOT)!=0:fail('E04E_SOURCE_ANCESTRY')
 for field,code in [('worker','WORKER'),('evaluator','EVALUATOR'),('launcher','LAUNCHER'),('loss','LOSS'),('cpu_loss_test','LOSS_TEST'),('protocol','PROTOCOL'),('control_schedule','CONTROL_SCHEDULE'),('treatment_schedule','TREATMENT_SCHEDULE'),('token_accounting','TOKEN_ACCOUNTING'),('checkpoint_manifest','CHECKPOINT_MANIFEST'),('runtime_config','RUNTIME_CONFIG'),('v7_reference_raw','V7_RAW'),('director_response','DIRECTOR_RESPONSE')]:_verify_file(b[field+'_path'],b[field+'_sha256'],code)
 response=read_json(ROOT/b['director_response_path'])
 auth=response.get('execution_authorization',{})
 if response.get('response_id')!='E04_E_V2_RUN_001_OOM_RESULT_AND_REPLACEMENT_AUTHORIZATION_BRIEF_RESPONSE' or response.get('decision')!='CONTINUE_CONTROLLER' or auth.get('maximum_detached_jobs')!=1 or auth.get('authorized_run')!='RUN_002' or auth.get('arm_names')!=list(ARMS) or auth.get('per_arm_runtime_cap_seconds')!=PER_ARM_CAP_SECONDS or auth.get('joint_runtime_cap_seconds')!=JOINT_CAP_SECONDS:fail('E04E_DIRECTOR_AUTH')
 if output.exists():
  # The parent owns the root.  A per-arm child may only inherit its two
  # pre-created bookkeeping entries; any other content is a duplicate/stale run.
  allowed={'PREFLIGHT_RECEIPT.json','arms'}
  present={p.name for p in output.iterdir()}
  if not child or not present.issubset(allowed):fail('E04E_OUTPUT_NOT_FRESH')
 return b

def _verify_checkpoint(m:dict[str,Any])->None:
 for root_key,files_key in (('base_path','base_files'),('adapter_path','adapter_files')):
  base=Path(m[root_key])
  for x in m[files_key]:
   p=base/x['name']
   if not p.is_file() or p.stat().st_size!=x['bytes'] or sha_path(p)!=x['sha256']:fail('E04E_CHECKPOINT:'+x['name'])

def _runtime_rows(binding:dict[str,Any])->tuple[dict[str,list[dict[str,Any]]],dict[str,Any]]:
 proof=static_schedule_preflight(); manifest=read_json(ROOT/binding['checkpoint_manifest_path']);config=read_json(ROOT/binding['runtime_config_path'])
 from inference.nvarc_native import checkpoint_native_tokenizer
 tokenizer,meta=checkpoint_native_tokenizer(Path(manifest['base_path']),Path(config['native_config_dir']))
 schedules={arm:read_json(BASE/('CONTROL_SCHEDULE.json' if arm==ARMS[0] else 'TREATMENT_SCHEDULE.json')) for arm in ARMS}; values={}
 for arm in ARMS:
  rows=[]
  for item in build_arm_samples(arm,schedules[arm]):
   sample=item['sample'];actual=tokenizer.apply_chat_template(sample['messages'],add_generation_prompt=False,tokenize=True,return_tensors=None);ids=list(actual[0] if actual and isinstance(actual[0],list) else actual)
   if ids!=sample['input_ids']:fail('E04E_PRODUCTION_TOKENIZER:'+str(item['slot']))
   rows.append({'slot':item['slot'],'episode':item['episode'],'input_ids':ids,'labels':list(sample['labels']),'attention_mask':[1]*len(ids),'supervised':item['supervised']})
  values[arm]=rows
 return values,{'status':'PASS_PRODUCTION_SERIALIZER_COLLATOR','static':proof,'tokenizer':meta,'model_imported':False,'optimizer_constructed':False,'target_sidecar_accessed':False}

def _load_model(manifest:dict[str,Any]):
 import torch
 from peft import PeftModel
 from transformers import AutoModelForCausalLM
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():fail('E04E_CUDA_BF16')
 m=AutoModelForCausalLM.from_pretrained(manifest['base_path'],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0')
 m=PeftModel.from_pretrained(m,manifest['adapter_path'],is_trainable=True);m.enable_input_require_grads();m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
 params=[p for p in m.parameters() if p.requires_grad]
 if not params or any(p.requires_grad for n,p in m.named_parameters() if 'lora_' not in n):fail('E04E_LORA_PARTITION')
 return m,params,torch

def _generate(model,manifest:dict[str,Any],config:dict[str,Any],deadline:float,arm_root:Path)->None:
 from inference.nvarc_native import NVARCNativeProvider,checkpoint_native_tokenizer,parse_native_grid
 from scripts.e04_orientation_v3_no_update import canon,prompt_messages,read_jsonl
 from scripts.e04_v3_full768_b1_remeasurement import atomic_jsonl,validate_full_raw
 tokenizer,_=checkpoint_native_tokenizer(Path(manifest['base_path']),Path(config['native_config_dir']))
 model.eval();provider=NVARCNativeProvider(model_path=Path(manifest['base_path']),tokenizer_config_dir=Path(config['native_config_dir']),device='cuda:0');provider.model,provider.tokenizer=model,tokenizer
 prompts=read_jsonl(ROOT/config['prompt_path']);rows=[]
 for i,prompt in enumerate(prompts):
  if time.monotonic()>=deadline:fail('E04E_PER_ARM_CAP_EVAL')
  out=provider.generate_many([prompt_messages(prompt)],max_new_tokens=int(config['max_new_tokens']),context_window=int(config['context_window']),seeds=[int(config['seed'])+i])
  if len(out)!=1:fail('E04E_GENERATION_ALIGNMENT')
  x=out[0]; tokens=[int(v) for v in x.token_ids]; parsed=parse_native_grid(x.text)
  rows.append({'row_index':i,'input_sha256':hashlib.sha256(canon(prompt)).hexdigest(),'generated_token_ids':tokens,'text':x.text,'parser_valid':parsed is not None,'parsed_grid':None if parsed is None else [list(r) for r in parsed],'prompt_tokens':int(x.prompt_tokens),'completion_tokens':int(x.completion_tokens),'elapsed_seconds':float(x.elapsed_seconds),'effective_batch_size':1,'physical_returned_token_ids':tokens,'pad_token_count':0})
 raw=arm_root/'PRIMARY_B1_RAW.jsonl';atomic_jsonl(raw,rows);validate_full_raw(raw,prompts)
 atomic_json(arm_root/'RAW_GENERATION_RECEIPT.json',{'status':'RAW_GENERATIONS_FROZEN_NO_TARGETS','rows':len(rows),'raw_sha256':sha_path(raw),'physical_batch_size':1,'target_sidecar_accessed':False,'final_audit_opened':False})

def run_arm(binding_path:Path,output:Path,launch_commit:str,arm:str)->None:
 started=time.monotonic();root=output/'arms'/arm;terminal_receipt=root/'TERMINAL_RECEIPT.json'
 try:
  b=load_binding(binding_path,output,launch_commit,child=True)
  if arm not in ARMS or str(root).replace('\\','/')!=b['arm_output_roots'][arm] or root.exists():fail('E04E_ARM_OUTPUT')
  manifest=read_json(ROOT/b['checkpoint_manifest_path']);_verify_checkpoint(manifest);rows,runtime=_runtime_rows(b)
  root.mkdir(parents=True);atomic_json(root/'PRE_OPTIMIZER_PREFLIGHT.json',{'arm':arm,'runtime':runtime,'model_imported':False,'optimizer_constructed':False,'target_sidecar_accessed':False})
  import bitsandbytes as bnb
  os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['TOKENIZERS_PARALLELISM']='false'
  import torch
  torch.manual_seed(int(b['seed']));torch.cuda.manual_seed_all(int(b['seed']))
  model,params,torch=_load_model(manifest);optimizer=bnb.optim.PagedAdamW8bit(params,lr=0.00005);torch.cuda.reset_peak_memory_stats();memory_after_model_load=cuda_memory(torch);deadline=started+PER_ARM_CAP_SECONDS
  processed=supervised=steps=0
  for offset in range(0,384,4):
   if time.monotonic()>=deadline:fail('E04E_PER_ARM_CAP_TRAIN')
   group=rows[arm][offset:offset+4]
   if len(group)!=4 or {x['episode']['within_step_role_position'] for x in group}!={0,1,2,3}:fail('E04E_TRAIN_STEP_GROUP')
   optimizer.zero_grad(set_to_none=True); counts=[]; loss_value=0.0; slot_memory=[]
   expected_counts=[x['supervised'] for x in group]
   for row in group:
    ids=torch.tensor([row['input_ids']],device='cuda:0',dtype=torch.long);lab=torch.tensor([row['labels']],device='cuda:0',dtype=torch.long);att=torch.tensor([row['attention_mask']],device='cuda:0',dtype=torch.long)
    out=model(input_ids=ids,attention_mask=att,use_cache=False)
    slot_term,count=backward_equal_slot_term(out.logits,lab);counts.append(count);loss_value+=float(slot_term);processed+=len(row['input_ids']);supervised+=row['supervised'];slot_memory.append({'slot':int(row['slot']),'supervised_token_count':int(count),'cuda_memory':cuda_memory(torch)});del out,slot_term
   if counts!=expected_counts or not bool(torch.isfinite(torch.tensor(loss_value))):fail('E04E_LOSS_CONTRACT')
   for cfg in optimizer.param_groups:cfg['lr']=0.00005*min(1.0,(steps+1)/3.0)
   optimizer.step();steps+=1
   atomic_json(root/'TRAINING_PROGRESS.json',{'status':'RUNNING','arm':arm,'optimizer_steps':steps,'processed_tokens':processed,'supervised_tokens':supervised,'loss_normalization':'four_independent_slot_means_times_0.25','cuda_memory_after_model_load':memory_after_model_load,'cuda_memory_per_slot':slot_memory,'cuda_memory_after_step':cuda_memory(torch),'target_sidecar_accessed':False})
  expected=runtime['static']['arm_totals'][arm]
  if steps!=96 or processed!=expected['raw_transformer_tokens'] or supervised!=expected['raw_supervised_tokens']:fail('E04E_COMPLETION_TOTALS')
  model.save_pretrained(root/'adapter');config=read_json(ROOT/b['runtime_config_path']);_generate(model,manifest,config,deadline,root)
  atomic_json(terminal_receipt,{'status':'COMPLETED_RAW_FROZEN_NO_TARGETS','protocol_id':PROTOCOL_ID,'arm':arm,'optimizer_steps':steps,'parameter_updates':steps,'processed_tokens':processed,'supervised_tokens':supervised,'loss_normalization':'four_independent_slot_means_times_0.25','adapter_sha256':sha_path(root/'adapter'/'adapter_model.safetensors'),'raw_sha256':sha_path(root/'PRIMARY_B1_RAW.jsonl'),'elapsed_seconds':time.monotonic()-started,'runtime_cap_seconds':PER_ARM_CAP_SECONDS,'cuda_memory_after_model_load':memory_after_model_load,'cuda_memory_final':cuda_memory(torch),'target_sidecar_accessed':False,'final_audit_opened':False})
 except Exception as e:
  root.mkdir(parents=True,exist_ok=True);atomic_json(terminal_receipt,{'status':'FAILED_OR_PARTIAL','protocol_id':PROTOCOL_ID,'arm':arm,'error':f'{type(e).__name__}:{e}','elapsed_seconds':time.monotonic()-started,'cuda_memory_final':cuda_memory(locals().get('torch')),'target_sidecar_accessed':False,'final_audit_opened':False,'traceback':traceback.format_exc(limit=4)});raise

def run_parent(binding_path:Path,output:Path,launch_commit:str)->None:
 started=time.monotonic();terminal_receipt=output/'TERMINAL_RECEIPT.json'
 try:
  b=load_binding(binding_path,output,launch_commit);proof=static_schedule_preflight();output.mkdir();(output/'arms').mkdir();atomic_json(output/'PREFLIGHT_RECEIPT.json',{'status':'PASS_CPU_NO_MODEL_IMPORT','binding_sha256':sha_path(binding_path),'launch_commit':launch_commit,'proof':proof,'model_imported':False,'optimizer_constructed':False,'target_sidecar_accessed':False})
  for arm in ARMS:
   if time.monotonic()-started>=JOINT_CAP_SECONDS:fail('E04E_JOINT_CAP')
   r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--binding',str(binding_path),'--output-root',str(output),'--launch-commit',launch_commit,'--arm',arm],cwd=ROOT,check=False)
   if r.returncode:fail('E04E_ARM_FAILED:'+arm)
  raw={a:sha_path(output/'arms'/a/'PRIMARY_B1_RAW.jsonl') for a in ARMS}
  receipt={'status':'COMPLETE_PENDING_CPU_SCORE','protocol_id':PROTOCOL_ID,'binding_sha256':sha_path(binding_path),'launch_commit':launch_commit,'arm_order':list(ARMS),'optimizer_steps_by_arm':{a:96 for a in ARMS},'parameter_updates_by_arm':{a:96 for a in ARMS},'raw_sha256':raw,'target_sidecar_accessed':False,'final_audit_opened':False,'wall_seconds':time.monotonic()-started,'joint_runtime_cap_seconds':JOINT_CAP_SECONDS}
  atomic_json(output/'RAW_FREEZE_RECEIPT.json',receipt);atomic_json(terminal_receipt,receipt)
 except Exception as e:
  output.mkdir(parents=True,exist_ok=True);atomic_json(terminal_receipt,{'status':'FAILED_OR_PARTIAL','protocol_id':PROTOCOL_ID,'error':f'{type(e).__name__}:{e}','wall_seconds':time.monotonic()-started,'cuda_memory_final':cuda_memory(locals().get('torch')),'target_sidecar_accessed':False,'final_audit_opened':False,'traceback':traceback.format_exc(limit=4)});raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path);p.add_argument('--output-root',type=Path);p.add_argument('--launch-commit');p.add_argument('--arm',choices=ARMS);p.add_argument('--self-test',action='store_true');p.add_argument('--identity-check',action='store_true');a=p.parse_args()
 if a.self_test:print(json.dumps(static_schedule_preflight(),sort_keys=True));return
 if not a.binding or not a.output_root or not a.launch_commit:raise SystemExit('E04E_ARGS')
 if a.identity_check:
  b=load_binding(a.binding,a.output_root,a.launch_commit)
  print(json.dumps({'status':'PASS_E04E_BINDING_IDENTITY_CPU_ONLY','binding_sha256':sha_path(a.binding),'worker_source_commit':b['worker_source_commit'],'static_preflight':static_schedule_preflight()},sort_keys=True));return
 if a.arm:run_arm(a.binding,a.output_root,a.launch_commit,a.arm)
 else:run_parent(a.binding,a.output_root,a.launch_commit)
if __name__=='__main__':main()
