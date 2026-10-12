#!/usr/bin/env python3
"""No-update worker. CPU self-test is the only enabled mode before authorization."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.connected_component_fit_generation_runtime_contract import teacher_forced_layout,token_metric,aggregate_token_metrics,first_divergence,semantic_taxonomy,batch_gate,validate_sidecar,require_inventory_partition,validate_reuse
import hashlib

def self_test():
    layout=teacher_forced_layout([11,1],[2,15]); assert layout['labels']==[-100,-100,2,15] and layout['target_logit_indices']==[1,2]
    metrics=[token_metric([0,3,1],1),token_metric([0,1,3],2)]; agg=aggregate_token_metrics(metrics); assert agg['top1_count']==2
    assert first_divergence([1,2],[1,3])==1 and first_divergence([1],[1,2])==1
    assert semantic_taxonomy([[1]],[[1]],True)['primary']=='EXACT'
    tol={'correct_token_rank_abs_delta_max':1,'Top1_and_Top2_aggregate_count_delta_max':0,'per_row_mean_margin_abs_delta_max':.02,'token_weighted_NLL_abs_delta_max':.005}
    assert batch_gate({'a':metrics[0]},{'a':metrics[0]},tol)['status']=='PASS'
    return {'status':'PASS_CPU_PRODUCTION_SHAPED_WORKER_SELF_TEST','model_loaded':False,'gpu_used':False,'optimizer_steps':0,'backward_calls':0}

def _write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n');os.replace(tmp,path)
def _read_rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def _write_rows(path,values):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(b''.join(json.dumps(x,sort_keys=True,separators=(',',':')).encode()+b'\n' for x in values));os.replace(tmp,path)

def _native_generate(checkpoint,items,binding):
 """Real no-update native-greedy generation for the explicitly missing rows."""
 import torch
 from peft import PeftModel
 from transformers import AutoModelForCausalLM
 from inference.nvarc_native import NVARCNativeProvider,checkpoint_native_tokenizer,parse_native_grid,serialize_grid
 runtime=binding['checkpoint_runtime'][checkpoint]
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA_BF16_REQUIRED')
 base=AutoModelForCausalLM.from_pretrained(runtime['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0')
 model=PeftModel.from_pretrained(base,runtime['adapter_path'],is_trainable=False).eval()
 tok,_=checkpoint_native_tokenizer(Path(runtime['base_path']),Path(runtime['tokenizer_config_dir']))
 provider=NVARCNativeProvider(model_path=Path(runtime['base_path']),tokenizer_config_dir=Path(runtime['tokenizer_config_dir']),device='cuda:0');provider.model,provider.tokenizer=model,tok
 def messages(task):
  out=[]
  for pair in task['train']: out += [{'role':'user','content':serialize_grid(pair['input'])},{'role':'assistant','content':serialize_grid(pair['output'])}]
  out.append({'role':'user','content':serialize_grid(task['test'][0]['input'])});return out
 ladder=[32,16,8,4,1];size=32;index=0;output=[];events=[];ordered=sorted(items,key=lambda x:(len(json.dumps(x['task'],sort_keys=True)),x['cohort_record_id']))
 while index<len(ordered):
  batch=ordered[index:index+size]
  try: got=provider.generate_many([messages(x['task']) for x in batch],max_new_tokens=int(binding['execution']['max_new_tokens']),context_window=int(binding['execution']['context_window']),seeds=[int(binding['execution']['seed'])+index+i for i in range(len(batch))])
  except RuntimeError as exc:
   if 'out of memory' not in str(exc).lower() or size==1: raise
   next_size=ladder[ladder.index(size)+1];events.append({'from':size,'to':next_size,'reason':'CUDA_OOM'});size=next_size;torch.cuda.empty_cache();continue
  for item,result in zip(batch,got,strict=True):
   parsed=parse_native_grid(result.text);output.append({'checkpoint':checkpoint,'cohort_record_id':item['cohort_record_id'],'prompt_sha256':item['prompt_task_sha256'],'generated_token_ids':[int(x) for x in result.token_ids],'parsed_grid':None if parsed is None else [list(x) for x in parsed],'parser_valid':parsed is not None,'effective_batch_size':size,'provenance':'GENERATE_NEW_ONCE_AFTER_FRESH_AUTHORIZATION'})
  index+=len(batch)
 del model;torch.cuda.empty_cache();return output,events

def execute(binding_path,output_root):
 """Import 168 exact rows, generate only 48, then freeze raw evidence."""
 b=json.loads(Path(binding_path).read_text()); execution=b.get('execution',{})
 if execution.get('optimizer_steps') not in (0,None) or execution.get('backward_calls') not in (0,None) or execution.get('retry') not in (False,None): raise RuntimeError('NO_UPDATE_EXECUTION_REQUIRED')
 for rel,digest in b.get('source_hashes',{}).items():
  if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest: raise RuntimeError('SOURCE_IDENTITY_MISMATCH:'+rel)
 inventory=json.loads(Path(execution['inventory_path']).read_text());reused,fresh=require_inventory_partition(inventory)
 out=Path(output_root)
 if out.exists(): raise RuntimeError('FRESH_OUTPUT_REQUIRED')
 old={}
 for path in Path(execution['run_002_raw_root']).glob('*_RAW.jsonl'): old[path.name[:-9]]={row.get('prompt_sha256'):row for row in _read_rows(path)}
 validate_reuse(old,inventory); imported=[]
 for item in reused:
  row=dict(old[item['checkpoint']][item['prompt_task_sha256']]);
  if {'target_grid','target_token_ids','gold','dGold'} & set(row) or hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=item['run_002_raw_record_sha256']: raise RuntimeError('REUSED_RAW_IDENTITY_MISMATCH')
  row.update({'checkpoint':item['checkpoint'],'cohort_record_id':item['cohort_record_id'],'provenance':'REUSE_IMMUTABLE_RUN_002_RAW'});imported.append(row)
 tasks_document=json.loads(Path(execution['missing_prompt_tasks_path']).read_text());tasks=tasks_document.get('records',tasks_document) if isinstance(tasks_document,dict) else tasks_document
 if not isinstance(tasks,list) or (isinstance(tasks_document,dict) and tasks_document.get('target_content_present') is not False): raise RuntimeError('MISSING_PROMPT_TARGET_BOUNDARY_INVALID')
 by_key={(x['checkpoint'],x['cohort_record_id']):x for x in tasks};wanted={(x['checkpoint'],x['cohort_record_id']) for x in fresh}
 if set(by_key)!=wanted: raise RuntimeError('MISSING_PROMPT_SET_DIFFERENCE_INVALID')
 generated=[];events=[]
 for checkpoint in sorted({x['checkpoint'] for x in fresh}):
  fresh_rows,fall=_native_generate(checkpoint,[by_key[(x['checkpoint'],x['cohort_record_id'])] for x in fresh if x['checkpoint']==checkpoint],b);generated+=fresh_rows;events+=fall
 if len(imported)!=168 or len(generated)!=48: raise RuntimeError('RAW_COMPLETENESS_INVALID')
 raw=sorted(imported+generated,key=lambda x:(x['checkpoint'],x['cohort_record_id']))
 if len({(x['checkpoint'],x['cohort_record_id']) for x in raw})!=216: raise RuntimeError('RAW_DUPLICATE_OR_MISSING')
 out.mkdir(parents=True);raw_path=out/'RAW_FROZEN.jsonl';_write_rows(raw_path,raw);receipt={'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),'raw_rows':216,'reused_rows':168,'new_rows':48,'fallback_events':events,'optimizer_steps':0,'backward_calls':0,'targets_accessed_before_raw_freeze':False,'gpu_used':True};_write(out/'RAW_FREEZE_RECEIPT.json',receipt);return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cpu-self-test',action='store_true');p.add_argument('--cpu-fixture',type=Path);p.add_argument('--output',type=Path);p.add_argument('--preflight',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--output-root',type=Path);p.add_argument('--binding',type=Path);p.add_argument('--sidecar',type=Path);p.add_argument('--inventory',type=Path);a=p.parse_args()
 if a.preflight:
  b=json.loads(a.binding.read_text()); side=json.loads(a.sidecar.read_text()); inv=json.loads(a.inventory.read_text()); records=validate_sidecar(side,inv)
  if len(records)!=72 or b.get('execution',{}).get('reused_rows')!=168 or b.get('execution',{}).get('new_free_running_rows')!=48: raise RuntimeError('PREFLIGHT_LAYOUT')
  for rel,digest in b['source_hashes'].items():
   if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest: raise RuntimeError('PREFLIGHT_SOURCE_IDENTITY')
  print(json.dumps({'status':'PASS_REAL_INPUT_CPU_PREFLIGHT','sidecar_records':72,'inventory_rows':216,'reused_rows':168,'new_rows':48,'model_loaded':False,'gpu_used':False,'targets_exposed_to_prompt':False},sort_keys=True));raise SystemExit(0)
 if a.cpu_fixture:
  f=json.loads(a.cpu_fixture.read_text()); assert f['rows']==216 and f['reused']==168 and f['missing']==48 and f['ood_new']==0 and f['sidecar_records']==72 and f['fallback']==['CUDA_OOM_ONLY',32,16,8,4,1]; out={'status':'COMPLETE_CPU_FIXTURE_RAW_FROZEN','raw_rows':48,'reused_rows':168,'targets_accessed_before_raw_freeze':False,'optimizer_steps':0,'backward_calls':0,'gpu_used':False}; a.output.write_text(json.dumps(out,sort_keys=True)); print(json.dumps(out,sort_keys=True)); raise SystemExit(0)
 if a.execute:
  if not a.binding or not a.output_root: raise RuntimeError('BINDING_AND_OUTPUT_REQUIRED')
  print(json.dumps(execute(a.binding,a.output_root),sort_keys=True));raise SystemExit(0)
 if not a.cpu_self_test: raise SystemExit('REFUSE_GPU_EXECUTION_WITHOUT_FINAL_BINDING_AND_OWNER_ONE_SHOT_AUTHORIZATION')
 print(json.dumps(self_test(),sort_keys=True))
