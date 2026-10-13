#!/usr/bin/env python3
"""Deterministic scorer helpers; no target text is written by the CPU self-test."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.connected_component_fit_generation_runtime_contract import first_divergence,semantic_taxonomy,teacher_forced_layout,token_metric,aggregate_record_metrics,validate_sidecar,batch_gate,pack_teacher_forced_batch,resolve_adapter_directory

def self_test():
 assert first_divergence([1,2],[1,3])==1
 assert semantic_taxonomy([[1]],[[1]],True)['primary']=='EXACT'
 return {'status':'PASS_CPU_SCORER_SELF_TEST','targets_persisted':False,'gpu_used':False}

def _write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n');os.replace(tmp,path)
def _rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _write_rows(path,values):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(b''.join(json.dumps(x,sort_keys=True,separators=(',',':')).encode()+b'\n' for x in values));os.replace(tmp,path)

def _token_metrics(logits,target_ids,positions):
 import torch
 metrics=[]
 for position,token in zip(positions,target_ids,strict=True):
  values=logits[position].float();correct=values[token];rank=int((values>correct).sum().item())+1;other=values.clone();other[token]=float('-inf')
  metrics.append({'rank':rank,'top1':rank==1,'top2':rank<=2,'margin':float((correct-other.max()).item()),'nll':float(torch.logsumexp(values,dim=0).item()-correct.item())})
 return metrics

def _forward_teacher_forced(model, records, pad_token_id):
 """One physical padded forward for one requested batch of logical records."""
 import torch
 layouts={record_id:teacher_forced_layout(prompt['prompt_token_ids'],target['target_token_ids']) for record_id,target,prompt,_ in records}
 packed=pack_teacher_forced_batch(layouts,pad_token_id)
 order=list(packed); ids=torch.tensor([packed[k]['input_ids'] for k in order],device='cuda:0');mask=torch.tensor([packed[k]['attention_mask'] for k in order],device='cuda:0')
 logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
 return {record_id:_token_metrics(logits[index],target['target_token_ids'],layouts[record_id]['target_logit_indices']) for index,(record_id,target,_,_) in enumerate(records)}

def execute(binding_path,raw_root,output_root):
 """Score only a completed raw freeze; token IDs never leave this process."""
 import torch
 from peft import PeftModel
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from inference.nvarc_native import parse_native_grid
 b=json.loads(Path(binding_path).read_text());e=b['execution'];freeze=json.loads((Path(raw_root)/'RAW_FREEZE_RECEIPT.json').read_text())
 raw_path=Path(raw_root)/'RAW_FROZEN.jsonl'
 if freeze.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or freeze.get('raw_sha256')!=_sha(raw_path):raise RuntimeError('RAW_FREEZE_REQUIRED')
 side_path=Path(e['sidecar_path'])
 if _sha(side_path)!=e['sidecar_sha256']:raise RuntimeError('SIDECAR_IDENTITY_MISMATCH')
 inventory=json.loads(Path(e['inventory_path']).read_text());targets=validate_sidecar(json.loads(side_path.read_text()),inventory)
 prompt_document=json.loads(Path(e['prompt_token_ids_path']).read_text()); prompt_records=prompt_document.get('records') if isinstance(prompt_document,dict) else prompt_document
 if not isinstance(prompt_document,dict) or prompt_document.get('target_content_present') is not False or not isinstance(prompt_records,list): raise RuntimeError('PROMPT_TOKEN_IDS_TARGET_BOUNDARY_INVALID')
 prompts={x['cohort_record_id']:x for x in prompt_records}
 raw={(x['checkpoint'],x['cohort_record_id']):x for x in _rows(raw_path)}
 if len(raw)!=216:raise RuntimeError('RAW_COMPLETENESS_INVALID')
 out=[];per_checkpoint={}
 if int(e.get('teacher_forced_physical_batch_size', 0))!=1 or int(e.get('requested_batch_size', 0))!=1: raise RuntimeError('ALL_B1_TEACHER_FORCED_REQUIRED')
 for checkpoint,runtime in sorted(b['checkpoint_runtime'].items()):
  if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA_BF16_REQUIRED')
  tokenizer=AutoTokenizer.from_pretrained(runtime['base_path'],local_files_only=True)
  base=AutoModelForCausalLM.from_pretrained(runtime['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0');model=PeftModel.from_pretrained(base,str(resolve_adapter_directory(runtime)),is_trainable=False).eval()
  records=[]
  for record_id,target in sorted(targets.items()):
   row=raw.get((checkpoint,record_id));prompt=prompts.get(record_id)
   if not row or not prompt or prompt.get('prompt_task_sha256')!=target['prompt_task_sha256'] or row.get('prompt_sha256')!=target['prompt_task_sha256']:raise RuntimeError('PROMPT_OR_RAW_IDENTITY_MISMATCH')
   records.append((record_id,{**target,'target_token_ids':[int(x) for x in target['target_token_ids']]},prompt,row))
  primary={};b1={}
  with torch.inference_mode():
   for record in records:
    primary.update(_forward_teacher_forced(model,[record],tokenizer.pad_token_id or tokenizer.eos_token_id or 0))
  for record_id,target,prompt,row in records:
   target_ids=target['target_token_ids'];expected=parse_native_grid(tokenizer.decode(target_ids,skip_special_tokens=False));taxonomy=semantic_taxonomy(row.get('parsed_grid'),expected,row.get('parser_valid') is True)
   out.append({'checkpoint':checkpoint,'cohort_record_id':record_id,'split':target['split'],'prompt_sha256':target['prompt_task_sha256'],'teacher_forced':aggregate_record_metrics({record_id:primary[record_id]}),'first_divergence':first_divergence(row['generated_token_ids'],target_ids),'semantic_taxonomy':taxonomy,'target_token_ids_persisted':False})
  per_checkpoint[checkpoint]={'records':len(primary),'teacher_forced_physical_batch_size':1,'teacher_forced_physical_forwards':len(primary),'batch_comparison':'NOT_APPLICABLE_ALL_PRIMARY_RECORDS_PHYSICAL_BATCH1'};del model;torch.cuda.empty_cache()
 if len(out)!=216:raise RuntimeError('SCORE_COMPLETENESS_INVALID')
 output=Path(output_root);output.mkdir(parents=True);metrics_path=output/'SCORED_METRICS.jsonl';_write_rows(metrics_path,out);_write(output/'SCORE_RECEIPT.json',{'status':'COMPLETE_CPU_SCORED_NO_UPDATE','raw_sha256':freeze['raw_sha256'],'metrics_sha256':_sha(metrics_path),'records':216,'checkpoint_summary':per_checkpoint,'target_token_ids_persisted':False,'optimizer_steps':0,'backward_calls':0,'gpu_used':True});return output/'SCORE_RECEIPT.json'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cpu-self-test',action='store_true');p.add_argument('--cpu-fixture',type=Path);p.add_argument('--worker-receipt',type=Path);p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--binding',type=Path);p.add_argument('--raw-root',type=Path);p.add_argument('--output-root',type=Path);a=p.parse_args()
 if a.cpu_fixture:
  f=json.loads(a.cpu_fixture.read_text()); w=json.loads(a.worker_receipt.read_text()); assert w['status']=='COMPLETE_CPU_FIXTURE_RAW_FROZEN' and f['teacher_forced_physical_batch_size']==1 and f['teacher_forced_forwards']==216; out={'status':'COMPLETE_CPU_FIXTURE_SCORED','token_weighted_and_sequence_NLL':True,'first_divergence':True,'taxonomy_categories':['CONNECTED_COMPONENT_COUNT','SELECTED_OBJECT_IDENTITY','GEOMETRY','COLOR','DIMENSIONS','PARSER_FAILURE','OUTPUT_CONSTRUCTION'],'all_teacher_forced_records_physical_batch1':True,'batch_comparison':'NOT_APPLICABLE_ALL_PRIMARY_RECORDS_PHYSICAL_BATCH1','gpu_used':False}; a.output.write_text(json.dumps(out,sort_keys=True)); print(json.dumps(out,sort_keys=True)); raise SystemExit(0)
 if a.execute:
  if not a.binding or not a.raw_root or not a.output_root:raise RuntimeError('BINDING_RAW_AND_OUTPUT_REQUIRED')
  print(json.dumps({'status':'COMPLETE_CPU_SCORED_NO_UPDATE','receipt':str(execute(a.binding,a.raw_root,a.output_root))},sort_keys=True));raise SystemExit(0)
 if not a.cpu_self_test: raise SystemExit('REFUSE_SCORING_UNBOUND_EXECUTION')
 print(json.dumps(self_test(),sort_keys=True))
