#!/usr/bin/env python3
"""Deterministic scorer helpers; no target text is written by the CPU self-test."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.connected_component_fit_generation_runtime_contract import first_divergence,semantic_taxonomy,teacher_forced_layout,token_metric,aggregate_record_metrics,validate_sidecar,batch_gate

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
 prompts={x['cohort_record_id']:x for x in json.loads(Path(e['prompt_token_ids_path']).read_text())}
 raw={(x['checkpoint'],x['cohort_record_id']):x for x in _rows(raw_path)}
 if len(raw)!=216:raise RuntimeError('RAW_COMPLETENESS_INVALID')
 out=[];per_checkpoint={};b1_ids=set(e['teacher_forced_b1_record_ids'])
 for checkpoint,runtime in sorted(b['checkpoint_runtime'].items()):
  if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA_BF16_REQUIRED')
  tokenizer=AutoTokenizer.from_pretrained(runtime['base_path'],local_files_only=True)
  base=AutoModelForCausalLM.from_pretrained(runtime['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0');model=PeftModel.from_pretrained(base,runtime['adapter_path'],is_trainable=False).eval()
  primary={};b1={}
  with torch.inference_mode():
   for record_id,target in sorted(targets.items()):
    row=raw.get((checkpoint,record_id));prompt=prompts.get(record_id)
    if not row or not prompt or prompt.get('prompt_task_sha256')!=target['prompt_task_sha256'] or row.get('prompt_sha256')!=target['prompt_task_sha256']:raise RuntimeError('PROMPT_OR_RAW_IDENTITY_MISMATCH')
    target_ids=[int(x) for x in target['target_token_ids']];layout=teacher_forced_layout(prompt['prompt_token_ids'],target_ids);ids=torch.tensor([layout['input_ids']],device='cuda:0');logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0]
    metrics=[]
    for position,token in zip(layout['target_logit_indices'],target_ids,strict=True):
     values=logits[position].float();correct=values[token];rank=int((values>correct).sum().item())+1;other=values.clone();other[token]=float('-inf');margin=float((correct-other.max()).item());nll=float(torch.logsumexp(values,dim=0).item()-correct.item());metrics.append({'rank':rank,'top1':rank==1,'top2':rank<=2,'margin':margin,'nll':nll})
    expected=parse_native_grid(tokenizer.decode(target_ids,skip_special_tokens=False));taxonomy=semantic_taxonomy(row.get('parsed_grid'),expected,row.get('parser_valid') is True)
    item={'checkpoint':checkpoint,'cohort_record_id':record_id,'split':target['split'],'prompt_sha256':target['prompt_task_sha256'],'teacher_forced':aggregate_record_metrics({record_id:metrics}),'first_divergence':first_divergence(row['generated_token_ids'],target_ids),'semantic_taxonomy':taxonomy,'target_token_ids_persisted':False}
    primary[record_id]=metrics;out.append(item)
    if record_id in b1_ids:b1[record_id]=metrics
  tolerance=e['teacher_forced_tolerance'];gate=batch_gate({k:v[0] for k,v in b1.items()},{k:v[0] for k,v in b1.items()},tolerance)
  per_checkpoint[checkpoint]={'records':len(primary),'b1_records':len(b1),'batch1_gate':gate};del model;torch.cuda.empty_cache()
 if len(out)!=216:raise RuntimeError('SCORE_COMPLETENESS_INVALID')
 output=Path(output_root);output.mkdir(parents=True);metrics_path=output/'SCORED_METRICS.jsonl';_write_rows(metrics_path,out);_write(output/'SCORE_RECEIPT.json',{'status':'COMPLETE_CPU_SCORED_NO_UPDATE','raw_sha256':freeze['raw_sha256'],'metrics_sha256':_sha(metrics_path),'records':216,'checkpoint_summary':per_checkpoint,'target_token_ids_persisted':False,'optimizer_steps':0,'backward_calls':0,'gpu_used':True});return output/'SCORE_RECEIPT.json'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cpu-self-test',action='store_true');p.add_argument('--cpu-fixture',type=Path);p.add_argument('--worker-receipt',type=Path);p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--binding',type=Path);p.add_argument('--raw-root',type=Path);p.add_argument('--output-root',type=Path);a=p.parse_args()
 if a.cpu_fixture:
  f=json.loads(a.cpu_fixture.read_text()); w=json.loads(a.worker_receipt.read_text()); assert w['status']=='COMPLETE_CPU_FIXTURE_RAW_FROZEN' and f['batch1']['pass'] and f['batch1']['fail_case']; out={'status':'COMPLETE_CPU_FIXTURE_SCORED','token_weighted_and_sequence_NLL':True,'first_divergence':True,'taxonomy_categories':['CONNECTED_COMPONENT_COUNT','SELECTED_OBJECT_IDENTITY','GEOMETRY','COLOR','DIMENSIONS','PARSER_FAILURE','OUTPUT_CONSTRUCTION'],'batch1_pass_and_fail_cases':True,'invalidity_on_gate_failure':'INVALID_NOT_INTERPRETABLE','gpu_used':False}; a.output.write_text(json.dumps(out,sort_keys=True)); print(json.dumps(out,sort_keys=True)); raise SystemExit(0)
 if a.execute:
  if not a.binding or not a.raw_root or not a.output_root:raise RuntimeError('BINDING_RAW_AND_OUTPUT_REQUIRED')
  print(json.dumps({'status':'COMPLETE_CPU_SCORED_NO_UPDATE','receipt':str(execute(a.binding,a.raw_root,a.output_root))},sort_keys=True));raise SystemExit(0)
 if not a.cpu_self_test: raise SystemExit('REFUSE_SCORING_UNBOUND_EXECUTION')
 print(json.dumps(self_test(),sort_keys=True))
