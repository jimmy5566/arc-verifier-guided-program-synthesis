#!/usr/bin/env python3
"""Fresh-process, target-blind audit through the first Clean-HF R128 B2 pack."""
from __future__ import annotations
import argparse, dataclasses, json, os, tempfile, hashlib
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
import sys;sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from inference.hf_peft_backend import load_hf_peft_inference
from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig
from inference.nvarc_turbodfs_dynamic_ready import run_ready_scheduler
from scripts.run_clean_hf_parallel_regret_dfs_v1 import VIEWS,_prompt_ids,_cache_transform,_assert_challenge_only,_verify_frozen_foundation
from scripts.run_clean_hf_kv_lifetime_audit_v1 import atomic,sha,rows,frames,model_bytes,cache_meta

class StopAfterFirstPack(Exception): pass
def main():
 p=argparse.ArgumentParser()
 for n in ('model_path','adapter_path','challenge','native_config_dir','adapter_foundation','output'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
 p.add_argument('--device',default='cuda:0');a=p.parse_args();import torch;from arc.io import load_dataset
 _assert_challenge_only(a.challenge);a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();
 foundation=json.loads(a.adapter_foundation.read_text()); atomic(a.output/'CONTRACT.json',{'experiment':'CLEAN_HF_R128_PRE_B2_MEMORY_AUDIT_V1','pid':pid,'target_blind':True,'gold_loaded':False,'budget':128,'source':'fresh_process'})
 waterfall=[];pack=[]; alias={}
 def memory(label,cells,extra=None):
  stores={};per=[]
  for c in cells:
   fs,cs=frames(c);local={}
   for ca in cs: local.update(cache_meta(ca)[1])
   stores.update(local);per.append({'cell_key':c.cell_key,'position':c.request.position if c.request else None,'recursion_depth':len(fs),'cache_object_count':len({id(x) for x in cs}),'unique_kv_bytes':sum(local.values())})
  free,total=torch.cuda.mem_get_info();waterfall.append({'checkpoint':label,'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'free_bytes':free,'total_bytes':total,'total_live_kv_bytes':sum(stores.values()),'cells':json.dumps(per),'extra':json.dumps(extra or {})})
 torch.cuda.reset_peak_memory_stats();memory('P0',[])
 model,tok,identity=load_hf_peft_inference(model_path=a.model_path,adapter_path=a.adapter_path,device=a.device,native_config_dir=a.native_config_dir,frozen_adapter_identity={'adapter_sha256':foundation['adapter_sha256'],'adapter_config_sha256':foundation['adapter_config_sha256']});bind=_verify_frozen_foundation(a.adapter_foundation,identity);persistent=model_bytes(model);memory('P1',[],persistent)
 cfg=D1TurboDFSConfig('CUMULATIVE_REGRET_r=4.00',931,1.6094379124341003,None,128,32,frontier_floor=1,diagnostic_trace=False,independent_lane_budgets=True);task=load_dataset(a.challenge)['d59b0160'];cells=[]
 for i,v in enumerate(VIEWS,1):
  from inference.nvarc_turbodfs_dynamic_ready import start_ready_cell
  enc=_prompt_ids(tokenizer=tok,task=task,output_index=0,view=v,device=a.device);cells.append(start_ready_cell(model=model,input_ids=enc,config=cfg,cell_key=f'd59b0160:o0:d24:{v}',normalize_root_cache=True,root_cache_transform=_cache_transform));memory(f'P{i+1}',cells)
 def obs(stage,data):
  if stage=='before_forward' and data['physical_batch']==2: memory('P6',cells,{'selected':[x.cell_key for x in data['selected_cells']]})
 def packobs(stage,data):
  memory({'legacy_views':'B1','packed_layer':'B2','merged_legacy':'B3','before_restore':'B3','after_restore':'B4','after_split':'B7'}.get(stage,stage),cells,{'stage':stage,'layer':data.get('layer_index')})
  if stage=='after_restore':
   legacy=data['merged_legacy'];converted=data['merged_cache']; l=set(cache_meta(legacy)[1]);c=set(cache_meta(converted)[1]);alias.update({'from_legacy_cache':'ZERO_COPY_WRAPPER' if l==c else 'PARTIAL_COPY' if l&c else 'FULL_COPY','legacy_storage_count':len(l),'converted_storage_count':len(c),'shared_storage_count':len(l&c)})
   raise StopAfterFirstPack()
 try:
  run_ready_scheduler(model=model,cells=cells,dynamic_batch2=True,scheduling_policy='dynamic_ready',observer=obs,cache_pack_observer=packobs)
 except StopAfterFirstPack: pass
 rows(a.output/'PRE_B2_MEMORY_WATERFALL.csv',waterfall);rows(a.output/'B2_PACK_MEMORY_WATERFALL.csv',[x for x in waterfall if x['checkpoint'].startswith('B')]);atomic(a.output/'CACHE_CONVERSION_STORAGE_ALIAS.json',alias)
 p6=next((x for x in waterfall if x['checkpoint']=='P6'),waterfall[-1]);unat=int(p6['allocated_bytes'])-persistent['model_parameter_bytes']-persistent['model_buffer_bytes']-int(p6['total_live_kv_bytes']);root={'pid':pid,'persistent':persistent,'p6_unattributed_bytes':unat,'classification':'B2_PACK_TRANSIENT_MULTIPLIER' if any(x['checkpoint'].startswith('B') for x in waterfall) else 'R128_INTRINSIC_STATE_GROWTH','adapter':bind,'alias':alias};atomic(a.output/'ROOT_CAUSE.json',root);(a.output/'PRE_B2_LIVE_CUDA_STORAGE.csv').write_text('scope,bytes\nmodel_and_adapter,%d\nlive_kv,%d\nunattributed,%d\n'%(persistent['model_parameter_bytes']+persistent['model_buffer_bytes'],int(p6['total_live_kv_bytes']),unat));(a.output/'REPORT.md').write_text('# CLEAN_HF_R128_PRE_B2_MEMORY_AUDIT_V1\nFresh-process target-blind audit stopped after first DynamicCache conversion, before B2 model forward.\n');hashes={x.name:sha(x) for x in a.output.iterdir() if x.is_file()};atomic(a.output/'HASHES.json',hashes)
if __name__=='__main__':main()
