#!/usr/bin/env python3
"""Target-blind R32 Clean-HF KV lifetime audit; never executes physical B2."""
from __future__ import annotations
import argparse, csv, dataclasses, hashlib, json, os, tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from inference.hf_peft_backend import load_hf_peft_inference
from inference.nvarc_turbodfs_dynamic_ready import _legacy_cache, _cache_batch_size, run_ready_scheduler, start_ready_cell
from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig
from scripts.run_clean_hf_parallel_regret_dfs_v1 import VIEWS, _cache_transform, _prompt_ids, _assert_challenge_only, _verify_frozen_foundation

SNAPS={1,4,8,16,32,64,96,128}
def atomic(path:Path, value:Any)->None:
 path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.',suffix='.tmp')
 try:
  with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(path:Path, values:list[dict[str,Any]])->None:
 keys=sorted({k for x in values for k in x})
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(values)
def cache_meta(cache:Any)->tuple[dict[str,Any],dict[tuple[Any,...],int]]:
 legacy=_legacy_cache(cache); stores={}; logical=0
 for layer in legacy:
  for t in layer:
   logical+=t.numel()*t.element_size(); s=t.untyped_storage(); stores[(str(t.device),s.data_ptr(),s.nbytes())]=s.nbytes()
 return {'sequence_length':int(legacy[0][0].shape[-2]),'layers':len(legacy),'batch_size':_cache_batch_size(legacy),'logical_tensor_bytes':logical},stores
def frames(cell:Any)->tuple[list[dict[str,Any]],list[Any]]:
 out=[];caches=[]; g=cell.generator;d=0
 while g is not None:
  fr=g.gi_frame
  if fr is None:break
  loc=fr.f_locals; cache=loc.get('cache'); outputs=loc.get('outputs')
  item={'cell_key':cell.cell_key,'recursion_depth':d,'generator_id':id(g),'cache_id':id(cache) if cache is not None else None,'cache_type':type(cache).__qualname__ if cache is not None else None,'has_outputs':outputs is not None,'outputs_id':id(outputs) if outputs is not None else None,'outputs_cache_id':id(getattr(outputs,'past_key_values',None)) if outputs is not None else None,'outputs_cache_is_cache':getattr(outputs,'past_key_values',None) is cache}
  if cache is not None: item.update(cache_meta(cache)[0]);caches.append(cache)
  out.append(item);g=g.gi_yieldfrom;d+=1
 return out,caches
def model_bytes(model:Any)->dict[str,int]:
 def total(items):
  seen={}
  for x in items:
   s=x.untyped_storage();seen[(str(x.device),s.data_ptr(),s.nbytes())]=s.nbytes()
  return sum(seen.values())
 params=list(model.parameters()); buffs=list(model.buffers()); return {'model_parameter_bytes':total(params),'model_buffer_bytes':total(buffs)}
def main()->None:
 p=argparse.ArgumentParser();
 for n in ('model_path','adapter_path','challenge','native_config_dir','adapter_foundation','output'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
 p.add_argument('--device',default='cuda:0');a=p.parse_args(); import torch; from arc.io import load_dataset
 _assert_challenge_only(a.challenge); foundation=json.loads(a.adapter_foundation.read_text());
 model,tok,identity=load_hf_peft_inference(model_path=a.model_path,adapter_path=a.adapter_path,device=a.device,native_config_dir=a.native_config_dir,frozen_adapter_identity={'adapter_sha256':foundation['adapter_sha256'],'adapter_config_sha256':foundation['adapter_config_sha256']}); bind=_verify_frozen_foundation(a.adapter_foundation,identity)
 cfg=D1TurboDFSConfig('CUMULATIVE_REGRET_r=4.00',931,1.6094379124341003,None,32,32,frontier_floor=1,diagnostic_trace=False,independent_lane_budgets=True)
 task=load_dataset(a.challenge)['d59b0160']; atomic(a.output/'CONTRACT.json',{'experiment':'CLEAN_HF_KV_LIFETIME_AUDIT_V1','target_blind':True,'gold_loaded':False,'budget':32,'physical_batch':1,'adapter':bind,'config':dataclasses.asdict(cfg)})
 torch.cuda.reset_peak_memory_stats(); persistent=model_bytes(model); waterfall=[];all_frames=[]
 def snapshot(label:str,cells:list[Any]):
  stores={}; cellinfo=[]
  for c in cells:
   fs,cs=frames(c);all_frames.extend(fs); local={};logical=0
   for cache in cs:
    m,ss=cache_meta(cache);logical+=m['logical_tensor_bytes'];local.update(ss);stores.update(ss)
   cellinfo.append({'cell_key':c.cell_key,'frames':len(fs),'cache_references':len(cs),'cache_object_ids':len({id(x) for x in cs}),'unique_kv_bytes':sum(local.values()),'logical_kv_bytes':logical,'expanded_nodes':c.state['expanded_nodes'],'sequence_lengths':json.dumps([x.get('sequence_length') for x in fs])})
  free,total=torch.cuda.mem_get_info(); allocated=torch.cuda.memory_allocated();waterfall.append({'snapshot':label,'allocated_bytes':allocated,'reserved_bytes':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'free_bytes':free,'total_bytes':total,'total_unique_kv_bytes':sum(stores.values()),'unattributed_runtime_bytes':allocated-persistent['model_parameter_bytes']-persistent['model_buffer_bytes']-sum(stores.values()),'cells':json.dumps(cellinfo)})
 snapshot('M0',[]); cells=[]
 for v in VIEWS:
  enc=_prompt_ids(tokenizer=tok,task=task,output_index=0,view=v,device=a.device);cells.append(start_ready_cell(model=model,input_ids=enc,config=cfg,cell_key=f'd59b0160:o0:d24:{v}',normalize_root_cache=True,root_cache_transform=_cache_transform));snapshot('M'+str(len(cells)),cells)
 mutation={}
 def observe(stage:str,data:dict[str,Any]):
  if stage=='before_forward' and not mutation:
   r=data['requests'][0];m,s=cache_meta(r.cache);mutation.update({'input_cache_id':id(r.cache),'input_sequence_length':m['sequence_length'],'input_storage_keys':[list(k) for k in s]})
  if stage=='after_model_forward' and mutation and 'output_cache_id' not in mutation:
   o=data['outputs_by_cell'][0].past_key_values;m,s=cache_meta(o);mutation.update({'output_cache_id':id(o),'output_sequence_length':m['sequence_length'],'same_object':id(o)==mutation['input_cache_id'],'output_storage_keys':[list(k) for k in s]})
  if stage=='after_model_forward':
   n=data['physical_forward_index'];
   if n in SNAPS:snapshot('F'+str(n),cells)
 sched=run_ready_scheduler(model=model,cells=cells,dynamic_batch2=False,scheduling_policy='round_robin',observer=observe);snapshot('END',cells)
 mutation['classification']='IN_PLACE_CACHE_UPDATE' if mutation.get('same_object') else 'NEW_CACHE_OBJECT_SHARED_STORAGE' if set(map(tuple,mutation.get('input_storage_keys',[]))) & set(map(tuple,mutation.get('output_storage_keys',[]))) else 'NEW_CACHE_OBJECT_NEW_STORAGE'
 atomic(a.output/'DYNAMICCACHE_UPDATE_SEMANTICS.json',mutation); rows(a.output/'KV_MEMORY_WATERFALL.csv',waterfall);rows(a.output/'GPU_MEMORY_ATTRIBUTION.csv',[{**persistent,**{k:v for k,v in x.items() if k!='cells'}} for x in waterfall]);
 # END is intentionally cache-empty.  Project from the maximum simultaneous
 # four-cell snapshot instead, otherwise the published B2 estimate is zero.
 snapshots_with_cells=[row for row in waterfall if len(json.loads(row['cells']))==4]
 projection_source=max(snapshots_with_cells,key=lambda row:int(row['total_unique_kv_bytes']))
 source_cells=json.loads(projection_source['cells']);four=sum(x['unique_kv_bytes'] for x in source_cells);per_view=max([x['unique_kv_bytes'] for x in source_cells],default=0)
 merged=per_view*2;split=per_view*2;projection={'projection_source_snapshot':projection_source['snapshot'],'four_current_kv_bytes':four,'estimated_b2_merged_kv_bytes':merged,'estimated_b2_split_kv_bytes':split,'projected_peak_expected_bytes':persistent['model_parameter_bytes']+persistent['model_buffer_bytes']+four+merged+split,'total_gpu_bytes':int(projection_source['total_bytes']),'fits_24gb':persistent['model_parameter_bytes']+persistent['model_buffer_bytes']+four+merged+split<=int(projection_source['total_bytes']),'corrected_end_state_projection_bug':True};atomic(a.output/'FOUR_VIEW_B2_MEMORY_PROJECTION.json',projection)
 (a.output/'STATIC_KV_LIFETIME.md').write_text('Suspended _ready_dfs generator frames retain local cache, outputs, candidates, and child via gi_yieldfrom; physical duplication is measured separately.\n')
 atomic(a.output/'DECISION.json',{'status':'PASS','target_blind':True,'scheduler':sched,'mutation_class':mutation['classification'],'persistent':persistent,'final_cells':final});
 (a.output/'REPORT.md').write_text('# CLEAN_HF_KV_LIFETIME_AUDIT_V1\nTarget-blind R32 B1-only audit complete.\n'); hashes={x.name:sha(x) for x in a.output.iterdir() if x.is_file()};atomic(a.output/'HASHES.json',hashes)
if __name__=='__main__':main()
