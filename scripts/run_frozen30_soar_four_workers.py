"""Frozen30 SOAR induction using four independent single-L4 workers."""
from __future__ import annotations
import argparse,json,multiprocessing as mp,os,sys,hashlib
from pathlib import Path
from time import perf_counter,time
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from arc.io import load_dataset,load_solutions
from inference.dual_reasoning_smoke import soar_prompt,extract_program,verify_program,execute_program
from run_frozen30_native_soar_complementarity import native_labels, write_json
from run_soar_single_gpu_parallel_smoke import model_path, load_single, generate, K as PROGRAM_BUDGET, SEED_BASE

def atomic(path:Path,value:dict)->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value),encoding='utf-8');os.replace(tmp,path)
def valid(path:Path,config_hash:str)->bool:
 try:
  d=json.loads(path.read_text());return d.get('config_hash')==config_hash and len(d.get('candidate_programs',[]))==PROGRAM_BUDGET
 except Exception:return False
def worker(wid:int,gpu:int,model_s:str,items:list[tuple[str,list]],out_s:str,config_hash:str)->None:
 os.environ['CUDA_VISIBLE_DEVICES']=str(gpu); started=time();model,tok=load_single(Path(model_s));out=Path(out_s);out.mkdir(parents=True,exist_ok=True)
 for local,(tid,pairs) in enumerate(items):
  target=out/'tasks'/f'{tid}.json'
  if valid(target,config_hash): print(json.dumps({'event':'SOAR_4W_RESUMED','worker_id':wid,'task_id':tid}),flush=True);continue
  rows=[(x['input'],x['output']) for x in pairs];prompt=soar_prompt(rows); cs=[]
  for ci in range(PROGRAM_BUDGET):
   raw=generate(model,tok,prompt,SEED_BASE+(wid*10000)+(local*PROGRAM_BUDGET)+ci);cs.append({'candidate_index':ci,'raw_model_output':raw,'extracted_code':extract_program(raw)})
  atomic(target,{'task_id':tid,'worker_id':wid,'gpu_id':gpu,'model_path':model_s,'config_hash':config_hash,'candidate_programs':cs,'completed_epoch':time()});print(json.dumps({'event':'SOAR_4W_GENERATED','worker_id':wid,'task_id':tid,'count':PROGRAM_BUDGET}),flush=True)
 write_json(out/f'worker_{wid}.json',{'worker_id':wid,'gpu_id':gpu,'started_epoch':started,'completed_epoch':time(),'task_count':len(items)})

def startup_worker(wid:int,gpu:int,model_s:str,out_s:str)->None:
 os.environ['CUDA_VISIBLE_DEVICES']=str(gpu); started=time(); model,tok=load_single(Path(model_s)); write_json(Path(out_s)/f'worker_{wid}.json',{'worker_id':wid,'gpu_id':gpu,'started_epoch':started,'ready_epoch':time(),'status':'MODEL_READY','visible_devices':os.environ['CUDA_VISIBLE_DEVICES']}); del model,tok

def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--challenge-path',type=Path,required=True);p.add_argument('--solutions-path',type=Path,required=True);p.add_argument('--native-frozen',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--startup-smoke',action='store_true');p.add_argument('--checkpoint-smoke',action='store_true');a=p.parse_args();started=perf_counter()
 native=json.loads(a.native_frozen.read_text());ids=list(native['records']);tasks=load_dataset(a.challenge_path);timeout=float(json.loads(a.calibration.read_text())['timeout_policy']['selected_timeout_seconds']);path=model_path(a.input_root); labels=native_labels(native,load_solutions(a.solutions_path));assert sum(x['native_failure_class']=='SELECTION_MISS' for x in labels.values())==12 and sum(x['native_failure_class']=='GENERATION_MISS' for x in labels.values())==9
 rawdir=a.output_root/'generation';ps=[]
 if a.startup_smoke:
  for w in range(4):
   x=mp.get_context('spawn').Process(target=startup_worker,args=(w,w,str(path),str(rawdir)));x.start();ps.append(x)
  for x in ps:x.join()
  if any(x.exitcode for x in ps):raise RuntimeError(f'worker exits={[x.exitcode for x in ps]}')
  write_json(a.output_root/'four_worker_startup.json',{'status':'PASS','workers':[json.loads((rawdir/f'worker_{i}.json').read_text()) for i in range(4)]});return
 if a.checkpoint_smoke: ids=ids[:2]
 payload=[(tid,[{'input':e.input.to_list(),'output':e.output.to_list()} for e in tasks[tid].train]) for tid in ids]; shards=[payload[i::4] for i in range(4)];config_hash=hashlib.sha256(json.dumps({'K':PROGRAM_BUDGET,'seed':SEED_BASE,'prompt':'soar_v1','sampling':[.2,.95,768]},sort_keys=True).encode()).hexdigest()
 for w,shard in enumerate(shards):
  x=mp.get_context('spawn').Process(target=worker,args=(w,w,str(path),shard,str(rawdir),config_hash));x.start();ps.append(x)
 for x in ps:x.join()
 if any(x.exitcode for x in ps):raise RuntimeError(f'worker exits={[x.exitcode for x in ps]}')
 files=list((rawdir/'tasks').glob('*.json'));assert len(files)==len(ids) and all(valid(rawdir/'tasks'/f'{tid}.json',config_hash) for tid in ids);write_json(a.output_root/'generation_manifest.json',{'task_ids':ids,'config_hash':config_hash,'task_file_count':len(files),'total_candidates':len(files)*PROGRAM_BUDGET});
 if a.checkpoint_smoke:return
 records={tid:json.loads((rawdir/'tasks'/f'{tid}.json').read_text()) for tid in ids}
 for tid in ids:
  pairs=[(e.input.to_list(),e.output.to_list()) for e in tasks[tid].train]
  for c in records[tid]['candidate_programs']:
   v=verify_program(c['extracted_code'],pairs,timeout_seconds=timeout) if c['extracted_code'] else {'all_train_exact':False,'train_execution':[]};c['verification']=v;c['all_train_exact']=v['all_train_exact']
  hit=next((c for c in records[tid]['candidate_programs'] if c['all_train_exact']),None); records[tid].update({'train_exact_available':hit is not None,'selected_candidate_index':hit['candidate_index'] if hit else None,'selection_rule':'lowest_candidate_index_among_all_train_exact','frozen_test_prediction':execute_program(hit['extracted_code'],tasks[tid].test[0].input.to_list(),timeout_seconds=timeout).get('grid') if hit else None})
 soar={'status':'SOAR_FROZEN30_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING','records':records,'program_budget':PROGRAM_BUDGET,'sandbox_timeout_seconds':timeout};write_json(a.output_root/'soar_frozen30.json',soar)
 sol=load_solutions(a.solutions_path);lab=native_labels(native,sol);rows=[]
 for tid in ids:
  se=records[tid]['frozen_test_prediction']==sol[tid] if records[tid]['frozen_test_prediction'] is not None else False;n=lab[tid];o='BOTH_SOLVED' if n['native_top1_correct'] and se else 'NATIVE_ONLY' if n['native_top1_correct'] else 'SOAR_ONLY' if se else 'NEITHER';rows.append({'task_id':tid,**n,'soar_train_exact_available':records[tid]['train_exact_available'],'soar_final_test_exact':se,'overlap':o})
 overlap={k:sum(x['overlap']==k for x in rows) for k in ('BOTH_SOLVED','NATIVE_ONLY','SOAR_ONLY','NEITHER')};rescue={k:{'native_task_count':sum(x['native_failure_class']==k for x in rows),'soar_rescues':sum(x['native_failure_class']==k and x['soar_final_test_exact'] for x in rows)} for k in ('SELECTION_MISS','GENERATION_MISS')};write_json(a.output_root/'native_soar_comparison.json',{'tasks':rows,'union_native_top1_or_soar':sum(x['native_top1_correct'] or x['soar_final_test_exact'] for x in rows)});write_json(a.output_root/'overlap_summary.json',overlap);write_json(a.output_root/'failure_rescue_matrix.json',rescue);write_json(a.output_root/'runtime_summary.json',{'wall_seconds':perf_counter()-started,'workers':[json.loads((rawdir/f'worker_{i}.json').read_text()) for i in range(4)]});print(json.dumps({'event':'FROZEN30_4W_COMPLETE','overlap':overlap,'rescue':rescue}),flush=True)
if __name__=='__main__':main()
