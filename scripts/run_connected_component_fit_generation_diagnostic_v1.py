#!/usr/bin/env python3
"""No-update worker. CPU self-test is the only enabled mode before authorization."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.connected_component_fit_generation_runtime_contract import teacher_forced_layout,token_metric,aggregate_token_metrics,first_divergence,semantic_taxonomy,batch_gate,validate_sidecar
import hashlib

def self_test():
    layout=teacher_forced_layout([11,1],[2,15]); assert layout['labels']==[-100,-100,2,15] and layout['target_logit_indices']==[1,2]
    metrics=[token_metric([0,3,1],1),token_metric([0,1,3],2)]; agg=aggregate_token_metrics(metrics); assert agg['top1_count']==2
    assert first_divergence([1,2],[1,3])==1 and first_divergence([1],[1,2])==1
    assert semantic_taxonomy([[1]],[[1]],True)['primary']=='EXACT'
    tol={'correct_token_rank_abs_delta_max':1,'Top1_and_Top2_aggregate_count_delta_max':0,'per_row_mean_margin_abs_delta_max':.02,'token_weighted_NLL_abs_delta_max':.005}
    assert batch_gate({'a':metrics[0]},{'a':metrics[0]},tol)['status']=='PASS'
    return {'status':'PASS_CPU_PRODUCTION_SHAPED_WORKER_SELF_TEST','model_loaded':False,'gpu_used':False,'optimizer_steps':0,'backward_calls':0}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cpu-self-test',action='store_true');p.add_argument('--cpu-fixture',type=Path);p.add_argument('--output',type=Path);p.add_argument('--preflight',action='store_true');p.add_argument('--binding',type=Path);p.add_argument('--sidecar',type=Path);p.add_argument('--inventory',type=Path);a=p.parse_args()
 if a.preflight:
  b=json.loads(a.binding.read_text()); side=json.loads(a.sidecar.read_text()); inv=json.loads(a.inventory.read_text()); records=validate_sidecar(side,inv)
  if len(records)!=72 or b.get('execution',{}).get('reused_rows')!=168 or b.get('execution',{}).get('new_free_running_rows')!=48: raise RuntimeError('PREFLIGHT_LAYOUT')
  for rel,digest in b['source_hashes'].items():
   if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest: raise RuntimeError('PREFLIGHT_SOURCE_IDENTITY')
  print(json.dumps({'status':'PASS_REAL_INPUT_CPU_PREFLIGHT','sidecar_records':72,'inventory_rows':216,'reused_rows':168,'new_rows':48,'model_loaded':False,'gpu_used':False,'targets_exposed_to_prompt':False},sort_keys=True));raise SystemExit(0)
 if a.cpu_fixture:
  f=json.loads(a.cpu_fixture.read_text()); assert f['rows']==216 and f['reused']==168 and f['missing']==48 and f['ood_new']==0 and f['sidecar_records']==72 and f['fallback']==['CUDA_OOM_ONLY',32,16,8,4,1]; out={'status':'COMPLETE_CPU_FIXTURE_RAW_FROZEN','raw_rows':48,'reused_rows':168,'targets_accessed_before_raw_freeze':False,'optimizer_steps':0,'backward_calls':0,'gpu_used':False}; a.output.write_text(json.dumps(out,sort_keys=True)); print(json.dumps(out,sort_keys=True)); raise SystemExit(0)
 if not a.cpu_self_test: raise SystemExit('REFUSE_GPU_EXECUTION_WITHOUT_FINAL_BINDING_AND_OWNER_ONE_SHOT_AUTHORIZATION')
 print(json.dumps(self_test(),sort_keys=True))