#!/usr/bin/env python3
"""No-update worker. CPU self-test is the only enabled mode before authorization."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.connected_component_fit_generation_runtime_contract import teacher_forced_layout,token_metric,aggregate_token_metrics,first_divergence,semantic_taxonomy,batch_gate

def self_test():
    layout=teacher_forced_layout([11,1],[2,15]); assert layout['labels']==[-100,-100,2,15] and layout['target_logit_indices']==[1,2]
    metrics=[token_metric([0,3,1],1),token_metric([0,1,3],2)]; agg=aggregate_token_metrics(metrics); assert agg['top1_count']==2
    assert first_divergence([1,2],[1,3])==1 and first_divergence([1],[1,2])==1
    assert semantic_taxonomy([[1]],[[1]],True)['primary']=='EXACT'
    tol={'correct_token_rank_abs_delta_max':1,'Top1_and_Top2_aggregate_count_delta_max':0,'per_row_mean_margin_abs_delta_max':.02,'token_weighted_NLL_abs_delta_max':.005}
    assert batch_gate({'a':metrics[0]},{'a':metrics[0]},tol)['status']=='PASS'
    return {'status':'PASS_CPU_PRODUCTION_SHAPED_WORKER_SELF_TEST','model_loaded':False,'gpu_used':False,'optimizer_steps':0,'backward_calls':0}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cpu-self-test',action='store_true');a=p.parse_args()
 if not a.cpu_self_test: raise SystemExit('REFUSE_GPU_EXECUTION_WITHOUT_FINAL_BINDING_AND_OWNER_ONE_SHOT_AUTHORIZATION')
 print(json.dumps(self_test(),sort_keys=True))