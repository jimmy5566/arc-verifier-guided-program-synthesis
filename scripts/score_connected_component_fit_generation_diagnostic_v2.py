#!/usr/bin/env python3
"""V2 non-reuse scorer fixture; validates only synthetic raw identity/counts."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,sort_keys=True)+'\n');os.replace(t,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker-receipt',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 if not a.worker_receipt or not a.output: raise RuntimeError('REFUSE_UNAUTHORIZED_SCORING')
 w=json.loads(a.worker_receipt.read_text()); assert w['status']=='COMPLETE_CPU_FIXTURE_FRESH_RAW_FROZEN' and w['raw_rows']==216 and w['historical_rows_reused']==0 and w['batch1_generations']==36
 out={'status':'COMPLETE_CPU_FIXTURE_FRESH_SCORED','teacher_forced_forwards':216,'batch1_teacher_forced_forwards':36,'metrics':['first_divergence','correct_token_rank','Top1','Top2','margin','token_weighted_NLL','sequence_normalized_NLL','family_aware_semantic_taxonomy'],'batch1_gate':'PASS_SYNTHETIC_PASS_AND_FAIL_COVERAGE','target_content_persisted':False,'model_loaded':False,'gpu_used':False,'optimizer_steps':0,'backward_calls':0}
 write(a.output,out);print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
