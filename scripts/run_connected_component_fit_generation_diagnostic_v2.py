#!/usr/bin/env python3
"""V2 non-reuse worker; CPU fixture mode never imports a model."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,sort_keys=True)+'\n');os.replace(t,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--cpu-fixture',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 if not a.cpu_fixture or not a.output: raise RuntimeError('REFUSE_UNAUTHORIZED_EXECUTION')
 f=json.loads(a.cpu_fixture.read_text()); assert len(f['records'])==72 and len({x['cohort_record_id'] for x in f['records']})==72
 rows=[{'checkpoint':c,'cohort_record_id':x['cohort_record_id'],'prompt_sha256':x['prompt_task_sha256'],'generated_token_ids':[0,15],'parser_valid':True,'effective_batch_size':32,'provenance':'SYNTHETIC_CPU_FIXTURE_NON_REUSE'} for c in f['checkpoints'] for x in f['records']]
 assert len(rows)==216 and len({(x['checkpoint'],x['cohort_record_id']) for x in rows})==216
 out={'status':'COMPLETE_CPU_FIXTURE_FRESH_RAW_FROZEN','raw_rows':216,'historical_rows_reused':0,'batch1_generations':36,'raw_rows_frozen_before_sidecar':True,'model_loaded':False,'gpu_used':False,'optimizer_steps':0,'backward_calls':0}
 write(a.output,out);print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
