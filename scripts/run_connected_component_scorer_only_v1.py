#!/usr/bin/env python3
"""Scorer-only worker: never imports or invokes the generation worker."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n');os.replace(t,p)
def execute(binding_path,raw_root,output_root):
 b=json.loads(Path(binding_path).read_text());e=b['execution']
 if int(e.get('free_running_generations',-1))!=0 or int(e.get('raw_rows_created',-1))!=0: raise RuntimeError('SCORER_ONLY_GENERATION_FORBIDDEN')
 from scripts.score_connected_component_fit_generation_diagnostic_v1 import execute as score_execute
 try:
  receipt=score_execute(binding_path,raw_root,output_root);score=json.loads(Path(receipt).read_text())
 except Exception as exc:
  write(Path(output_root)/'TERMINAL_RECEIPT.json',{'status':'FAILED_SCORER_ONLY','error_type':type(exc).__name__,'raw_sha256':e['immutable_raw_sha256'],'optimizer_steps':0,'backward_calls':0,'free_running_generations':0});raise
 terminal={'status':'COMPLETE_SCORER_ONLY_ALL_B1_NO_UPDATE','raw_sha256':e['immutable_raw_sha256'],'score_receipt_sha256':sha(receipt),'metrics_sha256':score['metrics_sha256'],'records':score['records'],'teacher_forced_physical_batch_size':1,'teacher_forced_forwards':216,'free_running_generations':0,'optimizer_steps':0,'backward_calls':0,'gpu_used':True}
 write(Path(output_root)/'TERMINAL_RECEIPT.json',terminal);return terminal
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--binding',type=Path);p.add_argument('--raw-root',type=Path);p.add_argument('--output-root',type=Path);a=p.parse_args()
 if not a.execute or not a.binding or not a.raw_root or not a.output_root: raise RuntimeError('SCORER_ONLY_ARGUMENTS_REQUIRED')
 print(json.dumps(execute(a.binding,a.raw_root,a.output_root),sort_keys=True))
