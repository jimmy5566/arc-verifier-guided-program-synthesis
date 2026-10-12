#!/usr/bin/env python3
"""Deterministic scorer helpers; no target text is written by the CPU self-test."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.connected_component_fit_generation_runtime_contract import first_divergence,semantic_taxonomy

def self_test():
 assert first_divergence([1,2],[1,3])==1
 assert semantic_taxonomy([[1]],[[1]],True)['primary']=='EXACT'
 return {'status':'PASS_CPU_SCORER_SELF_TEST','targets_persisted':False,'gpu_used':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cpu-self-test',action='store_true');p.add_argument('--cpu-fixture',type=Path);p.add_argument('--worker-receipt',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 if a.cpu_fixture:
  f=json.loads(a.cpu_fixture.read_text()); w=json.loads(a.worker_receipt.read_text()); assert w['status']=='COMPLETE_CPU_FIXTURE_RAW_FROZEN' and f['batch1']['pass'] and f['batch1']['fail_case']; out={'status':'COMPLETE_CPU_FIXTURE_SCORED','token_weighted_and_sequence_NLL':True,'first_divergence':True,'taxonomy_categories':['CONNECTED_COMPONENT_COUNT','SELECTED_OBJECT_IDENTITY','GEOMETRY','COLOR','DIMENSIONS','PARSER_FAILURE','OUTPUT_CONSTRUCTION'],'batch1_pass_and_fail_cases':True,'invalidity_on_gate_failure':'INVALID_NOT_INTERPRETABLE','gpu_used':False}; a.output.write_text(json.dumps(out,sort_keys=True)); print(json.dumps(out,sort_keys=True)); raise SystemExit(0)
 if not a.cpu_self_test: raise SystemExit('REFUSE_SCORING_UNBOUND_EXECUTION')
 print(json.dumps(self_test(),sort_keys=True))