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
 p=argparse.ArgumentParser();p.add_argument('--cpu-self-test',action='store_true');a=p.parse_args()
 if not a.cpu_self_test: raise SystemExit('REFUSE_SCORING_UNBOUND_EXECUTION')
 print(json.dumps(self_test(),sort_keys=True))