#!/usr/bin/env python3
"""CPU injected-model integration of actual V7 production functions."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.v7_component_diagnostic_production_v1 import launch
class G:
 def generate(self,rows,batch_size,max_new_tokens): return [{'token_ids':[i%10,15],'parsed':i%2==0} for i,_ in enumerate(rows)]
class M:
 def teacher_forced(self,record_id,target): return [[0.0]*16 for _ in target]
def main():
 records=[{'cohort_record_id':f'R{i:02d}','prompt_task_sha256':f'p{i:02d}'} for i in range(72)]
 side={x['cohort_record_id']:{'target_token_ids':[1,15]} for x in records}
 out=launch(records,G(),side,M(),{f'R{i:02d}' for i in range(12)})
 assert out['raw']['status']=='RAW_FROZEN' and out['score']['status']=='SCORED' and len(out['score']['records'])==72
 print('PASS_V7_INJECTED_PRODUCTION_PATH_72_RECORDS')
if __name__=='__main__': main()
