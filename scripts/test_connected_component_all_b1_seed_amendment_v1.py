#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.freeze_connected_component_all_b1_seed_amendment_v1 import build_seed_manifest,SEED_BASE
p=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/connected_component_count_fit_vs_generation_diagnostic_v1/MISSING_TRAIN_FIT_PROMPT_TASKS_V1.json'
rows=json.loads(p.read_text())['records']; m=build_seed_manifest(rows)
assert len(m['records'])==48 and len({(x['checkpoint'],x['cohort_record_id']) for x in m['records']})==48
for checkpoint in sorted({x['checkpoint'] for x in rows}):
 got=[x for x in m['records'] if x['checkpoint']==checkpoint]
 assert [x['ordinal_within_checkpoint'] for x in got]==list(range(len(got)))
 assert [x['generation_seed'] for x in got]==[SEED_BASE+i for i in range(len(got))]
assert build_seed_manifest(list(reversed(rows)))==m
# The bound seed of every identity survives all allowed physical partitions.
lookup={(x['checkpoint'],x['cohort_record_id']):x['generation_seed'] for x in m['records']}
for width in (32,16,8,4,1):
 for start in range(0,len(rows),width):
  for row in rows[start:start+width]: assert lookup[(row['checkpoint'],row['cohort_record_id'])] == next(x['generation_seed'] for x in m['records'] if x['checkpoint']==row['checkpoint'] and x['cohort_record_id']==row['cohort_record_id'])
assert m['seed_rule'].startswith('generation_seed=2000031+')
for bad in (rows[:-1],rows+[rows[0]]):
 try: build_seed_manifest(bad)
 except RuntimeError: pass
 else: raise AssertionError('invalid seed identity accepted')
print('PASS_ALL_B1_SEED_AMENDMENT_V1')
