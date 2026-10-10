"""CPU scorer for E04 V3 full768 B1, after target-blind freeze and B1 repeatability."""
from __future__ import annotations
import argparse,json,random
from collections import defaultdict
from pathlib import Path
from scripts.e04_orientation_v3_no_update import CONDITIONS
from scripts.e04_v3_full768_b1_remeasurement import ROOT,FullB1Failure,atomic_json,load_config,read_json,read_jsonl,sha_path,validate_full_raw,verify_embedded_reference

def fail(x): raise FullB1Failure(x)
def percentile(xs,p):
 xs=sorted(xs); return xs[int(round((len(xs)-1)*p))]
def score(raw_path:Path,receipt_path:Path,destination:Path):
 c=load_config(); receipt=read_json(receipt_path)
 if receipt.get('status')!='RAW_GENERATIONS_FROZEN_NO_TARGETS' or receipt.get('target_sidecar_accessed') is not False or receipt.get('primary_b1_raw_sha256')!=sha_path(raw_path): fail('E04_FULL768_RAW_FREEZE_IDENTITY')
 prompts=read_jsonl(ROOT/c['prompt_path']); raw=validate_full_raw(raw_path,prompts); verify_embedded_reference(raw,c)
 # The repeatability gate has passed; only now open the sealed target sidecar.
 side=ROOT/c['target_sidecar_path']
 if sha_path(side)!=c['target_sidecar_sha256']: fail('E04_FULL768_TARGET_IDENTITY')
 targets=read_jsonl(side)
 if len(targets)!=768 or [x.get('row_index') for x in targets]!=list(range(768)): fail('E04_FULL768_TARGET_MAPPING')
 rows=[]
 for pred,truth in zip(raw,targets,strict=True): rows.append({'row_index':pred['row_index'],'canonical_base_id':truth['canonical_base_id'],'condition':truth['condition'],'reserved_cell':truth['reserved_cell'],'control_marker_turn':truth['control_marker_turn'],'parser_valid':pred['parser_valid'],'exact_grid':pred['parsed_grid']==truth['target']})
 by_condition=defaultdict(list);by_base=defaultdict(lambda:defaultdict(list));by_cell=defaultdict(set)
 for row in rows: by_condition[row['condition']].append(row);by_base[row['canonical_base_id']][row['condition']].append(row);by_cell[json.dumps(row['reserved_cell'],sort_keys=True)].add(row['canonical_base_id'])
 if set(by_condition)!=set(CONDITIONS) or any(len(v)!=192 for v in by_condition.values()): fail('E04_FULL768_CONDITION_COVERAGE')
 unit={}
 for base,conditions in by_base.items():
  if set(conditions)!=set(CONDITIONS) or any(len(v)!=4 for v in conditions.values()): fail('E04_FULL768_MATCHED_UNIT_COVERAGE')
  unit[base]={k:sum(float(r['exact_grid']) for r in v)/4 for k,v in conditions.items()}
 accuracy={k:sum(float(r['exact_grid']) for r in v)/len(v) for k,v in by_condition.items()}; parser={k:sum(float(r['parser_valid']) for r in v)/len(v) for k,v in by_condition.items()}
 rng=random.Random(20261010);cells=[sorted(v) for _,v in sorted(by_cell.items())]; contrasts={}
 for control in ('FIXED_TURN_ROTATION_CONTROL','MARKER_BINDING_CONTROL','NO_TRANSFORM_RETENTION_CONTROL'):
  point=sum(unit[b]['ROTATION_TARGET']-unit[b][control] for b in unit)/len(unit); draws=[]
  for _ in range(10000):
   sample=[rng.choice(cell) for cell in cells for _ in range(len(cell))];draws.append(sum(unit[b]['ROTATION_TARGET']-unit[b][control] for b in sample)/len(sample))
  contrasts[control]={'point':point,'ci95_low':percentile(draws,.025),'ci95_high':percentile(draws,.975)}
 result={'schema_version':1,'protocol_id':c['protocol_id'],'status':'SCORED_DEVELOPMENT_EVIDENCE','scientific_scope':'REUSED_DEVELOPMENT_COHORT_NOT_INDEPENDENT_ARC_GENERALIZATION_CONFIRMATION','raw_freeze_receipt_sha256':sha_path(receipt_path),'primary_b1_raw_sha256':sha_path(raw_path),'sensitivity_reference_raw_sha256':c['sensitivity_reference_raw_sha256'],'b1_repeatability_gate':'PASS_64_OF_64','target_scorer_sidecar_sha256':sha_path(side),'target_sidecar_opened_only_after_raw_freeze_and_b1_repeatability':True,'rows':768,'canonical_bases':48,'condition_exact_grid_accuracy':accuracy,'condition_parser_validity':parser,'matched_rotation_target_minus_control_bootstrap':contrasts,'forbidden_operations':{'optimizer_steps':0,'parameter_updates':0,'backward_calls':0,'final_audit_opened':False}}
 if destination.exists(): fail('E04_FULL768_SCORE_OUTPUT_FRESH')
 atomic_json(destination,result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--primary-b1-raw',type=Path,required=True);p.add_argument('--raw-freeze-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(score(a.primary_b1_raw,a.raw_freeze_receipt,a.output),sort_keys=True))
if __name__=='__main__':main()
