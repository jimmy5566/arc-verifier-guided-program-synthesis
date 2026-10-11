"""CPU-only E04-E V2 scoring after target-blind raw generations are frozen."""
from __future__ import annotations
import argparse,json,random,sys
from collections import defaultdict
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.e04_e_equal_slot_execution import ARMS,atomic_json,fail,sha_path
from scripts.e04_v3_full768_b1_remeasurement import load_config,read_jsonl,validate_full_raw
CONDS=('ROTATION_TARGET','FIXED_TURN_ROTATION_CONTROL','MARKER_BINDING_CONTROL','NO_TRANSFORM_RETENTION_CONTROL')
def percentile(values:list[float],p:float)->float:
 values=sorted(values);return values[int(round((len(values)-1)*p))]
def scored(raw:Path,prompts:list[dict[str,Any]],targets:list[dict[str,Any]])->list[dict[str,Any]]:
 return [{'row_index':x['row_index'],'canonical_base_id':y['canonical_base_id'],'condition':y['condition'],'parser_valid':x['parser_valid'],'exact_grid':x['parsed_grid']==y['target']} for x,y in zip(validate_full_raw(raw,prompts),targets,strict=True)]
def base_values(rows:list[dict[str,Any]])->dict[str,dict[str,float]]:
 by=defaultdict(lambda:defaultdict(list))
 for x in rows:by[x['canonical_base_id']][x['condition']].append(float(x['exact_grid']))
 if len(by)!=48 or any(set(v)!=set(CONDS) or any(len(v[c])!=4 for c in CONDS) for v in by.values()):fail('E04E_SCORE_BASE_COVERAGE')
 return {base:{c:sum(v[c])/4 for c in CONDS} for base,v in by.items()}
def paired(delta:dict[str,float])->dict[str,float]:
 bases=sorted(delta); point=sum(delta.values())/len(bases);rng=random.Random(20261011);draws=[sum(delta[rng.choice(bases)] for _ in bases)/len(bases) for _ in range(10000)]
 return {'accuracy_delta':point,'episode_delta':point*192.0,'ci95_low':percentile(draws,.025),'ci95_high':percentile(draws,.975)}
def summary(rows:list[dict[str,Any]],ref:list[dict[str,Any]])->dict[str,Any]:
 a,b=base_values(rows),base_values(ref);accuracy={c:sum(float(x['exact_grid']) for x in rows if x['condition']==c)/192 for c in CONDS};parser={c:sum(float(x['parser_valid']) for x in rows if x['condition']==c)/192 for c in CONDS}
 return {'condition_exact_grid_accuracy':accuracy,'condition_parser_validity':parser,'delta_vs_v7_by_condition':{c:paired({base:a[base][c]-b[base][c] for base in a}) for c in CONDS},'_bases':a}
def score(control_raw:Path,treatment_raw:Path,receipt_path:Path,binding_path:Path,output:Path)->dict[str,Any]:
 if output.exists():fail('E04E_SCORE_OUTPUT_FRESH')
 receipt=json.loads(receipt_path.read_text());binding=json.loads(binding_path.read_text())
 if receipt.get('status')!='COMPLETE_PENDING_CPU_SCORE' or receipt.get('target_sidecar_accessed') is not False or receipt.get('raw_sha256')!={ARMS[0]:sha_path(control_raw),ARMS[1]:sha_path(treatment_raw)}:fail('E04E_RAW_FREEZE')
 config=load_config();prompts=read_jsonl(ROOT/config['prompt_path']);ref_path=ROOT/binding['v7_reference_raw_path']
 if sha_path(ref_path)!=binding['v7_reference_raw_sha256']:fail('E04E_V7_RAW')
 sidecar=ROOT/config['target_sidecar_path']
 if sha_path(sidecar)!=config['target_sidecar_sha256']:fail('E04E_TARGET_SIDECAR')
 targets=read_jsonl(sidecar)
 if len(targets)!=768 or [x.get('row_index') for x in targets]!=list(range(768)):fail('E04E_TARGET_MAPPING')
 ref_rows=scored(ref_path,prompts,targets);control_rows=scored(control_raw,prompts,targets);treatment_rows=scored(treatment_raw,prompts,targets)
 control=summary(control_rows,ref_rows);treatment=summary(treatment_rows,ref_rows);cb,tb=control.pop('_bases'),treatment.pop('_bases')
 pair={c:paired({base:tb[base][c]-cb[base][c] for base in tb}) for c in CONDS}
 primary=pair['MARKER_BINDING_CONTROL']; marker=treatment['delta_vs_v7_by_condition']['MARKER_BINDING_CONTROL'];rotation=treatment['delta_vs_v7_by_condition']['FIXED_TURN_ROTATION_CONTROL'];rotation_control=pair['FIXED_TURN_ROTATION_CONTROL'];retention=treatment['delta_vs_v7_by_condition']['NO_TRANSFORM_RETENTION_CONTROL']
 success=primary['episode_delta']>=8 and primary['ci95_low']>0 and marker['episode_delta']>=-8 and rotation['episode_delta']>=8 and rotation['ci95_low']>0 and rotation_control['episode_delta']>=-8 and retention['episode_delta']>=-8
 if success:decision='MARKER_PRESERVATION_GAIN'
 elif marker['episode_delta']<-8 or rotation['episode_delta']<8 or rotation['ci95_low']<=0 or rotation_control['episode_delta']<-8 or retention['episode_delta']<-8:decision='PROTECTION_OR_EFFICACY_FAILURE'
 else:decision='NO_MARKER_REPLAY_BENEFIT'
 result={'schema_version':1,'protocol_id':binding['protocol_id'],'status':'SCORED_DEVELOPMENT_EVIDENCE','scientific_scope':'REUSED_E04_DEVELOPMENT_COHORT_NOT_INDEPENDENT_ARC_GENERALIZATION_CONFIRMATION','binding_sha256':sha_path(binding_path),'raw_freeze_receipt_sha256':sha_path(receipt_path),'control_raw_sha256':sha_path(control_raw),'treatment_raw_sha256':sha_path(treatment_raw),'v7_reference_raw_sha256':sha_path(ref_path),'target_scorer_sidecar_sha256':sha_path(sidecar),'target_sidecar_opened_only_after_all_raw_freeze_validation':True,'bootstrap_seed':20261011,'bootstrap_replicates':10000,'control_vs_v7':control,'treatment_vs_v7':treatment,'treatment_minus_control':{'by_condition':pair,'primary_marker_binding':primary},'decision':decision,'rules':{'primary_success':primary,'marker_protection_vs_v7':marker,'rotation_efficacy_vs_v7':rotation,'rotation_noninferiority_vs_control':rotation_control,'no_transform_retention_vs_v7':retention},'final_audit_opened':False}
 atomic_json(output,result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--control-raw',type=Path,required=True);p.add_argument('--treatment-raw',type=Path,required=True);p.add_argument('--raw-freeze-receipt',type=Path,required=True);p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(score(a.control_raw,a.treatment_raw,a.raw_freeze_receipt,a.binding,a.output),sort_keys=True))
if __name__=='__main__':main()
