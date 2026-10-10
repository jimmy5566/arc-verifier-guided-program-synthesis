from __future__ import annotations
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'experiments/capability_repair_baseline_v1/e04_orientation_frozen_cohort_v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return p.relative_to(ROOT).as_posix()
def main():
    files=[D/'COHORT.jsonl',D/'SUCCESSOR_METADATA_V2.json',D/'CONDITION_TO_BASE_MAP_V2.json',D/'COLLISION_REPORT_V2.json',D/'INDEPENDENT_TARGET_ORACLE_REPORT_V2.json',D/'TURN_CONTINGENCY_V2.json',D/'TEMPORARY_DIRECTORY_DETERMINISTIC_REGENERATION_REPORT_V1.json',ROOT/'experiments/capability_repair_baseline_v1/CPU_ONLY_E04_ORIENTATION_GREENFIELD_PREFREEZE_SPECIFICATION_V2.json',ROOT/'experiments/capability_repair_baseline_v1/E04_ORIENTATION_GREENFIELD_GENERATOR_DESIGN_SPEC_V2.json',ROOT/'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json',ROOT/'scripts/generate_e04_orientation_frozen_cohort_v1.py',ROOT/'scripts/validate_e04_orientation_frozen_cohort_v1.py']
    package={
      'schema_version':1,
      'protocol_id':'E04_ORIENTATION_V7_ONLY_NO_UPDATE_EVALUATION_V1',
      'status':'CPU_ONLY_PRELAUNCH_PACKAGE_REVIEW_REQUIRED',
      'scientific_question':'On the frozen development cohort, does V7 fail ROTATION_TARGET after both prerequisite controls are exact within the same independent base tuple?',
      'checkpoint':{'identity':'RECONSTRUCTED_FOUNDATION_V2_V7','manifest_path':rel(ROOT/'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json'),'manifest_sha256':sha(ROOT/'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json'),'model_access_authorized':False},
      'cohort':{'path':rel(D/'COHORT.jsonl'),'sha256':sha(D/'COHORT.jsonl'),'rows':1344,'validation_base_tuples':48,'validation_condition_views':192,'independent_unit':'condition-excluded base_tuple; four condition views are paired repeated measurements only','development_evidence_only':True},
      'conditions':{'target':'ROTATION_TARGET','prerequisites':['FIXED_TURN_ROTATION_CONTROL','MARKER_BINDING_CONTROL'],'retention_control':'NO_TRANSFORM_RETENTION_CONTROL'},
      'estimands':{'primary':'Per reserved cell, V7 ROTATION_TARGET exact-grid failure rate restricted to base tuples where both prerequisite controls are exact; equal-cell macro average only when each cell has >=8 conditional analyzable base tuples.','secondary':['Unconditional ROTATION_TARGET exact-grid accuracy by cell and equal-cell macro average.','Exact-grid accuracy for each paired condition.','Paired within-base control/target outcome table.'],'uncertainty':{'method':'reserved-cell-stratified bootstrap at the independent base-tuple level with paired views retained within resamples','seed':20261010,'replicates':10000,'insufficient_conditional_count':'INCONCLUSIVE'}},
      'runtime_proposal_for_review':{'mode':'one V7-only no-update exact-grid evaluation','generation_required':True,'optimizer_steps':0,'parameter_updates':0,'backward':False,'training':False,'gradient_computation':False,'Gold_access':False,'dGold_access':False,'FINAL_AUDIT_access':False,'requested_batch_size':32,'frozen_fallback_ladder':[32,16,8,4,1],'batch1_validation':'fixed representative subset must have parser/scorer/aggregate agreement within a predeclared numerical tolerance; material prediction or metric drift is INVALID_NOT_INTERPRETABLE','authorization':'No model loading, inference, GPU use, launch binding, nonce, output root, or remote job is authorized by this package.'},
      'invalidation':['any bound source/cohort/checkpoint identity mismatch => INVALID_NOT_INTERPRETABLE','missing complete paired views or a condition counted as an independent unit => INVALID_NOT_INTERPRETABLE','fewer than 8 conditional analyzable base tuples in any reserved cell => conditional primary estimand INCONCLUSIVE','any sealed data access, update, backward pass, optimizer construction, training, or checkpoint mutation => INVALID_NOT_INTERPRETABLE'],
      'bound_files':[{ 'path':rel(p),'sha256':sha(p)} for p in files],
      'review_request':'Authorize or reject one bounded V7-only no-update evaluation under these frozen cohort and estimand conditions. This package does not request a new training protocol or change the frozen cohort.'
    }
    out=D/'E04_ORIENTATION_V7_ONLY_NO_UPDATE_EVALUATION_PRELAUNCH_PACKAGE_V1.json'
    out.write_text(json.dumps(package,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'path':rel(out),'sha256':sha(out),'cohort_sha256':package['cohort']['sha256']},sort_keys=True))
if __name__=='__main__':main()
