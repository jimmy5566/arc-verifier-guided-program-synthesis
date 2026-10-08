"""Freeze the non-executable prospective characterization replacement.

No payload reference, model, generator, or scorer is opened or run here.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments'/'foundation_v2_reconstruction_and_targeted_repair_v2'/'round_009_equivalence_evaluation'
NEW=ROOT/'experiments'/'foundation_v2_reconstruction_and_targeted_repair_v2'/'round_009_post_hoc_capability_characterization'
def dig(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p:Path,x:dict)->None:p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
def main()->int:
    NEW.mkdir(parents=True,exist_ok=True)
    surfaces={}
    for name,label in [('NOVEL_GENERATION_INPUT_MANIFEST.json','NOVEL'),('REPLAY_GENERATION_INPUT_MANIFEST.json','REPLAY'),('PROTECTED_CAPABILITY_INPUT_MANIFEST.json','PROTECTED')]:
        x=json.loads((OLD/name).read_text(encoding='utf-8')); items=x['items'];
        assert len(items)==x['episode_count'] and len({i['sample_id'] for i in items})==len(items)
        surfaces[label]={'source_manifest':str((OLD/name).relative_to(ROOT)).replace('\\','/'),'source_manifest_sha256':dig(OLD/name),'item_count':len(items),'unique_item_ids':len({i['sample_id'] for i in items}),'contains_targets':False,'reference_mapping_status':'UNRESOLVED_NO_PAYLOAD_OPENED'}
    closure={'schema_version':1,'closed_stage_id':'RECONSTRUCTION_EQUIVALENCE_EVALUATION_STAGE_V1','closed_stage_status':'UNEXECUTED_FAIL_CLOSED','closure_reason':'EXACT_SEALED_REFERENCE_AND_PROTECTED_BASELINE_PROVENANCE_MISSING','immutable_artifacts':[str((OLD/n).relative_to(ROOT)).replace('\\','/') for n in ['RECONSTRUCTION_EQUIVALENCE_EVALUATION_STAGE_V1.json','ROUND_009_SEALED_EQUIVALENCE_EVALUATION_V1_PROTOCOL.json','ROUND_009_SEALED_EQUIVALENCE_EVALUATION_V1_NONCES.json','ROUND_009_SEALED_REFERENCE_PROVENANCE_AUDIT.json']],'actions_forbidden':['consume_old_nonces','populate_old_output_roots','relabel_old_stage_pass','mutate_round009','retrain','targeted_repair','FINAL_AUDIT']}
    write(NEW/'ROUND_009_EQUIVALENCE_STAGE_V1_CLOSURE.json',closure)
    protocol={'schema_version':1,'protocol_id':'ROUND_009_POST_HOC_CAPABILITY_CHARACTERIZATION_V1','condition':'PROSPECTIVE_POST_HOC_CHARACTERIZATION_OF_FIXED_RECONSTRUCTED_CHECKPOINT','not_original_equivalence_gate':True,'not_foundation_v2_recovery':True,'not_targeted_repair_entry':True,'checkpoint':{'adapter_model_sha256':'1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3','adapter_config_sha256':'faa1a3bfb957209ca10ea58afbbf275ce5b51592dd78f289b68f7b655ede2b69'},'populations':surfaces,'protected_comparator_claim':'OMITTED_UNLESS_EXACT_PROSPECTIVE_COMPARATOR_IS_BOUND_BEFORE_EXECUTION','two_phase_requirements':['target_blind_phase_a','sealed_phase_b','no_feedback','no_checkpoint_or_curriculum_change'],'execution_authorized':False,'status':'PREPARATION_BLOCKED_EXACT_NON_GOLD_REFERENCE_MAPPINGS_UNRESOLVED','forbidden':['Eval60 Gold','diagnostic Gold','exposed Novel holdout targets','FINAL_AUDIT','targeted-repair data','model loading','generation','scoring','GPU inference']}
    write(NEW/'ROUND_009_POST_HOC_CAPABILITY_CHARACTERIZATION_V1_PROTOCOL.json',protocol)
    contract={'schema_version':1,'protocol_id':protocol['protocol_id'],'population_cardinality':{k:v['item_count'] for k,v in surfaces.items()},'mapping_bijection_required':True,'reference_isolation_required':True,'generator_identity_required_before_phase_a':True,'scorer_identity_required_before_phase_b':True,'deterministic_decode_seed_required':True,'fresh_nonce_and_output_root_required':True,'receipt_completeness_required':True,'regression_tests_required':['item_count','mapping_bijection','reference_isolation','duplicate_missing_invalid_predictions','nonce_reuse','receipt_completeness'],'no_model_preflight_status':'BLOCKED_PENDING_EXACT_NON_GOLD_REFERENCE_MAPPINGS','execution_authorized':False}
    write(NEW/'POST_HOC_CHARACTERIZATION_EXECUTION_CONTRACT.json',contract)
    print('POST_HOC_CHARACTERIZATION_FREEZE=PASS');return 0
if __name__=='__main__':raise SystemExit(main())
