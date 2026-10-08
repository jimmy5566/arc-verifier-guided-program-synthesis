"""Freeze post-hoc Phase A/B command templates without self-referential Git hashes.

The template is committed with entrypoint hashes.  A separate runtime lock is
created after the exact checkout exists and must bind that immutable template
hash to the live commit before a nonce can be consumed.
"""
from __future__ import annotations
import hashlib, json, secrets
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'experiments/foundation_v2_reconstruction_and_targeted_repair_v2/round_009_post_hoc_capability_characterization'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
def main():
 old=json.loads((P/'ROUND_009_POST_HOC_PHASE_A_BINDING_V1.json').read_text())
 directive=ROOT/'orchestration/director/directives/DIRECTOR_DIRECTIVE_024.json'
 common={'schema_version':2,'protocol_id':old['protocol_id'],'round_id':old['round_id'],'execution_authorized':False,'training_authorized':False,'final_audit_authorized':False,'runtime_launch_lock_required':True,'execution_checkout_commit':None,'authorization_directive_id':'DIRECTOR_DIRECTIVE_024','authorization_directive_sha256':sha(directive),'entrypoint_identities':{n:sha(ROOT/'scripts'/n) for n in ('run_round009_posthoc_phase_a_v1.py','run_round009_posthoc_phase_b_v1.py','run_round009_posthoc_stage_launcher_v1.py')},'runtime':old['runtime'],'checkpoint':old['checkpoint'],'input_manifests':old['input_manifests'],'asset_identities':old['base_and_tokenizer_files']+[{'path':old['checkpoint']['path']+'/adapter_model.safetensors','sha256':old['checkpoint']['adapter_model_sha256']},{'path':old['checkpoint']['path']+'/adapter_config.json','sha256':old['checkpoint']['adapter_config_sha256']}],'binding_template_not_self_referential':True}
 na=secrets.token_hex(24); nb=secrets.token_hex(24)
 a={**common,'phase':'A_GENERATION','nonce':'phase-a-v2-'+na,'output_root':'/workspace/arc2/active_runs/reconstructed_foundation_v2_v2_009/post_hoc_characterization_v1/phase_a_predictions_v2_'+na,'terminal_receipt_path':'{OUTPUT_ROOT}/PHASE_A_TERMINAL_RECEIPT.json','sealed_mapping_must_be_unreadable':True,'argv':['scripts/run_round009_posthoc_phase_a_v1.py','--binding','{BINDING_PATH}','--output','{OUTPUT_ROOT}','--receipt','{TERMINAL_RECEIPT_PATH}','--execute'],'required_environment':old['runtime']['environment'],'failure':'FAIL_CLOSED_NO_NONCE_CONSUMPTION_ON_PRECHECK_FAILURE'}
 b={**common,'phase':'B_SEALED_SCORING','nonce':'phase-b-v2-'+nb,'output_root':'/workspace/arc2/active_runs/reconstructed_foundation_v2_v2_009/post_hoc_characterization_v1/sealed_scorer/phase_b_scores_v2_'+nb,'terminal_receipt_path':'{OUTPUT_ROOT}/PHASE_B_TERMINAL_RECEIPT.json','requires_phase_a_terminal_immutable':True,'denominator_rules':'EXACT_128_64_64_NO_SHRINKING','protected_semantics':'ABSOLUTE_ONLY','argv':['scripts/run_round009_posthoc_phase_b_v1.py','--binding','{BINDING_PATH}','--prediction-manifest','{PHASE_A_PREDICTION_MANIFEST}','--phase-a-receipt','{PHASE_A_TERMINAL_RECEIPT}','--sealed-mapping','{SEALED_MAPPING_PATH}','--output','{OUTPUT_ROOT}','--receipt','{TERMINAL_RECEIPT_PATH}'],'failure':'FAIL_CLOSED_IF_PHASE_A_RECEIPT_OR_PREDICTIONS_INVALID'}
 write(P/'ROUND_009_POST_HOC_PHASE_A_BINDING_V2.json',a);write(P/'ROUND_009_POST_HOC_PHASE_B_BINDING_V2.json',b)
 print('PHASE_BINDING_V2_FREEZE=PASS')
if __name__=='__main__':main()
