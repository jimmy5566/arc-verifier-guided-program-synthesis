from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1/e4c_qlcp_v1'

def rows(name):return json.loads((PKG/name).read_text())
def test_e4c_panels_are_target_blind_balanced_and_disjoint():
 ood,marker=rows('OOD_INPUTS_AND_COMMITMENTS.json'),rows('MARKER_INPUTS_AND_COMMITMENTS.json')
 assert len(ood['records'])==96 and len({x['canonical_unit_id'] for x in ood['records']})==48
 assert {str(c):sum(x['query_count']==c for x in ood['records']) for c in (1,2,3)}=={'1':32,'2':32,'3':32}
 assert all('output' not in x['prompt']['test'][0] for x in ood['records']+marker['records'])
 assert len(marker['records'])==16 and len({x['canonical_unit_id'] for x in marker['records']})==8
 fp=lambda xs:{x['prompt_sha256'] for x in xs}|{x['query_input_sha256'] for x in xs}|{x['content_role_sha256'] for x in xs}
 assert not (fp(ood['records'])&fp(marker['records']))

def test_e4c_protocol_locks_only_query_local_code_pair_and_no_execution():
 p=rows('PROTOCOL.json')
 assert p['fixed_training']['updates']==96 and p['fixed_training']['slots_per_update']==4
 assert p['fixed_training']['learning_rate']==5e-5 and p['execution']['authorization_required'] is True
 assert 'GPU launch before authorization' in p['prohibitions']
 assert p['principal_variable'].count('(1,1)')==1 and p['principal_variable'].count('(1,2)')==1

def test_e4c_freeze_receipt_is_cpu_only_and_target_blind():
 r=rows('PROSPECTIVE_FREEZE_RECEIPT.json')
 assert r['status']=='PASS_E4C_PROSPECTIVE_FREEZE_NO_MODEL'
 assert r['model_loaded'] is False and r['gpu_used'] is False and r['targets_opened'] is False