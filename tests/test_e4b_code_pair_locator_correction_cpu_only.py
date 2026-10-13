from scripts.e4b_code_pair_locator_correction_cpu_only import locate_final_query_assistant
def test_final_query_locator_excludes_demo_turns_and_preserves_ids():
 labels=[-100,11,12,-100,21,22,-100,31,32,-100,41,42]; ids=[0,11,12,0,21,22,0,31,32,0,41,42]
 got=locate_final_query_assistant(labels,ids,3)
 assert got["assistant_turns"]==[(1,3),(4,6),(7,9),(10,12)]
 assert got["token_ids"]==[41,42]
def test_locator_rejects_malformed_assistant_turn_count():
 try: locate_final_query_assistant([-100,1],[-100,1],1)
 except ValueError as e: assert "GRID" in str(e)
 else: assert False

def test_locator_is_paired_with_connected_component_oracle_check():
 import json
 from pathlib import Path
 from scripts.connected_components_object_extraction_semantic_oracle import target_for_family,validate_demonstrations
 root=Path(__file__).resolve().parents[1]
 rows=json.loads((root/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1/TREATMENT_SCHEDULE.json').read_text())['episodes']
 task=next(x['task'] for x in rows if x['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT')
 validate_demonstrations(task,'CONNECTED_COMPONENT_COUNT')
 assert target_for_family('CONNECTED_COMPONENT_COUNT',task['test'][0]['input'])==task['test'][0]['output']
