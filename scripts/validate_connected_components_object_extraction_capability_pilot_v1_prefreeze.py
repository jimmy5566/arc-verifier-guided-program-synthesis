"""CPU-only schema gate for the Director-requested prefreeze draft."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DRAFT=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/PREFREEZE_DRAFT.json'

def validate(path:Path=DRAFT)->dict:
    d=json.loads(path.read_text(encoding='utf-8'))
    assert d['status']=='CPU_ONLY_PREFREEZE_DRAFT_NO_DATA_NO_MODEL_NO_EXECUTION'
    assert [x['arm_id'] for x in d['arms']]==['CONTROL_NARROW_STRUCTURE','TREATMENT_STRUCTURE_DIVERSE']
    assert len({x['initialization'] for x in d['arms']})==1
    assert len({x['protected_replay'] for x in d['arms']})==1
    assert set(d['matched_budgets'])=={'optimizer','update_steps','supervised_tokens','replay_examples','replay_bytes'}
    assert all(v=='IDENTICAL_TO_BE_FROZEN' for v in d['matched_budgets'].values())
    assert d['future_validation']['cohort']=='NEWLY_GENERATED_NON_GOLD_STRUCTURALLY_DISJOINT_TO_BE_FROZEN'
    assert d['future_validation']['primary_outcome']=='NATIVE_GREEDY_EXACT_GRID_ACCURACY'
    forbidden=set(d['forbidden_before_director_prelaunch_review'])
    assert {'MODEL_LOADING','GPU','TRAINING','BACKWARD','INFERENCE','TARGET_ACCESS','GOLD','DGOLD','FINAL_AUDIT'} <= forbidden
    return {'status':'PASS_CPU_ONLY_PREFREEZE_DRAFT','model_loaded':False,'gpu_used':False,'data_generated':False}

if __name__=='__main__': print(json.dumps(validate(),sort_keys=True))
