import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/run_post_v7_deconfounded_count_causal_pilot_v1.py'
spec=importlib.util.spec_from_file_location('e3_worker',SCRIPT); worker=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(worker)

def test_post_raw_sidecar_and_scorer_with_tiny_generation_mocks(tmp_path):
    """Exercises production order: raw receipt, then target derivation, then scoring."""
    binding=tmp_path/'BINDING.json'; out=tmp_path/'out'
    binding.write_text(json.dumps({'experiment_id':worker.ID,'training':{'seed':1}}),encoding='utf-8')
    out.mkdir(); worker.write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':{},'targets_accessed':False})
    sidecar=worker.make_sidecar(binding,out); targets=worker.load(sidecar)['targets']
    for tag,arm in [('V7',None),('CONTROL_ORIGINAL_CONFOUNDED','CONTROL_ORIGINAL_CONFOUNDED'),('TREATMENT_DECONFOUNDED','TREATMENT_DECONFOUNDED')]:
        rows=[]
        for row in worker.records(arm):
            # a parser-valid mock equals its post-freeze synthetic target.
            rows.append({'record_id':row['id'],'parser_valid':True,'parsed_grid':targets[row['id']],'generated_token_ids':[0]})
        (out/(tag+'_RAW.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
    result=worker.score(binding,out,sidecar)
    assert result['status']=='COMPLETE_E3_SCORE_AFTER_RAW_FREEZE'
    assert result['targets_opened_after_raw_freeze'] is True
    assert result['canonical_units']==48
