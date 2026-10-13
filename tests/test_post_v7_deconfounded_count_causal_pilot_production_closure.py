import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/run_post_v7_deconfounded_count_causal_pilot_v1.py'
spec=importlib.util.spec_from_file_location('e3_worker',SCRIPT); worker=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(worker)

def test_post_raw_sidecar_and_scorer_with_tiny_generation_mocks(tmp_path):
    """Exercises production order: raw receipt, then target derivation, then scoring."""
    binding=tmp_path/'BINDING.json'; out=tmp_path/'out'
    binding.write_text(json.dumps({'experiment_id':worker.ID,'training':{'seed':1},'evaluation':{
        'b1_subset_by_checkpoint':{
            tag:[x['id'] for x in sorted(worker.records(arm),key=lambda q:q['id'])[:12]]
            for tag,arm in [('V7',None),('CONTROL_ORIGINAL_CONFOUNDED','CONTROL_ORIGINAL_CONFOUNDED'),('TREATMENT_DECONFOUNDED','TREATMENT_DECONFOUNDED')]},
        'batch1_gate':{'generated_token_agreement_minimum':1.0,'parsed_grid_agreement_minimum':1.0,'exact_agreement_minimum':1.0}}}),encoding='utf-8')
    out.mkdir(); worker.write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':{},'targets_accessed':False})
    sidecar=worker.make_sidecar(binding,out); targets=worker.load(sidecar)['targets']
    raw={}
    for tag,arm in [('V7',None),('CONTROL_ORIGINAL_CONFOUNDED','CONTROL_ORIGINAL_CONFOUNDED'),('TREATMENT_DECONFOUNDED','TREATMENT_DECONFOUNDED')]:
        rows=[]
        for row in worker.records(arm):
            # a parser-valid mock equals its post-freeze synthetic target.
            rows.append({'record_id':row['id'],'prompt_sha256':__import__('hashlib').sha256(worker.canon(row['task'])).hexdigest(),'parser_valid':True,'parsed_grid':targets[row['id']],'generated_token_ids':[0]})
        p=out/(tag+'_RAW.jsonl');p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8');raw[tag]=worker.sha(p)
        ids=worker.load(binding)['evaluation']['b1_subset_by_checkpoint'][tag]
        b1=[x for x in rows if x['record_id'] in ids]
        p=out/(tag+'_B1_RAW.jsonl');p.write_text(''.join(json.dumps(x)+'\n' for x in b1),encoding='utf-8');raw[tag+'_B1']=worker.sha(p)
    worker.write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':raw,'targets_accessed':False})
    result=worker.score(binding,out,sidecar)
    assert result['status']=='COMPLETE_E3_SCORE_AFTER_RAW_FREEZE'
    assert result['targets_opened_after_raw_freeze'] is True
    assert result['canonical_units']==48
