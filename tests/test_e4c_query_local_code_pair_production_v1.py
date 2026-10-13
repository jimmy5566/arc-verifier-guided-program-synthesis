import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/run_e4c_query_local_code_pair_causal_pilot_v1.py'
spec=importlib.util.spec_from_file_location('e4c_worker',SCRIPT);w=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(w)

def test_real_tokenization_query_local_selector_has_two_final_tokens_and_exact_mass():
    audit=w.code_pair_audit();assert len(audit)==48
    assert all(len(x['final_query_indices'])==2 and len(x['demo_indices'])==6 for x in audit)
    assert all(x['control_mass']==x['treatment_mass'] for x in audit)
    assert {x['query_count'] for x in audit}=={1,2,3}

def test_frozen_panels_and_amended_marker_gate_are_loaded_without_targets():
    panels=w.validate_cohort();assert len(panels['ood']['records'])==96 and len(panels['marker']['records'])==16
    assert all('output' not in x['prompt']['test'][0] for x in panels['ood']['records']+panels['marker']['records'])
    protocol=w.load(w.PKG/'PROTOCOL.json');g=protocol['gates']
    assert g['marker_binding_every_endpoint_query_local_minus_equal_min']==0
    assert g['marker_binding_aggregate_query_local_minus_equal_min']==0

def test_launcher_worker_raw_freeze_sidecar_scorer_no_model(tmp_path,monkeypatch):
    out=tmp_path/'out';binding=tmp_path/'binding.json'
    binding.write_text(json.dumps({'output_root':str(out),'hard_cap_seconds':7200,'nonce':'testnonce','retry':False,'evaluation':{'bootstrap_seed':7,'bootstrap_replicates':1000}}))
    monkeypatch.setattr(w,'preclaim',lambda *_:{'status':'PASS_E4C_PRODUCTION_PREFLIGHT_NO_MODEL','model_loaded':False,'gpu_used':False,'sidecar_opened':False})
    class CUDA:
        @staticmethod
        def empty_cache(): pass
    class Torch: cuda=CUDA()
    monkeypatch.setattr(w,'load_model',lambda *args:(object(),Torch(),{}));monkeypatch.setattr(w,'train_equal_slot',lambda *args:(object(),Torch(),{}));monkeypatch.setattr(w,'train_code_pair',lambda *args:(object(),Torch(),{}))
    def fake_generate(model,torch,manifest,items,b,outp,tag,deadline):
        vals=[{'record_id':row['id'],'split':row['split'],'prompt_sha256':w.task_hash(row['task']),'generated_token_ids':[1],'parser_valid':True,'parsed_grid':w.target_for_record(row),'effective_batch_size':1} for row in items]
        q=outp/(tag+'_RAW.jsonl');w.write_jsonl(q,vals);return w.sha(q)
    monkeypatch.setattr(w,'generate',fake_generate)
    w.execute(binding)
    assert w.load(out/'RAW_FREEZE_RECEIPT.json')['targets_accessed'] is False
    score=w.load(out/'SCORE_RECEIPT.json');assert score['marker_binding_preservation']['gate'] is True
    assert score['gates']['marker_binding_preservation'] is True
    assert w.load(out/'TERMINAL_RECEIPT.json')['status']=='COMPLETE'

def test_production_preclaim_contract_is_query_local_and_no_model():
    text=SCRIPT.read_text()
    assert 'query_local_code_pair_indices' in text
    assert "'tokens_per_slot':2" in text
    assert "'marker_binding_every_endpoint_delta_min':0" in text
    assert '--preclaim-only' in (ROOT/'scripts/launch_e4c_query_local_code_pair_causal_pilot_v1.sh').read_text()
