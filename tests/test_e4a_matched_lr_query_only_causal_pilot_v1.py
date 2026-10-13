import hashlib,importlib.util,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/run_e4a_matched_lr_query_only_causal_pilot_v1.py'
spec=importlib.util.spec_from_file_location('e4a_worker',SCRIPT);w=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(w)
FREEZER=ROOT/'scripts/freeze_e4a_matched_lr_query_only_causal_pilot_v1.py'
s2=importlib.util.spec_from_file_location('e4a_freezer',FREEZER);f=importlib.util.module_from_spec(s2);assert s2.loader;s2.loader.exec_module(f)

def test_exact_schedule_identity_and_query_only_mask_contract():
    schedule=w.schedule()
    assert len(w.sha(w.E3/'TREATMENT_SCHEDULE.json'))==64
    assert len(schedule)==384 and [x['slot'] for x in schedule]==list(range(384))
    audit=w.mask_audit()
    assert len(audit)==48 and {x['selected_count'] for x in audit}=={15} and {x['all_assistant_count'] for x in audit}=={60}
    count=next(x for x in schedule if x['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT')
    sample,masked,detail=w.query_only_labels(w.serializer_task(count))
    assert sum(x!=w.IGNORE for x in masked)==15
    assert all(x==w.IGNORE for x in masked[:detail['assistant_turn_runs'][-1][0]])
    assert [x for x in masked if x!=w.IGNORE]==detail['selected_token_ids']
    noncount=next(x for x in schedule if x['role']=='PRIMARY_OBJECT_EXTRACTION')
    original=w.serializer_task(noncount)
    from training_data.pipeline import task_to_sample
    assert task_to_sample(original)['labels']!=w.query_only_labels(original)[1]  # masking helper is never applied to non-Count slots

def test_causal_shift_scalar_mass_and_noncount_equal_slot_gradient():
    from scripts.e04_e_equal_slot_loss import causal_slot_mean_ce,backward_equal_slot_term
    labels=torch.tensor([[-100,1,2,3,4]],dtype=torch.long)
    query=torch.tensor([[-100,-100,-100,3,4]],dtype=torch.long)
    logits=torch.randn(1,5,7,dtype=torch.float64,requires_grad=True)
    m,n=causal_slot_mean_ce(logits,query)
    assert n==2
    (m*.25).backward()
    assert logits.grad is not None and logits.grad[:,0:2].abs().sum()==0  # unshifted query labels are predicted by prior positions
    # non-Count path remains exactly the established Equal-Slot function and coefficient.
    a=torch.randn(1,7,11,dtype=torch.float64,requires_grad=True);lab=torch.tensor([[-100,1,2,3,4,5,-100]],dtype=torch.long)
    b=a.detach().clone().requires_grad_(True)
    ta,na=backward_equal_slot_term(a,lab);tb,nb=backward_equal_slot_term(b,lab)
    assert na==nb and torch.allclose(ta,tb) and torch.allclose(a.grad,b.grad)

def test_new_cohort_is_target_blind_independent_and_zero_overlap():
    c=f.cohort()
    assert len(c['records'])==96 and len({x['canonical_unit_id'] for x in c['records']})==48
    assert c['topology_count_independence']['mutual_information_bits']==0.0
    assert c['zero_overlap']['training_e3_exposed_ood_prompt_sha256']
    assert c['zero_overlap']['training_e3_exposed_ood_query_input_sha256'] and c['zero_overlap']['training_e3_query_only_exposed_ood_prompt_sha256'] and c['zero_overlap']['training_all_exposed_content_role_sha256']
    assert all('output' not in x['prompt']['test'][0] for x in c['records'])

def _target(row,bases):
    return w.target_for_record(row,bases)

def test_launcher_worker_raw_freeze_sidecar_scorer_without_model(tmp_path,monkeypatch):
    out=tmp_path/'out'
    binding=tmp_path/'binding.json'
    binding.write_text(json.dumps({'output_root':str(out),'hard_cap_seconds':7200,'nonce':'testnonce','retry':False,'evaluation':{'bootstrap_seed':7,'bootstrap_replicates':10000}}))
    monkeypatch.setattr(w,'preclaim',lambda _: {'status':'PASS_NO_MODEL','model_loaded':False,'gpu_used':False,'sidecar_opened':False})
    class CUDA:
        @staticmethod
        def empty_cache(): pass
    class Torch: cuda=CUDA()
    monkeypatch.setattr(w,'load_model',lambda *args:(object(),Torch(),{}))
    monkeypatch.setattr(w,'train_equal_slot',lambda *args:(object(),Torch(),{}))
    monkeypatch.setattr(w,'train_query_only',lambda *args:(object(),Torch(),{}))
    from scripts import e04_orientation_fixed_demonstration_baseline_v3 as orient
    bases={x['canonical_base_id']:x for x in orient.v2.canonical_bases('VALIDATION')}
    def fake_generate(model,torch,manifest,items,b,outp,tag,deadline):
        rows=[]
        for row in items:
            rows.append({'record_id':row['id'],'split':row['split'],'prompt_sha256':w.task_hash(row['task']),'generated_token_ids':[1],'parser_valid':True,'parsed_grid':_target(row,bases),'effective_batch_size':1})
        p=outp/(tag+'_RAW.jsonl');w.write_jsonl(p,rows);return w.sha(p)
    monkeypatch.setattr(w,'generate',fake_generate)
    w.execute(binding)
    frozen=w.load(out/'RAW_FREEZE_RECEIPT.json')
    assert frozen['targets_accessed'] is False
    assert w.load(out/'SYNTHETIC_TARGET_SIDECAR.json')['opened_after_raw_freeze'] is True
    scored=w.load(out/'SCORE_RECEIPT.json')
    assert scored['physical_batch_size']==1 and scored['batch32_used'] is False
    assert w.load(out/'TERMINAL_RECEIPT.json')['status']=='COMPLETE'

def test_e3_audit_forbids_reuse_and_e4a_has_no_code_pair_dependency():
    audit=w.load(w.PKG/'E3_REFERENCE_EQUIVALENCE_AUDIT.json')
    assert audit['status']=='REUSE_FORBIDDEN_INCOMPLETE_PROVENANCE_FRESH_BOTH_ARMS_REQUIRED'
    assert audit['field_audit']['optimizer']=='UNKNOWN'
    src=w.Path(w.__file__).read_text()
    assert 'decision_token_supervision' not in src and 'locate_final_query_assistant' not in src
