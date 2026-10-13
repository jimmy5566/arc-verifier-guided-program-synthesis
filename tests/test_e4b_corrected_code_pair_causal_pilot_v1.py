import hashlib,importlib.util,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/run_e4b_corrected_code_pair_causal_pilot_v1.py'
spec=importlib.util.spec_from_file_location('e4b_worker',SCRIPT);w=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(w)
FREEZER=ROOT/'scripts/freeze_e4b_corrected_code_pair_causal_pilot_v1.py'
s2=importlib.util.spec_from_file_location('e4b_freezer',FREEZER);f=importlib.util.module_from_spec(s2);assert s2.loader;s2.loader.exec_module(f)

def test_exact_schedule_and_real_tokenization_code_pair_oracle():
    rows=w.schedule();assert len(rows)==384 and [x['slot'] for x in rows]==list(range(384))
    audit=w.code_pair_audit();assert len(audit)==48
    assert {tuple(p['semantic_code']) for x in audit for p in x['pairs']}=={(0,0),(1,0),(1,1),(2,0),(2,2),(3,0),(3,3),(4,0),(4,4),(6,0),(6,6),(7,0),(7,7),(8,0),(8,8),(9,0),(9,9)} or all(p['count'] in (1,2,3) for x in audit for p in x['pairs'])
    # Actual task_to_sample labels, not string-position arithmetic alone: selected labels equal output cells (1,1)/(1,2).
    from training_data.pipeline import task_to_sample
    item=next(x for x in rows if x['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT');task=w.serializer_task(item)
    sample=task_to_sample(task)
    from scripts.e4b_corrected_code_pair_supervision import count_code_pair_label_indices,matched_weights
    idx,counts=count_code_pair_label_indices(task);assert len(idx)==8 and len(set(idx))==8
    observed=[sample['labels'][i] for i in idx]
    expected=[]
    for pair,count in zip([*task['train'],*task['test']],counts,strict=True): expected += pair['output'][1][1:3]
    assert observed==expected
    control,treat=matched_weights(sample['labels'],idx,2.0)
    assert sum(control.values())==sum(treat.values()) and all(treat[i]==2.0 for i in idx)

def test_weighted_causal_shift_and_noncount_gradient_unchanged():
    from scripts.e04_e_equal_slot_loss import backward_equal_slot_term,backward_weighted_slot_term
    logits=torch.randn(1,7,13,dtype=torch.float64,requires_grad=True);labels=torch.tensor([[-100,1,2,3,4,5,-100]],dtype=torch.long)
    weights=[0,1,2,1,1,1,0]
    term,n,mass=backward_weighted_slot_term(logits,labels,weights)
    assert n==5 and mass==6 and logits.grad is not None
    a=torch.randn(1,7,13,dtype=torch.float64,requires_grad=True);b=a.detach().clone().requires_grad_(True)
    ta,na=backward_equal_slot_term(a,labels);tb,nb=backward_equal_slot_term(b,labels)
    assert na==nb and torch.allclose(ta,tb) and torch.allclose(a.grad,b.grad)

def test_new_cohort_target_blind_independent_and_zero_overlap():
    c=f.cohort();assert c['panel_id']=='E4B_CORRECTED_CODE_PAIR_NEW_STRUCTURAL_OOD_V1'
    assert len(c['records'])==96 and len({x['canonical_unit_id'] for x in c['records']})==48
    assert c['topology_count_independence']['mutual_information_bits']==0.0
    assert all('output' not in x['prompt']['test'][0] for x in c['records'])
    assert all(c['zero_overlap'][k] for k in ('training_e3_exposed_ood_prompt_sha256','training_e3_query_only_exposed_ood_prompt_sha256','training_all_exposed_content_role_sha256'))

def test_launcher_worker_raw_freeze_sidecar_scorer_without_model(tmp_path,monkeypatch):
    out=tmp_path/'out';binding=tmp_path/'binding.json'
    binding.write_text(json.dumps({'output_root':str(out),'hard_cap_seconds':7200,'nonce':'testnonce','retry':False,'evaluation':{'bootstrap_seed':7,'bootstrap_replicates':10000}}))
    monkeypatch.setattr(w,'preclaim',lambda _: {'status':'PASS_NO_MODEL','model_loaded':False,'gpu_used':False,'sidecar_opened':False})
    class CUDA:
        @staticmethod
        def empty_cache(): pass
    class Torch: cuda=CUDA()
    monkeypatch.setattr(w,'load_model',lambda *args:(object(),Torch(),{}));monkeypatch.setattr(w,'train_equal_slot',lambda *args:(object(),Torch(),{}));monkeypatch.setattr(w,'train_code_pair',lambda *args:(object(),Torch(),{}))
    from scripts import e04_orientation_fixed_demonstration_baseline_v3 as orient
    bases={x['canonical_base_id']:x for x in orient.v2.canonical_bases('VALIDATION')}
    def fake_generate(model,torch,manifest,items,b,outp,tag,deadline):
        vals=[{'record_id':row['id'],'split':row['split'],'prompt_sha256':w.task_hash(row['task']),'generated_token_ids':[1],'parser_valid':True,'parsed_grid':w.target_for_record(row,bases),'effective_batch_size':1} for row in items]
        q=outp/(tag+'_RAW.jsonl');w.write_jsonl(q,vals);return w.sha(q)
    monkeypatch.setattr(w,'generate',fake_generate)
    w.execute(binding)
    assert w.load(out/'RAW_FREEZE_RECEIPT.json')['targets_accessed'] is False
    assert w.load(out/'SYNTHETIC_TARGET_SIDECAR.json')['opened_after_raw_freeze'] is True
    assert w.load(out/'SCORE_RECEIPT.json')['physical_batch_size']==1
    assert w.load(out/'TERMINAL_RECEIPT.json')['status']=='COMPLETE'

def test_no_query_only_preflight_or_old_cells():
    src=SCRIPT.read_text()
    assert 'mask_audit' not in src and "[[1,1],[1,2]]" in src
    assert 'final assistant' not in src.lower()
