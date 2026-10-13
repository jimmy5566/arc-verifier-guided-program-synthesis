import importlib.util, json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/run_e3_batch1_only_measurement_successor_v1.py'
spec=importlib.util.spec_from_file_location('e3_batch1',SCRIPT);worker=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(worker)
FREEZER=ROOT/'scripts/freeze_e3_batch1_only_measurement_successor_v1.py'
spec2=importlib.util.spec_from_file_location('freeze_e3_batch1',FREEZER);freezer=importlib.util.module_from_spec(spec2);assert spec2.loader;spec2.loader.exec_module(freezer)

def _mock_grid(row):
 from scripts.connected_components_object_extraction_semantic_oracle import target_for_family
 from scripts import e04_orientation_fixed_demonstration_baseline_v3 as orient
 if row['split'] in {'OOD_INDEPENDENT','TRAIN_FIT_SECONDARY'}: return target_for_family('CONNECTED_COMPONENT_COUNT',row['task']['test'][0]['input'])
 if row['split']=='COMPOSITION_SECONDARY': return target_for_family('COUNT_THEN_OBJECT_EXTRACTION',row['task']['test'][0]['input'])
 q=row['protected'];bases={x['canonical_base_id']:x for x in orient.v2.canonical_bases('VALIDATION')};return orient.target(bases[q['canonical_base_id']],q['control_marker_turn'],q['condition'],q['source_row_index'])['target']

def _raw(worker,out,tag,arm,all_ood_wrong=False):
 rows=[]
 for r in worker.records(arm):
  wrong=all_ood_wrong and r['split']=='OOD_INDEPENDENT'
  rows.append({'record_id':r['id'],'prompt_sha256':hashlib.sha256(worker.canon(r['task'])).hexdigest(),'parser_valid':not wrong,'parsed_grid':None if wrong else _mock_grid(r),'generated_token_ids':[1],'effective_batch_size':1})
 p=out/(tag+'_RAW.jsonl');worker.write_jsonl(p,rows);return worker.sha(p)

def test_independent_cohort_is_target_blind_and_zero_overlap():
 c=freezer.cohort();assert len(c['records'])==96;assert len({x['canonical_unit_id'] for x in c['records']})==48
 assert c['zero_overlap']['training_and_exposed_ood_prompt_sha256'];assert c['zero_overlap']['training_and_exposed_ood_query_input_sha256']
 assert all('output' not in x['prompt']['test'][0] for x in c['records'])

def test_batch1_sidecar_and_scorer_accept_exact_grid_mocks(tmp_path):
 out=tmp_path/'out';out.mkdir();binding=tmp_path/'b.json';binding.write_text(json.dumps({'evaluation':{'bootstrap_seed':1,'bootstrap_replicates':10000}}))
 worker.write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':{},'targets_accessed':False})
 sidecar=worker.make_sidecar(binding,out);raw={}
 for tag,arm in [('V7',None),('CONTROL_ORIGINAL_CONFOUNDED','CONTROL_ORIGINAL_CONFOUNDED'),('TREATMENT_DECONFOUNDED','TREATMENT_DECONFOUNDED')]:raw[tag]=_raw(worker,out,tag,arm)
 worker.write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':raw,'targets_accessed':False})
 result=worker.score(binding,out,sidecar);assert result['physical_batch_size']==1;assert result['batch32_used'] is False;assert result['primary_independent_ood']['canonical_units']==48

def test_execute_orders_raw_freeze_before_sidecar_with_no_model_mocks(tmp_path,monkeypatch):
 out=tmp_path/'out';binding=tmp_path/'b.json';binding.write_text(json.dumps({'output_root':str(out),'hard_cap_seconds':3600,'nonce':'n','retry':False,'evaluation':{'bootstrap_seed':1,'bootstrap_replicates':10000}}))
 monkeypatch.setattr(worker,'preclaim',lambda _: {'status':'PASS','model_loaded':False,'gpu_used':False,'sidecar_opened':False})
 monkeypatch.setattr(worker,'load_model',lambda b,tag:(object(),object(),{}))
 seen=[]
 def fake_generate(model,torch,manifest,items,b,outp,tag,deadline):
  seen.append(tag);rows=[]
  for r in items:rows.append({'record_id':r['id'],'prompt_sha256':hashlib.sha256(worker.canon(r['task'])).hexdigest(),'parser_valid':True,'parsed_grid':_mock_grid(r),'generated_token_ids':[1],'effective_batch_size':1})
  p=outp/(tag+'_RAW.jsonl');worker.write_jsonl(p,rows);return worker.sha(p)
 monkeypatch.setattr(worker,'generate',fake_generate)
 class CUDA:
  @staticmethod
  def empty_cache():pass
 class Torch:cuda=CUDA()
 monkeypatch.setattr(worker,'load_model',lambda b,tag:(object(),Torch(),{}))
 worker.execute(binding)
 assert seen==['V7','CONTROL_ORIGINAL_CONFOUNDED','TREATMENT_DECONFOUNDED']
 assert worker.load(out/'RAW_FREEZE_RECEIPT.json')['targets_accessed'] is False
 assert worker.load(out/'TERMINAL_RECEIPT.json')['status']=='COMPLETE'

def test_score_handles_all_wrong_ood_and_post_raw_repair_without_model(tmp_path):
 out=tmp_path/'out';out.mkdir();binding=tmp_path/'b.json';binding.write_text(json.dumps({'output_root':str(out),'evaluation':{'bootstrap_seed':1,'bootstrap_replicates':10000}}))
 worker.write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':{},'targets_accessed':False})
 sidecar=worker.make_sidecar(binding,out);raw={}
 for tag,arm in [('V7',None),('CONTROL_ORIGINAL_CONFOUNDED','CONTROL_ORIGINAL_CONFOUNDED'),('TREATMENT_DECONFOUNDED','TREATMENT_DECONFOUNDED')]:raw[tag]=_raw(worker,out,tag,arm,all_ood_wrong=True)
 worker.write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':raw,'targets_accessed':False})
 worker.write(out/'TERMINAL_RECEIPT.json',{'status':'FAILED_OR_PARTIAL_NO_RETRY','error':"KeyError:'canonical'"})
 repair=worker.score_only(binding)
 assert repair['status']=='COMPLETE_POST_RAW_SCORER_REPAIR_NO_MODEL'
 assert worker.load(out/'SCORE_RECEIPT.json')['decision']=='REJECT'
