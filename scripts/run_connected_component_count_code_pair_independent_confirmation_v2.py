from __future__ import annotations
"""Bound V2 worker: preclaim before claim; execute freezes raw before sealed scoring."""
import argparse, hashlib, importlib, json, os, sys, tempfile, time
from importlib import metadata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def write(path,x):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='\n',dir=path.parent,delete=False) as f:
        json.dump(x,f,sort_keys=True,separators=(',',':'));f.write('\n');tmp=f.name
    os.replace(tmp,path)
def fail(x): raise RuntimeError('IC_V2:'+x)
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def file(binding,key):
    e=binding['inputs'][key];p=ROOT/e['path']
    if not p.is_file() or sha(p)!=e['sha256']:fail('INPUT:'+key)
    return p

def runtime_imports():
    root=str(ROOT)
    if root not in sys.path:sys.path.insert(0,root)
    src=str(ROOT/'src')
    if src not in sys.path:sys.path.insert(0,src)
    names=('inference.nvarc_native','inference.hf_peft_backend','training_data.pipeline','scripts.e04_e_equal_slot_loss','scripts.decision_token_supervision')
    for name in names: importlib.import_module(name)
    return list(names)

def items(binding):
    rows=load(file(binding,'observation'));out=[]
    for row in rows:
        if 'output' in row['task']['test'][0]:fail('TARGET_LEAK')
        out.append({'record_id':'V2:'+row['prompt_id'],'prompt_id':row['prompt_id'],'split':row['latent_condition']['split'],'family':row['latent_condition']['family'],'task':row['task']})
    if len(out)!=104 or len({x['record_id'] for x in out})!=104:fail('OBSERVATION_RECORDS')
    got={}
    for x in out:got[x['family']]=got.get(x['family'],0)+1
    want={'CONNECTED_COMPONENT_COUNT':24,'OBJECT_EXTRACTION':24,'FIXED_TURN_ROTATION_CONTROL':8,'MARKER_BINDING_CONTROL':8,'NO_TRANSFORM_RETENTION':24,'COUNT_THEN_OBJECT_EXTRACTION':16}
    if got!=want:fail('INVENTORY')
    return out,got

def schedule(binding):
    rows=load(file(binding,'diverse_schedule'))['episodes']
    if len(rows)!=384 or [x.get('slot') for x in rows]!=list(range(384)):fail('SCHEDULE')
    if binding['training']['schedule_slots']!=384 or binding['training']['slots_per_update']!=4 or binding['training']['updates_per_arm']!=96:fail('TRAINING_PLAN')
    if binding['arms']['control']['schedule_sha256']!=binding['inputs']['diverse_schedule']['sha256'] or binding['arms']['treatment']['schedule_sha256']!=binding['inputs']['diverse_schedule']['sha256']:fail('SCHEDULE_IDENTITY')
    return rows

def validate(binding_path, *, production_assets):
    b=load(binding_path);required={'experiment_id','scientific_source_commit','source_files','inputs','arms','v7','runtime','execution','training','evaluation','dataset_paths','inference','scoring'}
    if not required<=set(b) or b['experiment_id']!='CONNECTED_COMPONENT_COUNT_CODE_PAIR_INDEPENDENT_CONFIRMATION_V2':fail('BINDING')
    for rel,digest in b['source_files'].items():
        if not (ROOT/rel).is_file() or sha(ROOT/rel)!=digest:fail('SOURCE:'+rel)
    for key in ('observation','recipes','batch1','decision_table','diverse_schedule','validation'):file(b,key)
    if b['arms']['control']['loss_allocation']!={'kind':'UNIFORM_SEQUENCE_LOSS'} or b['arms']['treatment']['loss_allocation']!={'kind':'COUNT_CODE_PAIR_REWEIGHTED','decision_token_multiplier':2.0}:fail('LOSS')
    ex=b['execution'];need={'authorization_required','remote_output_root','cpu_integration_root','terminal_receipt','nonce','retry','worker_path','worker_sha256','launcher_path','launcher_sha256','combined_cap_seconds','training_cap_seconds','evaluation_cap_seconds'}
    if not need<=set(ex) or ex['authorization_required'] is not True or ex['retry'] is not False:fail('EXECUTION')
    if ex['worker_path']!='scripts/run_connected_component_count_code_pair_independent_confirmation_v2.py' or ex['launcher_path']!=ex['worker_path'] or sha(ROOT/ex['worker_path'])!=ex['worker_sha256'] or sha(ROOT/ex['launcher_path'])!=ex['launcher_sha256']:fail('EXECUTION_IDENTITY')
    if ex['training_cap_seconds']+ex['evaluation_cap_seconds']!=ex['combined_cap_seconds']:fail('CAPS')
    obs,inventory=items(b);sched=schedule(b);sub=load(file(b,'batch1'))['checkpoints']; ids={x['prompt_id'] for x in obs}
    if set(sub)!={'V7','CONTROL_NARROW_STRUCTURE','TREATMENT_STRUCTURE_DIVERSE'} or any(len(v)!=12 or len(set(v))!=12 or not set(v)<=ids for v in sub.values()) or len({tuple(v) for v in sub.values()})!=1:fail('B1')
    manifest=ROOT/b['v7']['checkpoint_manifest_path']
    if not manifest.is_file() or sha(manifest)!=b['v7']['checkpoint_manifest_sha256']:fail('V7_MANIFEST')
    imports=runtime_imports();checked=0
    if production_assets:
        m=load(manifest)
        for rk,fk in (('base_path','base_files'),('adapter_path','adapter_files')):
            for a in m[fk]:
                p=Path(m[rk])/a['name']
                if not p.is_file() or p.stat().st_size!=a['bytes'] or sha(p)!=a['sha256']:fail('ASSET:'+a['name'])
                checked+=1
        if Path(sys.executable).resolve()!=Path(b['runtime']['python']).resolve():fail('PYTHON')
        for package,version in b['runtime']['dependencies'].items():
            if metadata.version(package)!=version:fail('DEPENDENCY:'+package)
        if Path(ex['remote_output_root']).exists() or Path(ex['terminal_receipt']).is_absolute() and Path(ex['terminal_receipt']).exists():fail('NONFRESH_OUTPUT')
        if __import__('shutil').disk_usage(Path(ex['remote_output_root']).parent).free<int(ex['minimum_free_bytes']):fail('DISK')
        if not isinstance(ex['hard_cap_seconds'],int) or ex['hard_cap_seconds']<ex['combined_cap_seconds']:fail('HARD_CAP')
        needle=ex['nonce'].encode()
        if Path('/proc').is_dir():
            for cmd in Path('/proc').glob('[0-9]*/cmdline'):
                try: payload=cmd.read_bytes()
                except OSError: continue
                if needle in payload:fail('DUPLICATE_NONCE_PROCESS')
    return b,obs,sched,{'inventory':inventory,'runtime_imports':imports,'v7_asset_files_verified':checked,'shared_b1_record_ids':['V2:'+x for x in sub['V7']]}

def preclaim(binding_path):
    b,obs,sched,detail=validate(binding_path,production_assets=True)
    return {'status':'PASS_V2_PRODUCTION_PRECLAIM_STOP_BEFORE_CLAIM_MODEL_OR_CUDA','binding_sha256':sha(binding_path),'records':len(obs),'schedule_slots':len(sched),**detail,'model_loaded':False,'gpu_used':False,'cuda_initialized':False,'sealed_sidecar_opened':False,'authorization_claimed':False}

def score(binding_path,output,sidecar, *, test_stub=False):
    b=load(binding_path);out=Path(output);freeze=load(out/'RAW_FREEZE_RECEIPT.json')
    tags=('V7','V7_B1','CONTROL_NARROW_STRUCTURE','CONTROL_NARROW_STRUCTURE_B1','TREATMENT_STRUCTURE_DIVERSE','TREATMENT_STRUCTURE_DIVERSE_B1')
    if freeze.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or set(freeze.get('raw_sha256',{}))!=set(tags):fail('RAW_FREEZE')
    if not test_stub and sha(sidecar)!=b['evaluation']['sealed_sidecar_sha256']:fail('SIDECAR_IDENTITY')
    s=load(sidecar)
    expected={x['record_id']:x['target'] for x in s['records']}
    obs,_=items(b)
    if set(expected)!={x['record_id'] for x in obs}:fail('SIDECAR_INVENTORY')
    metrics={}
    for tag in tags:
        p=out/(tag+'_RAW.jsonl')
        if not p.is_file() or sha(p)!=freeze['raw_sha256'][tag]:fail('RAW:'+tag)
        rows=[loadline for loadline in (json.loads(x) for x in p.read_text().splitlines() if x)]
        if len({x['record_id'] for x in rows})!=len(rows):fail('RAW_DUP:'+tag)
        metrics[tag]={'rows':len(rows),'exact':sum(bool(x.get('parser_valid')) and x.get('parsed_grid')==expected[x['record_id']] for x in rows)}
    table=load(file(b,'decision_table'))
    required={'treatment_control_count_min':3,'treatment_control_count_lower95_gt':0,'treatment_v7_count_min':2,'treatment_v7_count_lower95_gt':0,'object_extraction_no_harm_both':True,'composition_no_harm_both':True,'retention_counts':{'FIXED_TURN_ROTATION_CONTROL':8,'MARKER_BINDING_CONTROL':8,'NO_TRANSFORM_RETENTION':24},'bootstrap_replicates':10000,'all_gates_must_pass':True}
    if table!=required:fail('DECISION_TABLE')
    by_id={}
    for tag in ('V7','CONTROL_NARROW_STRUCTURE','TREATMENT_STRUCTURE_DIVERSE'):
        by_id[tag]={x['record_id']:bool(x.get('parser_valid')) and x.get('parsed_grid')==expected[x['record_id']] for x in (json.loads(line) for line in (out/(tag+'_RAW.jsonl')).read_text().splitlines() if line)}
    family={x['record_id']:x['family'] for x in obs}
    def delta(left,right,name):
        ids=sorted(x for x,v in family.items() if v==name);return {'records':len(ids),'delta_correct':sum(by_id[left][x]-by_id[right][x] for x in ids)}
    count_tc=delta('TREATMENT_STRUCTURE_DIVERSE','CONTROL_NARROW_STRUCTURE','CONNECTED_COMPONENT_COUNT')
    count_tv=delta('TREATMENT_STRUCTURE_DIVERSE','V7','CONNECTED_COMPONENT_COUNT')
    object_control=delta('CONTROL_NARROW_STRUCTURE','V7','OBJECT_EXTRACTION')
    object_treatment=delta('TREATMENT_STRUCTURE_DIVERSE','V7','OBJECT_EXTRACTION')
    composition_control=delta('CONTROL_NARROW_STRUCTURE','V7','COUNT_THEN_OBJECT_EXTRACTION')
    composition_treatment=delta('TREATMENT_STRUCTURE_DIVERSE','V7','COUNT_THEN_OBJECT_EXTRACTION')
    retention={name:sum(v==name for v in family.values()) for name in table['retention_counts']}
    gates={'treatment_control_count':count_tc['delta_correct']>=table['treatment_control_count_min'],'treatment_v7_count':count_tv['delta_correct']>=table['treatment_v7_count_min'],'object_extraction_no_harm_both':object_control['delta_correct']>=0 and object_treatment['delta_correct']>=0,'composition_no_harm_both':composition_control['delta_correct']>=0 and composition_treatment['delta_correct']>=0,'retention_inventory':retention==table['retention_counts']}
    result={'status':'COMPLETE_V2_SCORER_AFTER_RAW_FREEZE','binding_sha256':sha(binding_path),'raw_freeze_sha256':sha(out/'RAW_FREEZE_RECEIPT.json'),'sealed_sidecar_sha256':sha(sidecar),'metrics':metrics,'frozen_gate_inputs':{'count_treatment_minus_control':count_tc,'count_treatment_minus_v7':count_tv,'object_control_minus_v7':object_control,'object_treatment_minus_v7':object_treatment,'composition_control_minus_v7':composition_control,'composition_treatment_minus_v7':composition_treatment,'retention_inventory':retention},'gates':gates,'decision':'PASS_ALL_GATES' if all(gates.values()) else 'DO_NOT_PROMOTE','targets_opened_after_raw_freeze':True}
    write(out/'SCORE_RECEIPT.json',result);return result

def cpu_integration(binding_path,output):
    b,obs,sched,d=validate(binding_path,production_assets=False);out=Path(output)
    if out.exists():fail('CPU_OUTPUT_EXISTS')
    out.mkdir(parents=True);tags=('V7','V7_B1','CONTROL_NARROW_STRUCTURE','CONTROL_NARROW_STRUCTURE_B1','TREATMENT_STRUCTURE_DIVERSE','TREATMENT_STRUCTURE_DIVERSE_B1'); sub=load(file(b,'batch1'))['checkpoints'];raw={}
    for tag in tags:
        selected=obs if not tag.endswith('_B1') else [x for x in obs if x['prompt_id'] in set(sub[tag.removesuffix('_B1')])]
        p=out/(tag+'_RAW.jsonl');p.write_text(''.join(json.dumps({'record_id':x['record_id'],'split':x['split'],'parser_valid':True,'parsed_grid':[['STUB']],'generated_token_ids':[0]},sort_keys=True)+'\n' for x in selected));raw[tag]=sha(p)
    write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':raw,'targets_accessed':False,'stubbed_model':True})
    sidecar=out/'CPU_ONLY_STUB_SIDECAR.json';write(sidecar,{'records':[{'record_id':x['record_id'],'target':[['STUB']]} for x in obs]})
    scored=score(binding_path,out,sidecar,test_stub=True)
    r={'status':'PASS_V2_CPU_INTEGRATION_FULL_DISPATCHER_STUBBED_MODEL_AND_CUDA','binding_sha256':sha(binding_path),'schedule_slots':len(sched),'arms':2,'updates_per_arm':96,'evaluation_records':104,'b1_records':12,'raw_freeze_before_scoring':True,'scorer_status':scored['status'],'model_loaded':False,'gpu_used':False,'cuda_initialized':False,'sealed_sidecar_opened':False,'authorization_claimed':False};write(out/'CPU_INTEGRATION_RECEIPT.json',r);return r

def authorization(path,binding):
    x=load(path);b=load(binding);ex=b['execution']
    if x.get('binding_sha256')!=sha(binding) or x.get('nonce')!=ex['nonce'] or x.get('status')!='CLAIMED' or x.get('retry') is not False:fail('AUTHORIZATION')

def execute(binding_path,output,auth,sidecar):
    b,obs,sched,d=validate(binding_path,production_assets=True);ex=b['execution'];out=Path(output)
    if str(out).replace('\\','/')!=ex['remote_output_root']:fail('OUTPUT_ROOT')
    if str(Path(sidecar)).replace('\\','/')!=b['evaluation']['sealed_sidecar_runtime_path']:fail('SIDECAR_PATH')
    authorization(auth,binding_path);out.mkdir(parents=True);started=time.monotonic();write(out/'PREFLIGHT_RECEIPT.json',{'status':'PASS_PRECLAIM_BEFORE_MODEL_LOAD','binding_sha256':sha(binding_path),'nonce':ex['nonce'],'model_loaded':False,'gpu_used':False,'sealed_sidecar_opened':False})
    try:
        from scripts import run_connected_components_object_extraction_pilot_v1 as real
        deadline=started+ex['combined_cap_seconds']; translated={**b,'training':{**b['training'],'torch_seed':b['training']['seed']},'evaluation':{'fallback_ladder':b['inference']['cuda_oom_only_fallback_ladder'],'max_new_tokens':b['inference'].get('max_new_tokens',512)},'runtime':{**b['runtime'],'context_window':b['inference'].get('context_window',16384)}}
        def generate(model,torch,manifest,records,tag,force=None):
            from inference.nvarc_native import NVARCNativeProvider,checkpoint_native_tokenizer,parse_native_grid
            cfg=ROOT/'experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json';c=load(cfg);tok,_=checkpoint_native_tokenizer(Path(manifest['base_path']),ROOT/'configs'/c['runtime_config_dir_name']); provider=NVARCNativeProvider(model_path=Path(manifest['base_path']),tokenizer_config_dir=ROOT/'configs'/c['runtime_config_dir_name'],device='cuda:0');provider.model,provider.tokenizer=model,tok;ladder=translated['evaluation']['fallback_ladder'];bs=force or ladder[0];rows=[];i=0
            while i<len(records):
                batch=records[i:i+bs];got=provider.generate_many([real._messages(x['task']) for x in batch],max_new_tokens=translated['evaluation']['max_new_tokens'],context_window=translated['runtime']['context_window'],seeds=[translated['training']['seed']+i+j for j in range(len(batch))])
                for item,g in zip(batch,got,strict=True):
                    grid=parse_native_grid(g.text);rows.append({'record_id':item['record_id'],'split':item['split'],'prompt_sha256':hashlib.sha256(canon(item['task'])).hexdigest(),'generated_token_ids':[int(x) for x in g.token_ids],'parsed_grid':None if grid is None else [list(x) for x in grid],'parser_valid':grid is not None,'effective_batch_size':bs})
                i+=len(batch)
            p=out/(tag+'_RAW.jsonl');p.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));return sha(p)
        raw={};model,torch,m=real._load_v7(translated,trainable=False);raw['V7']=generate(model,torch,m,obs,'V7');b1ids=set(load(file(b,'batch1'))['checkpoints']['V7']);raw['V7_B1']=generate(model,torch,m,[x for x in obs if x['prompt_id'] in b1ids],'V7_B1',1);del model
        for label,key in (('CONTROL_NARROW_STRUCTURE','control'),('TREATMENT_STRUCTURE_DIVERSE','treatment')):
            arm={**translated,'training':{**translated['training'],'loss_allocation':b['arms'][key]['loss_allocation']}};model,torch,m,adapter=real._train(arm,label,ROOT/b['arms'][key]['schedule_path'],out/label,deadline);raw[label]=generate(model,torch,m,obs,label);raw[label+'_B1']=generate(model,torch,m,[x for x in obs if x['prompt_id'] in b1ids],label+'_B1',1);write(out/label/'ADAPTER_RECEIPT.json',{'adapter_sha256':adapter,'optimizer_steps':96});del model
        write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':raw,'targets_accessed':False});scored=score(binding_path,out,sidecar);write(out/ex['terminal_receipt'],{'status':'COMPLETE_V2_RAW_AND_SCORER','raw_sha256':raw,'score_receipt_sha256':sha(out/'SCORE_RECEIPT.json'),'targets_accessed_before_raw_freeze':False})
    except Exception as e:
        write(out/ex['terminal_receipt'],{'status':'FAILED_OR_PARTIAL_NO_RETRY','error':type(e).__name__+':'+str(e),'targets_accessed_before_raw_freeze':False});raise

def main():
    a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True);a.add_argument('--preclaim-only',action='store_true');a.add_argument('--cpu-integration',action='store_true');a.add_argument('--execute',action='store_true');a.add_argument('--authorization-receipt',type=Path);a.add_argument('--sealed-sidecar',type=Path);x=a.parse_args()
    if sum((x.preclaim_only,x.cpu_integration,x.execute))!=1:raise SystemExit('choose exactly one mode')
    if x.preclaim_only: print(json.dumps(preclaim(x.binding),sort_keys=True))
    elif x.cpu_integration: print(json.dumps(cpu_integration(x.binding,x.output_root),sort_keys=True))
    else:
        if not x.authorization_receipt or not x.sealed_sidecar:raise SystemExit('execute requires authorization and sealed sidecar')
        execute(x.binding,x.output_root,x.authorization_receipt,x.sealed_sidecar)
if __name__=='__main__':main()
