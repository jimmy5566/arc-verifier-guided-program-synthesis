"""V4 R5 target-blind three-arm collector.

This revision preserves the frozen scientific inputs and decoder policy while
making runtime identity, B1 collection, nonce exclusion, and terminal evidence
executable.  Imports of model libraries remain inside ``execute_real`` only.
"""
from __future__ import annotations
import argparse, gc, hashlib, importlib, json, os, subprocess, time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update_r5_replacement_run_004'
PACKAGE = OUT / 'EXECUTION_PACKAGE.json'
LOCAL_IMPORT_GRAPH = ('inference.nvarc_native','scripts.run_unified_native_model_capability_baseline_v2')

class V4ExecutionFailure(RuntimeError): pass

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def canonical(x: Any) -> bytes: return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
def atomic_json(path: Path, value: dict[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_bytes(json.dumps(value,sort_keys=True,indent=2).encode()+b'\n'); os.replace(tmp,path)
def atomic_jsonl(path: Path, rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_bytes(b''.join(canonical(x)+b'\n' for x in rows)); os.replace(tmp,path)
def read_json(p:Path)->dict[str,Any]: return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def read_jsonl(p:Path)->list[dict[str,Any]]: return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def git(args:list[str])->str:
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True,stderr=subprocess.DEVNULL).strip()
def terminal_receipt(output:Path, **extra:Any)->None:
    atomic_json(output/'TERMINAL_RECEIPT.json',{'protocol_id':'ATOMIC_OOD_IDENTIFIABILITY_PILOT_V4_THREE_CHECKPOINT_NATIVE_GREEDY_NO_UPDATE_R5','target_sidecar_accessed':False,'optimizer_steps':0,'parameter_updates':0,'backward_calls':0,'final_audit_opened':False,**extra})

def validate_local_import_graph()->dict[str,Any]:
    """Import every worker-local runtime dependency without constructing a model.

    The outer launcher executes this actual worker entrypoint before the worker
    can validate assets, claim its nonce, create an output root, or touch CUDA.
    ROOT plus ROOT/src are the only import-path contract.
    """
    imported=[]
    for name in LOCAL_IMPORT_GRAPH:
        importlib.import_module(name); imported.append(name)
    return {'status':'PASS_V4R5_COMPLETE_LOCAL_IMPORT_GRAPH','modules':imported,'model_constructed':False,'gpu_used':False,'nonce_claimed':False,'target_sidecar_accessed':False}

def validate_preimport(package_path:Path=PACKAGE, *, output_root:Path|None=None, binding:dict[str,Any]|None=None)->dict[str,Any]:
    d=read_json(package_path)
    required={'schema_version','protocol_id','status','predecessor_package_sha256','v4_source','inputs','generation_preflight_inputs','native_runtime','three_checkpoints','decoder','batching','target_isolation','execution_output','runtime_estimate','forbidden','worker_sha256','launcher_sha256','package_fingerprint_scope'}
    if set(d) not in (required,required|{'tests_sha256'}) or d['protocol_id']!='ATOMIC_OOD_IDENTIFIABILITY_PILOT_V4_THREE_CHECKPOINT_NATIVE_GREEDY_NO_UPDATE_R5': raise V4ExecutionFailure('V4R5_PACKAGE_SCHEMA')
    if d['status']!='CPU_ONLY_FROZEN_AWAITING_DIRECTOR_EXECUTION_AUTHORIZATION': raise V4ExecutionFailure('V4R5_PACKAGE_STATUS')
    if d['worker_sha256']!=sha(Path(__file__)): raise V4ExecutionFailure('V4R5_WORKER_IDENTITY')
    if 'tests_sha256' in d and d['tests_sha256']!=sha(ROOT/'tests/test_atomic_ood_identifiability_pilot_v4_r5_execution.py'): raise V4ExecutionFailure('V4R5_TEST_IDENTITY')
    sidecar_rel=next((rel for rel in d['inputs'] if rel.endswith('/'+d['target_isolation']['sealed_target_sidecar'])),None)
    if sidecar_rel is None or d['inputs'][sidecar_rel] != d['v4_source']['sealed_target_sidecar_sha256']:
        raise V4ExecutionFailure('V4R5_SEALED_SIDECAR_METADATA')
    generation_inputs=d['generation_preflight_inputs']
    expected_generation={
        'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/MANIFEST.json',
        'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/INPUT_PROMPTS.jsonl',
    }
    if set(generation_inputs)!=expected_generation or sidecar_rel in generation_inputs:
        raise V4ExecutionFailure('V4R5_GENERATION_INPUT_SCOPE')
    # The sealed sidecar is metadata-only until the independently frozen raw
    # evidence and terminal receipt pass the post-freeze CPU release gate.
    for rel in generation_inputs:
        expected=d['inputs'][rel]
        if not (ROOT/rel).is_file() or sha(ROOT/rel)!=expected: raise V4ExecutionFailure('V4R5_FROZEN_INPUT_IDENTITY:'+rel)
    rows=read_jsonl(ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/INPUT_PROMPTS.jsonl')
    if len(rows)!=192 or len({x['episode_id'] for x in rows})!=192: raise V4ExecutionFailure('V4R5_PROMPT_COVERAGE')
    if any('target' in x['task']['test'][0] for x in rows): raise V4ExecutionFailure('V4R5_TARGET_LEAK')
    ids=d['batching']['fixed_b1_subset']['episode_ids']
    if len(ids)!=12 or len(set(ids))!=12 or set(ids)-{x['episode_id'] for x in rows}: raise V4ExecutionFailure('V4R5_B1_SUBSET')
    if d['batching']['fallback_ladder'] != [32,16,8,4,1] or d['batching']['primary_requested_batch_size']!=32: raise V4ExecutionFailure('V4R5_BATCHING')
    if d['runtime_estimate']['active_hard_cap_seconds']!=7200: raise V4ExecutionFailure('V4R5_CAP')
    if output_root is not None and output_root.exists(): raise V4ExecutionFailure('V4R5_FRESH_OUTPUT_REQUIRED')
    if binding is not None:
        if binding.get('package_sha256')!=sha(package_path) or binding.get('worker_sha256')!=d['worker_sha256'] or binding.get('launcher_sha256')!=d['launcher_sha256']: raise V4ExecutionFailure('V4R5_BINDING_HASH')
        if binding.get('authorization')!='DIRECTOR_ONCE' or binding.get('retry') is not False: raise V4ExecutionFailure('V4R5_BINDING_AUTHORIZATION')
        if binding.get('nonce')!=d['execution_output']['nonce'] or binding.get('output_root')!=str(output_root): raise V4ExecutionFailure('V4R5_BINDING_OUTPUT')
        if binding.get('active_hard_cap_seconds')!=7200: raise V4ExecutionFailure('V4R5_BINDING_CAP')
        if git(['rev-parse','HEAD'])!=binding.get('source_commit'): raise V4ExecutionFailure('V4R5_SOURCE_COMMIT')
        if git(['status','--porcelain','--untracked-files=no']): raise V4ExecutionFailure('V4R5_DIRTY_TRACKED_SOURCE')
    return {'status':'PASS_V4R5_PREIMPORT','prompt_rows':192,'arms':[x['arm_id'] for x in d['three_checkpoints']],'b1_rows_per_arm':12,'model_imported':False,'gpu_used':False,'target_sidecar_accessed':False,'optimizer_steps':0}

def validate_runtime_identity(package:dict[str,Any])->dict[str,Any]:
    """Verify every runtime identity before a model library is imported."""
    reference=package['three_checkpoints'][0]
    manifest_path=ROOT/reference['base_manifest_path']
    if not manifest_path.is_file() or sha(manifest_path)!=reference['base_manifest_sha256']:
        raise V4ExecutionFailure('V4R5_BASE_MANIFEST_IDENTITY')
    base_manifest=read_json(manifest_path); base=Path(base_manifest['base_path'])
    for entry in base_manifest['base_files']:
        path=base/entry['name']
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
            raise V4ExecutionFailure('V4R5_BASE_IDENTITY:'+entry['name'])
    adapter=Path(base_manifest['adapter_path'])
    for entry in base_manifest['adapter_files']:
        path=adapter/entry['name']
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
            raise V4ExecutionFailure('V4R5_V7_ADAPTER_IDENTITY:'+entry['name'])
    runtime=package['native_runtime']; identity_path=ROOT/runtime['config_identity_path']
    if not identity_path.is_file() or sha(identity_path)!=runtime['config_identity_sha256']:
        raise V4ExecutionFailure('V4R5_NATIVE_CONFIG_IDENTITY')
    identity=read_json(identity_path); config=ROOT/'configs'/identity['runtime_config_dir_name']
    for entry in identity['files']:
        path=config/entry['runtime_relative_path']
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
            raise V4ExecutionFailure('V4R5_NATIVE_CONFIG_FILE:'+entry['runtime_relative_path'])
    return {'base_manifest':base_manifest,'base':base,'adapter':adapter,'native_config_dir':config}

def claim_nonce(package:dict[str,Any], output:Path)->Path:
    ledger=ROOT/'.arc2-local'/'runtime'/'v4_active_nonces'; ledger.mkdir(parents=True,exist_ok=True)
    path=ledger/(package['execution_output']['nonce']+'.json')
    try:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL); os.write(fd,canonical({'nonce':package['execution_output']['nonce'],'output_root':str(output),'status':'ACTIVE'})); os.close(fd)
    except FileExistsError as e: raise V4ExecutionFailure('V4R5_DUPLICATE_NONCE') from e
    return path

def finalize_nonce(path:Path,status:str)->None: atomic_json(path,{'status':status,'nonce':path.stem})
def as_record(arm:str,row:dict[str,Any],phase:str,result:dict[str,Any],batch:int)->dict[str,Any]:
    ids=[int(x) for x in result['generated_token_ids']]; grid=result.get('parsed_grid')
    return {'arm_id':arm,'episode_id':row['episode_id'],'phase':phase,'input_sha256':hashlib.sha256(canonical(row)).hexdigest(),'generated_token_ids':ids,'decoded_text':str(result['decoded_text']),'parser_valid':bool(result['parser_valid']),'parsed_grid':grid,'canonical_grid_sha256':hashlib.sha256(canonical(grid)).hexdigest() if grid is not None else None,'prompt_tokens':int(result['prompt_tokens']),'completion_tokens':len(ids),'effective_batch_size':batch}

def length_bucketed_primary(prompts:list[dict[str,Any]], token_length:Callable[[dict[str,Any]],int])->list[dict[str,Any]]:
    return sorted(prompts,key=lambda row:(int(token_length(row)),str(row['episode_id'])))

def collect_with_generator(package_path:Path, output:Path, generator:Callable[[str,list[dict[str,Any]],int],list[dict[str,Any]]], *, primary_rows:list[dict[str,Any]]|None=None, after_arm:Callable[[str],None]|None=None, clock:Callable[[],float]=time.monotonic)->dict[str,Any]:
    package=read_json(package_path); prompts=read_jsonl(ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/INPUT_PROMPTS.jsonl'); primary_prompts=prompts if primary_rows is None else primary_rows; selected=set(package['batching']['fixed_b1_subset']['episode_ids']); deadline=clock()+7200; partial=[]
    if len(primary_prompts)!=192 or {x['episode_id'] for x in primary_prompts}!={x['episode_id'] for x in prompts}: raise V4ExecutionFailure('V4R5_PRIMARY_ORDER_COVERAGE')
    for arm in package['three_checkpoints']:
        primary=[]
        cursor,size=0,32
        while cursor < len(primary_prompts):
            if clock()>=deadline: raise V4ExecutionFailure('V4R5_ACTIVE_CAP_EXCEEDED')
            batch=primary_prompts[cursor:cursor+size]
            try:
                out=generator(arm['arm_id'],batch,len(batch))
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower() or len(batch)==1: raise
                size=next(x for x in (16,8,4,1) if x < len(batch)); continue
            if len(out)!=len(batch): raise V4ExecutionFailure('V4R5_GENERATOR_CARDINALITY')
            primary.extend(as_record(arm['arm_id'],r,'PRIMARY',v,len(batch)) for r,v in zip(batch,out))
            cursor += len(batch)
        b1rows=[]
        for row in prompts:
            if row['episode_id'] not in selected: continue
            if clock()>=deadline: raise V4ExecutionFailure('V4R5_ACTIVE_CAP_EXCEEDED')
            out=generator(arm['arm_id'],[row],1)
            if len(out)!=1: raise V4ExecutionFailure('V4R5_GENERATOR_CARDINALITY')
            b1rows.append(as_record(arm['arm_id'],row,'B1_SENSITIVITY',out[0],1))
        if len(primary)!=192 or len(b1rows)!=12: raise V4ExecutionFailure('V4R5_COLLECTION_CARDINALITY')
        rows=primary+b1rows; name=arm['arm_id']+'_RAW_TARGET_BLIND.jsonl'; atomic_jsonl(output/name,rows); partial.append({'arm_id':arm['arm_id'],'path':name,'sha256':sha(output/name),'primary_rows':192,'b1_rows':12})
        if after_arm is not None:
            after_arm(arm['arm_id'])
    return {'raw_evidence':partial,'raw_rows':sum(x['primary_rows']+x['b1_rows'] for x in partial),'b1_distinct_rows':36}

def execute_real(package_path:Path,runtime_binding_path:Path,output:Path)->dict[str,Any]:
    package=read_json(package_path)
    runtime=validate_runtime_identity(package)
    # Dynamic imports establish the model boundary only after static identity preflight.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from inference.nvarc_native import checkpoint_native_tokenizer, parse_native_grid
    from scripts.run_unified_native_model_capability_baseline_v2 import prompt
    from huggingface_hub import hf_hub_download
    base_manifest=runtime['base_manifest']; base=runtime['base']; tokenizer,_=checkpoint_native_tokenizer(base,runtime['native_config_dir']); loaded={}
    prompts=read_jsonl(ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/INPUT_PROMPTS.jsonl')
    primary_rows=length_bucketed_primary(prompts,lambda row:len(tokenizer(prompt(row['task']),add_special_tokens=False)['input_ids']))
    def generator(arm_id:str,batch:list[dict[str,Any]],batch_size:int)->list[dict[str,Any]]:
        if arm_id not in loaded:
            arm=next(x for x in package['three_checkpoints'] if x['arm_id']==arm_id)
            if arm_id=='V7': adapter=runtime['adapter']
            else:
                adapter=output/'_adapter_cache'/arm_id; adapter.mkdir(parents=True,exist_ok=True)
                for filename,expected in (('adapter_model.safetensors',arm['adapter_model_sha256']),('adapter_config.json',arm['adapter_config_sha256'])):
                    got=Path(hf_hub_download(arm['repository']['repo_id'],filename=arm['repository_path'].rsplit('/',1)[0]+'/'+filename,repo_type='model',revision=arm['repository']['immutable_revision'],token=True,local_dir=str(adapter),force_download=True))
                    if sha(got)!=expected: raise V4ExecutionFailure('V4R5_ADAPTER_IDENTITY:'+arm_id)
                    if got != adapter/filename: os.replace(got,adapter/filename)
            loaded[arm_id]=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0'),str(adapter),is_trainable=False).eval()
        encoded=tokenizer([prompt(x['task']) for x in batch],return_tensors='pt',padding=True,add_special_tokens=False); encoded={k:v.to('cuda:0') for k,v in encoded.items()}; width=int(encoded['input_ids'].shape[1])
        with torch.inference_mode(): generated=loaded[arm_id].generate(**encoded,do_sample=False,num_beams=1,max_new_tokens=128,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
        out=[]
        for i in range(len(batch)):
            ids=generated[i,width:].detach().cpu().tolist(); text=tokenizer.decode(ids,skip_special_tokens=True); grid=parse_native_grid(text)
            out.append({'generated_token_ids':ids,'decoded_text':text,'parser_valid':grid is not None,'parsed_grid':[list(x) for x in grid] if grid is not None else None,'prompt_tokens':int(encoded['attention_mask'][i].sum().item())})
        return out
    def release(arm_id:str)->None:
        # Release only after the arm's complete primary and B1 raw evidence
        # has been atomically frozen; never reload a completed arm.
        del loaded[arm_id]
        gc.collect()
        torch.cuda.empty_cache()
    result=collect_with_generator(package_path,output,generator,primary_rows=primary_rows,after_arm=release)
    loaded.clear(); torch.cuda.empty_cache(); return result

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--preflight-import-graph',action='store_true'); ap.add_argument('--package',type=Path,default=PACKAGE); ap.add_argument('--runtime-binding',type=Path); ap.add_argument('--output-root',type=Path); a=ap.parse_args()
    if a.preflight_import_graph: print(json.dumps(validate_local_import_graph(),sort_keys=True)); return
    if a.self_test: print(json.dumps(validate_preimport(a.package),sort_keys=True)); return
    if not a.runtime_binding or not a.output_root: raise SystemExit('RUNTIME_BINDING_AND_OUTPUT_ROOT_REQUIRED')
    binding=read_json(a.runtime_binding); started=time.monotonic(); claimed=None
    try:
        preflight=validate_preimport(a.package,output_root=a.output_root,binding=binding); claimed=claim_nonce(read_json(a.package),a.output_root); a.output_root.mkdir(parents=True); result=execute_real(a.package,a.runtime_binding,a.output_root); terminal_receipt(a.output_root,status='COMPLETE_NO_UPDATE_TARGET_BLIND',runtime_binding_sha256=sha(a.runtime_binding),preflight=preflight,wall_seconds=time.monotonic()-started,**result); finalize_nonce(claimed,'COMPLETE')
    except Exception as e:
        a.output_root.mkdir(parents=True,exist_ok=True)
        if not (a.output_root/'TERMINAL_RECEIPT.json').exists(): terminal_receipt(a.output_root,status='FAILED_NO_UPDATE_INFRASTRUCTURE',error=str(e),error_type=type(e).__name__,partial_evidence=[],wall_seconds=time.monotonic()-started)
        if claimed: finalize_nonce(claimed,'FAILED')
        raise
if __name__=='__main__': main()
