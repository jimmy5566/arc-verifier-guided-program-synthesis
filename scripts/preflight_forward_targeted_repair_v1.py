#!/usr/bin/env python3
"""Read-only machine admission for a frozen forward targeted-repair round."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()

def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)

def verify_tree(root: Path, entries: list[dict], label: str) -> dict:
    expected={str(x['name']):x for x in entries}
    observed={str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*') if p.is_file()}
    if observed != set(expected):
        raise RuntimeError(f'{label}_ALLOWLIST_MISMATCH:missing={sorted(set(expected)-observed)}:extra={sorted(observed-set(expected))}')
    verified=[]
    for name, item in sorted(expected.items()):
        path=root/name
        if path.stat().st_size != int(item['bytes']): raise RuntimeError(f'{label}_SIZE_MISMATCH:{name}')
        actual=sha256(path)
        if actual != item['sha256']: raise RuntimeError(f'{label}_SHA256_MISMATCH:{name}')
        verified.append({'name':name,'bytes':path.stat().st_size,'sha256':actual})
    return {'root':str(root),'files':verified}

def validate(config_path: Path, run_root: Path) -> dict:
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if run_root.exists(): raise RuntimeError('FRESH_OUTPUT_REQUIRED')
    if config.get('final_audit_opened') is not False: raise RuntimeError('FINAL_AUDIT_FORBIDDEN')
    if config.get('retention_sentinel_usage')!='EVALUATION_ONLY_NOT_TRAINING_DATA': raise RuntimeError('RETENTION_BOUNDARY_INVALID')
    if abs(sum(float(v) for v in config['role_weights'].values())-1.0)>1e-9: raise RuntimeError('ROLE_WEIGHTS_INVALID')
    if sha256(Path(config['train_path'])) != config['train_sha256']: raise RuntimeError('TRAIN_IDENTITY_FAIL')
    manifest_path=Path(config['checkpoint_manifest_path'])
    if sha256(manifest_path)!=config['checkpoint_manifest_sha256']: raise RuntimeError('CHECKPOINT_MANIFEST_FILE_SHA256_MISMATCH')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    return {'config_sha256':sha256(config_path),'train_sha256':config['train_sha256'],'checkpoint_manifest_sha256':config['checkpoint_manifest_sha256'],'base':verify_tree(Path(manifest['base_path']),manifest['base_files'],'BASE'),'adapter':verify_tree(Path(manifest['adapter_path']),manifest['adapter_files'],'ADAPTER'),'round_id':config['round_id'],'output_root':str(run_root)}

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--repo-root',type=Path,required=True);parser.add_argument('--expected-source-commit',required=True)
    parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--skip-runtime-imports',action='store_true')
    args=parser.parse_args(); receipt={'schema_version':1,'round_id':None,'scientific_training_started':False,'optimizer_constructed':False,'model_loaded':False,'status':'PRE_OPTIMIZER_FAILURE'}
    try:
        actual=subprocess.check_output(['git','-C',str(args.repo_root),'rev-parse','HEAD'],text=True).strip()
        if actual!=args.expected_source_commit: raise RuntimeError('SOURCE_COMMIT_MISMATCH')
        receipt.update(validate(args.config,args.run_root));receipt['source_commit']=actual
        if not args.skip_runtime_imports:
            import torch, transformers, peft, bitsandbytes
            receipt['runtime']={'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'bitsandbytes':bitsandbytes.__version__,'cuda_available':bool(torch.cuda.is_available()),'bf16_supported':bool(torch.cuda.is_bf16_supported())}
            if not receipt['runtime']['cuda_available'] or not receipt['runtime']['bf16_supported']: raise RuntimeError('CUDA_BF16_UNAVAILABLE')
        receipt['status']='PASS_NO_MODEL_LOAD_OR_OPTIMIZER'
    except Exception as error:
        receipt['error']=f'{type(error).__name__}:{error}'
    atomic(args.receipt,receipt)
    return 0 if receipt['status']=='PASS_NO_MODEL_LOAD_OR_OPTIMIZER' else 1
if __name__=='__main__': raise SystemExit(main())
