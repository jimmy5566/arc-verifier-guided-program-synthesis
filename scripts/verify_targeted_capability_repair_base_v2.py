#!/usr/bin/env python3
"""Revisioned full-byte base-model manifest builder and comparator (CPU I/O only)."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def manifest(root: Path) -> dict:
    if not root.is_dir(): raise RuntimeError('BASE_DIRECTORY_MISSING')
    files=[]
    for p in sorted(x for x in root.rglob('*') if x.is_file()):
        files.append({'path':str(p.relative_to(root)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':sha256(p)})
    required={'config.json'}
    names={x['path'] for x in files}
    if not required <= names or not any(n.startswith('tokenizer') for n in names) or not any(n.endswith(('.safetensors','.bin','.index.json')) for n in names):
        raise RuntimeError('INCOMPLETE_BASE_MODEL_ASSET_SET')
    canonical=json.dumps(files,sort_keys=True,separators=(',',':')).encode()
    return {'schema_version':2,'kind':'COMPLETE_BASE_BYTE_MANIFEST','base_dir':str(root),'files':files,'manifest_sha256':hashlib.sha256(canonical).hexdigest()}

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--base-dir',type=Path,required=True); p.add_argument('--expected-manifest',type=Path); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    actual=manifest(a.base_dir); result={'actual':actual,'status':'EXPECTED_MANIFEST_NOT_SUPPLIED'}
    if a.expected_manifest:
        expected=json.loads(a.expected_manifest.read_text(encoding='utf-8')); result['expected_manifest_sha256']=expected.get('manifest_sha256'); result['status']='PASS' if expected==actual else 'BASE_BYTE_MANIFEST_MISMATCH'
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n'); print(result['status']); return 0 if result['status']=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
