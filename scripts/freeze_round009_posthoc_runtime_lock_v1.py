"""Create the post-commit exact-checkout launch lock for a Phase binding."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 b=json.loads(a.binding.read_text(encoding='utf-8-sig'))
 if not b.get('runtime_launch_lock_required') or b.get('execution_checkout_commit') is not None: raise RuntimeError('BINDING_NOT_RUNTIME_LOCK_TEMPLATE')
 head=subprocess.check_output(['git','-C',str(a.source_root),'rev-parse','HEAD'],text=True).strip()
 lock={'schema_version':1,'binding_path':str(a.binding.resolve()),'binding_sha256':sha(a.binding),'execution_checkout_commit':head,'source_root':str(a.source_root.resolve()),'entrypoint_identities':b['entrypoint_identities'],'nonce':b['nonce'],'phase':b['phase']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n');print(json.dumps(lock,sort_keys=True))
if __name__=='__main__':main()
