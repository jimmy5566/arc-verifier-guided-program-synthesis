"""Verify complete base and adapter directory identities before model import."""
import argparse, hashlib, json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def verify_files(root, records, tag):
 if not records: raise RuntimeError(tag+'_EMPTY_MANIFEST')
 expected={r['name']:r['sha256'] for r in records}; actual={p.name for p in root.iterdir() if p.is_file()}
 missing=set(expected)-actual
 if missing:raise RuntimeError(tag+'_MISSING:'+','.join(sorted(missing)))
 for name,digest in expected.items():
  if sha(root/name)!=digest:raise RuntimeError(tag+'_SHA:'+name)
  
 if tag=='BASE':
  if not any(name.startswith('model-') and name.endswith('.safetensors') for name in expected):raise RuntimeError('BASE_MANIFEST_NO_WEIGHT_SHARD')
  if not {'config.json','tokenizer_config.json'} <= set(expected):raise RuntimeError('BASE_MANIFEST_INCOMPLETE_CONFIG_TOKENIZER')
  extra={name for name in actual- set(expected) if name.startswith('model-') and name.endswith('.safetensors')}
  if extra:raise RuntimeError('BASE_UNEXPECTED_WEIGHT_SHARD:'+','.join(sorted(extra)))
 if tag=='ADAPTER':
  if not {'adapter_model.safetensors','adapter_config.json'} <= set(expected):raise RuntimeError('ADAPTER_MANIFEST_INCOMPLETE_WEIGHT_CONFIG')
  extra={name for name in actual-set(expected) if name.startswith('adapter_model.') or name.startswith('adapter_config.')}
  if extra:raise RuntimeError('ADAPTER_UNEXPECTED_IDENTITY_FILE:'+','.join(sorted(extra)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint-manifest',type=Path,action='append',required=True);a=ap.parse_args()
 verified=[]
 for path in a.checkpoint_manifest:
  m=json.loads(path.read_text());verify_files(Path(m['base_path']),m['base_files'],'BASE');verify_files(Path(m['adapter_path']),m['adapter_files'],'ADAPTER');verified.append(m.get('checkpoint_id'))
 print(json.dumps({'status':'PASS','checkpoints':verified},sort_keys=True))
if __name__=='__main__':main()
