"""Fail closed if the frozen paired-fixed-B32 package drifts from bound bytes."""
import argparse, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,required=True);a=ap.parse_args()
 p=json.loads(a.package.read_text())
 if 'decision_rule' in p: raise RuntimeError('OBSOLETE_AGGREGATE_DECISION_RULE_PRESENT')
 rules=p.get('authoritative_classifier',{}).get('mutually_exclusive_rules',[])
 if len(rules)!=4: raise RuntimeError('AUTHORITATIVE_CLASSIFIER_INCOMPLETE')
 bound=p.get('execution',{}).get('tested_source_hashes',{})
 if not bound: raise RuntimeError('TESTED_SOURCE_BINDINGS_MISSING')
 for rel,digest in bound.items():
  if sha(ROOT/rel)!=digest: raise RuntimeError('SOURCE_BINDING_MISMATCH:'+rel)
 if p['execution'].get('worker_sha256')!=bound.get(p['execution'].get('worker_path')):raise RuntimeError('WORKER_BINDING_MISMATCH')
 print(json.dumps({'status':'PASS','package_sha256':sha(a.package),'bound_files':len(bound)},sort_keys=True))
if __name__=='__main__':main()
