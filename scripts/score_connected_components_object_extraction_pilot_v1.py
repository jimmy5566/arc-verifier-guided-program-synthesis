"""CPU-only post-freeze scorer.  It refuses any unfrozen generation output."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n');os.replace(q,p)
def fail(x): raise RuntimeError('E1_SCORE_'+x)
def score(binding_path:Path,output:Path):
 b=json.loads(binding_path.read_text())
 if sha(ROOT/b['score_entrypoint_path'])!=b['score_entrypoint_sha256']: fail('SCORER_IDENTITY')
 freeze=json.loads((output/'RAW_FREEZE_RECEIPT.json').read_text())
 if freeze.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or freeze.get('targets_accessed') is not False: fail('RAW_FREEZE_REQUIRED')
 raw={}
 for tag,expected in freeze['raw_sha256'].items():
  p=output/f'{tag}_RAW.jsonl'
  if not p.is_file() or sha(p)!=expected: fail('RAW_IDENTITY:'+tag)
  raw[tag]=[json.loads(line) for line in p.read_text().splitlines() if line]
 # All target reconstruction happens only after the raw SHA gate above.
 import importlib.util, sys
 def module(path,name):
  spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.path.insert(0,str((ROOT/path).parent));spec.loader.exec_module(m);return m
 freezer=module('scripts/freeze_connected_components_object_extraction_capability_pilot_v1.py','e1_freezer')
 v3=module('scripts/e04_orientation_fixed_demonstration_baseline_v3.py','e1_v3'); v2=v3.v2
 base=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
 normal={}
 for filename,split in [('OOD_INPUT_PROMPTS.json','OOD'),('RETENTION_INPUT_PROMPTS.json','RETENTION'),('COMPOSITION_SENTINEL_INPUT_PROMPTS.json','COMPOSITION')]:
  for i,row in enumerate(json.loads((base/filename).read_text())):
   c=row['latent_condition'];normal[f'{split}:{i}']=freezer.task(c['family'],c['topology'],10000+i*4,row['task']['source_id'],True)['test'][0]['output']
 protected={r['protected_id']:r for r in json.loads((base/'measurement_contract_v1/PROTECTED_CAPABILITY_INPUT_PROMPTS.json').read_text())['records']}
 bases={x['canonical_base_id']:x for x in v2.canonical_bases('VALIDATION')}
 train={}
 for arm,file in [('CONTROL_NARROW_STRUCTURE','CONTROL_SCHEDULE.json'),('TREATMENT_STRUCTURE_DIVERSE','TREATMENT_SCHEDULE.json')]:
  for row in json.loads((base/file).read_text())['episodes']:
   train[f'TRAIN_FIT:{arm}:{row["slot"]}']=row['task']['test'][0]['output']
 def expected(r):
  key=r['record_id']
  if key in normal:return normal[key]
  if key in train:return train[key]
  if key.startswith('PROTECTED:'):
   q=protected[key.split(':',1)[1]];return v3.target(bases[q['canonical_base_id']],q['control_marker_turn'],q['condition'],q['source_row_index'])['target']
  fail('UNKNOWN_RECORD:'+key)
 summary={}
 for tag,values in raw.items():
  by={}
  for row in values:
   exact=bool(row['parser_valid']) and row['parsed_grid']==expected(row)
   bucket=by.setdefault(row['split'],{'rows':0,'parser_valid':0,'exact':0});bucket['rows']+=1;bucket['parser_valid']+=bool(row['parser_valid']);bucket['exact']+=exact
  summary[tag]={'raw_sha256':freeze['raw_sha256'][tag],'by_split':by}
 result={'status':'CPU_POST_FREEZE_DETERMINISTIC_SCORE_COMPLETE','binding_sha256':sha(binding_path),'raw_freeze_sha256':sha(output/'RAW_FREEZE_RECEIPT.json'),'by_artifact':summary,'targets_opened_after_raw_freeze':True}
 write(output/'SCORE_RECEIPT.json',result);return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True);z=a.parse_args();print(json.dumps(score(z.binding,z.output_root),sort_keys=True))
