"""Phase B sealed scorer. It joins immutable predictions to sealed hash references."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path: sys.path.insert(0,str(ROOT/'src'))
from generation_capability_audit_v1.audit import canonical_sha256,grid_from_native_tokens

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(t,p)
def parsed(tokens):
 try:
  body=list(tokens);body=body[:body.index(15)] if 15 in body else body
  return grid_from_native_tokens(body)
 except Exception:return None
def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--prediction-manifest',type=Path,required=True);p.add_argument('--phase-a-receipt',type=Path,required=True);p.add_argument('--sealed-mapping',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 b=json.loads(a.binding.read_text(encoding='utf-8-sig'));r=json.loads(a.phase_a_receipt.read_text(encoding='utf-8-sig'));pred=json.loads(a.prediction_manifest.read_text(encoding='utf-8-sig'));mapping=json.loads(a.sealed_mapping.read_text(encoding='utf-8-sig'))
 if b.get('phase')!='B_SEALED_SCORING' or r.get('status')!='SUCCESS' or not a.execute:raise RuntimeError('PHASE_B_PREDECESSOR_OR_EXECUTE_INVALID')
 if not b.get('execution_authorized'):raise RuntimeError('PHASE_B_NOT_AUTHORIZED')
 if a.output.exists() and any(a.output.iterdir()):raise RuntimeError('PHASE_B_OUTPUT_ROOT_NOT_FRESH')
 refs={(s,x['manifest_item_id']):x for s,items in mapping['surfaces'].items() for x in items}; rows={(x['surface'],x['sample_id']):x for x in pred['predictions']}
 if set(rows)!=set(refs) or len(rows)!=256:raise RuntimeError('PHASE_B_ONE_TO_ONE_PREDICTION_MAPPING_FAILURE')
 scored=[]
 for k,x in rows.items():
  grid=parsed(x['generated_token_ids']);actual=None if grid is None else canonical_sha256(grid);ref=refs[k];scored.append({'surface':k[0],'sample_id':k[1],'parse_status':'VALID_GRID' if grid is not None else 'INVALID_FORMAT','exact_grid':actual==ref['canonical_reference_output_grid_sha256'],'prediction_sha256':x['prediction_sha256']})
 summary={s:{'denominator':sum(x['surface']==s for x in scored),'exact':sum(x['surface']==s and x['exact_grid'] for x in scored)} for s in ('NOVEL','REPLAY','PROTECTED')}
 if {s:v['denominator'] for s,v in summary.items()}!={'NOVEL':128,'REPLAY':64,'PROTECTED':64}:raise RuntimeError('FIXED_DENOMINATOR_VIOLATION')
 result={'schema_version':1,'phase':'B_SEALED_SCORING','status':'SUCCESS','scores':scored,'summary':summary,'protected_semantics':'ABSOLUTE_ONLY','mapping_sha256':sha(a.sealed_mapping)};write(a.output/'PHASE_B_RESULTS.json',result);rh=sha(a.output/'PHASE_B_RESULTS.json');receipt={'schema_version':1,'phase':'B_SEALED_SCORING','status':'SUCCESS','nonce':b['nonce'],'result_sha256':rh,'phase_a_receipt_sha256':sha(a.phase_a_receipt),'prediction_manifest_sha256':sha(a.prediction_manifest),'mapping_sha256':sha(a.sealed_mapping),'model_loaded':False,'scoring_started':True,'final_audit_opened':False};write(a.receipt,receipt);print(json.dumps(receipt,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
