#!/usr/bin/env python3
"""CPU postprocessor for the frozen E03 sufficient-statistics schema."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from e03_gradient_statistics import FAMILIES,WEAK,PROTECTED,bootstrap,summary_from_records,b1_reconstruction,sensitivity_envelope,expanded,protected_sign_consistent

def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n');os.replace(tmp,path)
def load(path):
 data=json.loads(path.read_text(encoding='utf-8'))
 if data.get('status')!='COMPLETE_NO_UPDATE_E03_SUFFICIENT_STATISTICS': raise RuntimeError('E03_RAW_STATUS_INVALID')
 if data.get('optimizer_steps')!=0 or data.get('parameter_updates')!=0 or data.get('generation_calls')!=0 or data.get('final_audit_opened') is not False: raise RuntimeError('E03_NO_UPDATE_BOUNDARY_FAIL')
 if data.get('manifest_rows')!=288 or data.get('batch1_rows')!=72: raise RuntimeError('E03_ROW_COUNT_INVALID')
 return data
def classify(intervals,envelopes,negative_frequency):
 weak={name:expanded(intervals['weak_by_equal_protected_macro'][name],max(envelopes[f'weak_protected:{name}|{p}'] for p in PROTECTED)) for name in WEAK}
 historical=expanded(intervals['historical_macro'],envelopes['historical_protected_macro'])
 protected=intervals['historical_by_protected']
 if not protected_sign_consistent(protected): return 'INCONCLUSIVE_PROTECTED_SIGNS_OPPOSE',weak,historical
 supported=sum(cell['upper']<0 and cell['lower']<-0.05 for cell in weak.values())>=3 or (historical['upper']<0 and historical['lower']<-0.05)
 if supported:return 'LOCAL_INTERFERENCE_SUPPORTED',weak,historical
 if all(cell['lower']>=0 for cell in protected.values()) and negative_frequency<0.10:return 'GRADIENT_CONFLICT_DEPRIORITIZED',weak,historical
 return 'INCONCLUSIVE',weak,historical
def process(data):
 records=data['primary_b8_microbatch_gradients']; b1=data['batch1_per_example_gradients']
 primary=summary_from_records(records); intervals=bootstrap(records,replicates=10000,seed=20261010)
 b1_rec=b1_reconstruction(b1); b8_0=[r for r in records if r['microbatch_index']==0]
 envelopes=sensitivity_envelope(b8_0,b1_rec)
 classification,weak_adjusted,historical_adjusted=classify(intervals,envelopes,primary['weak_protected_independent_pair_negative_frequency'])
 return {'protocol_id':'E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V1','status':'COMPLETE_NO_UPDATE_LOCAL_OPTIMIZATION_EVIDENCE_ONLY',
 'primary_summary':{k:v for k,v in primary.items() if k not in {'family_vectors','historical_combined','equal_family_combined'}},
 'bootstrap':{'method':'family-dependent resampling of four microbatches within each family','replicates':10000,'seed':20261010,'raw_intervals':intervals},
 'numerical_sensitivity':{'definition':'absolute B8 microbatch-0 versus token-mean reconstructed B1 difference','metric_specific_envelopes':envelopes,'adjusted_weak_intervals':weak_adjusted,'adjusted_historical_macro_interval':historical_adjusted},
 'classification':classification,'causal_limit':'local gradient disagreement is not causal proof of forgetting or ARC transfer',
 'development_scope':'legitimate synthetic TRAIN gradients only; no DEV/retention/Gold/dGold/FINAL_AUDIT evidence'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('E03_POSTPROCESS_OUTPUT_NON_OVERWRITE_REQUIRED')
 atomic(a.output,process(load(a.raw)))
if __name__=='__main__':main()
