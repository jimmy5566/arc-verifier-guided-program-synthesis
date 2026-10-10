#!/usr/bin/env python3
"""CPU postprocessor for bounded E03 Gram sufficient statistics."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
from e03_gram_statistics import WEAK,PROTECTED,bootstrap,summary,sensitivity,expand,signs_oppose

def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n');os.replace(t,p)
def load(p):
 d=json.loads(p.read_text(encoding='utf-8'))
 if d.get('status')!='COMPLETE_NO_UPDATE_E03_GRAM_SUFFICIENT_STATISTICS':raise RuntimeError('E03_RAW_STATUS_INVALID')
 for k,v in {'manifest_rows':288,'batch1_rows':72,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False}.items():
  if d.get(k)!=v:raise RuntimeError('E03_NO_UPDATE_OR_SHAPE_INVALID')
 if len(d.get('primary_records',[]))!=36 or len(d.get('batch1_family_records',[]))!=9:raise RuntimeError('E03_SUFFICIENT_STATISTICS_COMPLETENESS_FAIL')
 return d
def classify(point,intervals,envelope):
 weak={f:expand(intervals['weak_by_equal_protected_macro'][f],envelope[f'weak:{f}']) for f in WEAK}
 hist=expand(intervals['historical_macro'],envelope['historical_protected_macro'])
 equal=expand(intervals['equal_macro'],envelope['equal_protected_macro'])
 protected={p:expand(intervals['historical_by_protected'][p],envelope[f'historical_protected:{p}']) for p in PROTECTED}
 if signs_oppose(protected):return 'INCONCLUSIVE_PROTECTED_SIGNS_OPPOSE',weak,hist,equal,protected
 # Point estimate, then sensitivity-expanded bootstrap bound: do not use an
 # interval lower bound as a substitute for the frozen -0.05 effect threshold.
 weak_support=sum(point['weak_protected_macro_by_weak'][f]<-.05 and weak[f]['upper']<0 for f in WEAK)>=3
 hist_support=point['historical_protected_macro']<-.05 and hist['upper']<0
 if weak_support or hist_support:return 'LOCAL_INTERFERENCE_SUPPORTED',weak,hist,equal,protected
 freq=point['negative_pair_frequency']; e=envelope['negative_pair_frequency']
 if (freq-e)<.10<=(freq+e):return 'INCONCLUSIVE_NEGATIVE_FREQUENCY_SENSITIVITY',weak,hist,equal,protected
 if all(x['lower']>=0 for x in protected.values()) and freq+e<.10:return 'GRADIENT_CONFLICT_DEPRIORITIZED',weak,hist,equal,protected
 return 'INCONCLUSIVE',weak,hist,equal,protected
def process(d):
 primary=d['primary_records']; pg=d['primary_global_gram']; b1=d['batch1_family_records']; bg=d['batch1_global_gram']
 point=summary(primary,pg,4); intervals=bootstrap(primary,pg,10000,20261010)
 mb0=[r for r in primary if r['microbatch_index']==0]; envelope,b8subset,b1point=sensitivity(mb0,pg,b1,bg)
 result,weak,hist,equal,protected=classify(point,intervals,envelope)
 return {'protocol_id':'E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V1','status':'COMPLETE_NO_UPDATE_LOCAL_OPTIMIZATION_EVIDENCE_ONLY','classification':result,
 'point_estimates':{k:v for k,v in point.items() if not k.endswith('coefficients') and k!='family_coefficients'},
 'bootstrap':{'method':'family-stratified resampling of four B8 microbatches','replicates':10000,'seed':20261010,'raw_intervals':intervals},
 'numerical_sensitivity':{'definition':'B8 microbatch0 versus B1 CE-numerator reconstructed family estimator','metric_specific_envelopes':envelope,'b8_subset_point_estimates':{k:v for k,v in b8subset.items() if not k.endswith('coefficients') and k!='family_coefficients'},'b1_reconstructed_point_estimates':{k:v for k,v in b1point.items() if not k.endswith('coefficients') and k!='family_coefficients'},'adjusted_weak_intervals':weak,'adjusted_historical_macro':hist,'adjusted_equal_counterfactual_macro':equal,'adjusted_historical_protected':protected},
 'module_level_gram_groups':sorted(d.get('primary_module_grams',{})),'causal_limit':'local synthetic-TRAIN gradient geometry is not causal proof of forgetting or ARC transfer'}
def main():
 a=argparse.ArgumentParser();a.add_argument('--raw',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
 if x.output.exists():raise RuntimeError('E03_POSTPROCESS_OUTPUT_NON_OVERWRITE_REQUIRED')
 atomic(x.output,process(load(x.raw)))
if __name__=='__main__':main()
