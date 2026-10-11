"""Pure-Python E04-D clustered statistics and sensitivity classification."""
from __future__ import annotations
import math,random

def q(v,p):
 v=sorted(map(float,v)); return v[round((len(v)-1)*p)]
def interval(v): return {'lower':q(v,.025),'upper':q(v,.975)}
def chord(c): return math.sqrt(max(0.,2.-2.*min(1.,max(-1.,float(c)))))
def radius(repeats): return min(2.,sum(chord(v) for v in repeats.values()))
def env(ci,u): return {'lower':max(-1.,ci['lower']-u),'upper':min(1.,ci['upper']+u)}
def sign(c,u): return 'neg' if c+u<0 else 'nonneg' if c-u>=0 else 'ambiguous'
def boot(rows,replicates=10000,seed=20261011):
 by={}
 for r in rows: by.setdefault(r['canonical_base_id'],[]).append(r)
 if len(by)!=12 or any(len(v)!=4 for v in by.values()): raise ValueError('E04D_BASE_BLOCK_INVALID')
 bases=sorted(by); rng=random.Random(seed); agg=[]; lo=[];hi=[]
 for _ in range(replicates):
  sample=[r for b in (rng.choice(bases) for _ in bases) for r in by[b]]
  agg.append(sum(r['rotation_marker_cosine'] for r in sample)/len(sample))
  lo.append(sum(r['rotation_marker_sign']=='neg' for r in sample)/len(sample))
  hi.append(sum(r['rotation_marker_sign']!='nonneg' for r in sample)/len(sample))
 return interval(agg),interval(lo),interval(hi)
def classify(rows,repeat_cosines):
 u=radius(repeat_cosines)
 for r in rows:r['rotation_marker_sign']=sign(r['rotation_marker_cosine'],u)
 ci,flo,fhi=boot(rows); adj=env(ci,u)
 if adj['upper']<0 and flo['lower']>.5: decision='LOCAL_CONFLICT_SUPPORTED'
 elif adj['lower']>0 and fhi['upper']<.5: decision='LOCAL_CONFLICT_DEPRIORITIZED'
 else: decision='MIXED_OR_INCONCLUSIVE'
 return {'decision':decision,'u_rotation_marker':u,'aggregate_bootstrap_ci':ci,'aggregate_sensitivity_envelope':adj,'negative_frequency_lower_ci':flo,'negative_frequency_upper_ci':fhi}
