"""Exact Gram-matrix algebra for the E03 bounded gradient diagnostic.

The durable evidence contains row identities and Gram matrices, never complete
LoRA gradient vectors.  A coefficient vector represents a linear combination
of the transient gradient basis; all norms and cosines are exact under its Gram
matrix.
"""
from __future__ import annotations
import math, random

WEAK=("connected_components","inside_contains","width","difference","orientation")
PROTECTED=("same_color","color_mapping")
FAMILIES=WEAK+PROTECTED+("object_selection_then_action","novel_two_operation_order")

def _finite(x): return math.isfinite(float(x))
def validate_gram(gram,n):
 if len(gram)!=n or any(len(r)!=n for r in gram): raise RuntimeError('GRAM_DIMENSION_MISMATCH')
 for i,row in enumerate(gram):
  for j,x in enumerate(row):
   if not _finite(x): raise RuntimeError('NONFINITE_GRAM')
   if abs(float(x)-float(gram[j][i]))>1e-5: raise RuntimeError('GRAM_NOT_SYMMETRIC')
  if float(row[i])<=0: raise RuntimeError('ZERO_NORM_GRADIENT_INVALID_NOT_INTERPRETABLE')

def zero(n): return [0.0]*n
def scaled_sum(parts,n):
 out=zero(n)
 for coeff,scale in parts:
  if len(coeff)!=n or not _finite(scale): raise RuntimeError('INVALID_GRADIENT_COMBINATION')
  for i,x in enumerate(coeff): out[i]+=float(x)*float(scale)
 return out
def inner(a,b,gram):
 return sum(float(a[i])*float(gram[i][j])*float(b[j]) for i in range(len(a)) for j in range(len(b)))
def magnitude(a,gram):
 v=inner(a,a,gram)
 if not _finite(v) or v<=0: raise RuntimeError('ZERO_NORM_GRADIENT_INVALID_NOT_INTERPRETABLE')
 return math.sqrt(v)
def cosine(a,b,gram): return inner(a,b,gram)/(magnitude(a,gram)*magnitude(b,gram))
def unit(a,gram):
 n=magnitude(a,gram); return [x/n for x in a]

def _basis(row,n):
 idx=int(row['basis_index'])
 if not 0<=idx<n: raise RuntimeError('BASIS_INDEX_INVALID')
 out=zero(n);out[idx]=1.;return out

def by_family(records):
 out={f:[] for f in FAMILIES}
 for r in records:
  family=r.get('canonical_family')
  if family not in out: raise RuntimeError('UNEXPECTED_FAMILY')
  if int(r.get('supervised_token_count',0))<1: raise RuntimeError('ZERO_SUPERVISED_TOKENS')
  out[family].append(r)
 return out

def family_coefficients(records,gram,expected_per_family):
 n=len(gram); validate_gram(gram,n); by=by_family(records)
 if any(len(by[f])!=expected_per_family for f in FAMILIES): raise RuntimeError('FAMILY_MICROBATCH_COMPLETENESS_FAIL')
 coeff={}; totals={}
 for f,rows in by.items():
  total=sum(int(r['supervised_token_count']) for r in rows)
  totals[f]=total
  coeff[f]=scaled_sum([(_basis(r,n),int(r['supervised_token_count'])/total) for r in rows],n)
 return coeff,totals,by

def summary(records,gram,expected_per_family=4):
 n=len(gram); coeff,totals,by=family_coefficients(records,gram,expected_per_family)
 historical=scaled_sum([(coeff[f],totals[f]/sum(totals.values())) for f in FAMILIES],n)
 equal=scaled_sum([(unit(coeff[f],gram),1/len(FAMILIES)) for f in FAMILIES],n)
 historical_by={p:cosine(historical,coeff[p],gram) for p in PROTECTED}
 equal_by={p:cosine(equal,coeff[p],gram) for p in PROTECTED}
 weak_by={w:sum(cosine(coeff[w],coeff[p],gram) for p in PROTECTED)/len(PROTECTED) for w in WEAK}
 neg=[]
 for w in WEAK:
  for p in PROTECTED:
   for a in by[w]:
    for b in by[p]: neg.append(cosine(_basis(a,n),_basis(b,n),gram)<0)
 return {'token_totals':totals,'family_coefficients':coeff,'historical_coefficients':historical,'equal_coefficients':equal,
  'weak_protected_macro_by_weak':weak_by,'historical_by_protected':historical_by,'equal_by_protected':equal_by,
  'historical_protected_macro':sum(historical_by.values())/len(PROTECTED),
  'equal_protected_macro':sum(equal_by.values())/len(PROTECTED),
  'negative_pair_frequency':sum(neg)/len(neg)}

def interval(values):
 if not values: raise RuntimeError('EMPTY_BOOTSTRAP')
 x=sorted(float(v) for v in values); return {'lower':x[round((len(x)-1)*.025)],'upper':x[round((len(x)-1)*.975)]}

def bootstrap(records,gram,replicates=10000,seed=20261010):
 by=by_family(records)
 if any(len(by[f])!=4 for f in FAMILIES): raise RuntimeError('FAMILY_MICROBATCH_COMPLETENESS_FAIL')
 rng=random.Random(seed); out={'historical_macro':[],'equal_macro':[]};weak={f:[] for f in WEAK};hist={p:[] for p in PROTECTED};equal={p:[] for p in PROTECTED}
 for _ in range(replicates):
  sample=[]
  for f in FAMILIES: sample.extend(rng.choice(by[f]) for __ in range(4))
  s=summary(sample,gram,4)
  out['historical_macro'].append(s['historical_protected_macro']);out['equal_macro'].append(s['equal_protected_macro'])
  for f in WEAK: weak[f].append(s['weak_protected_macro_by_weak'][f])
  for p in PROTECTED: hist[p].append(s['historical_by_protected'][p]);equal[p].append(s['equal_by_protected'][p])
 ans={k:interval(v) for k,v in out.items()}
 ans['weak_by_equal_protected_macro']={k:interval(v) for k,v in weak.items()}
 ans['historical_by_protected']={k:interval(v) for k,v in hist.items()}
 ans['equal_by_protected']={k:interval(v) for k,v in equal.items()}
 return ans

def sensitivity(primary_mb0,primary_gram,b1_records,b1_gram):
 a=summary(primary_mb0,primary_gram,1); b=summary(b1_records,b1_gram,1)
 out={'negative_pair_frequency':abs(a['negative_pair_frequency']-b['negative_pair_frequency']),
      'historical_protected_macro':abs(a['historical_protected_macro']-b['historical_protected_macro']),
      'equal_protected_macro':abs(a['equal_protected_macro']-b['equal_protected_macro'])}
 for f in WEAK: out[f'weak:{f}']=abs(a['weak_protected_macro_by_weak'][f]-b['weak_protected_macro_by_weak'][f])
 for p in PROTECTED:
  out[f'historical_protected:{p}']=abs(a['historical_by_protected'][p]-b['historical_by_protected'][p])
  out[f'equal_protected:{p}']=abs(a['equal_by_protected'][p]-b['equal_by_protected'][p])
 return out,a,b

def expand(i,e): return {'lower':float(i['lower'])-float(e),'upper':float(i['upper'])+float(e)}
def signs_oppose(intervals):
 s=[-1 if x['upper']<0 else 1 if x['lower']>0 else 0 for x in intervals.values()]
 return -1 in s and 1 in s
