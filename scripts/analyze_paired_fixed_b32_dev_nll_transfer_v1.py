import json,collections,sys
from pathlib import Path
raw=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x]; out=Path(sys.argv[2]);W=('connected components','inside/contains','difference','width','orientation');C='ATOMIC_PREREQUISITE_FAMILY_BALANCED_LOSS_CONTROL_V1_001';V='CAPABILITY_REPAIR_BASELINE_V1_V7'
def avg(mode,ck,f):
 x=[r for r in raw if r['mode_id']==mode and r['checkpoint_id']==ck and r['family']==f];return sum(r['row_nll_sum'] for r in x)/sum(r['supervised_token_count'] for r in x) if x else None
p={f:avg('PRIMARY_B32',C,f)-avg('PRIMARY_B32',V,f) for f in W};b={f:avg('CROSSCHECK_B1',C,f)-avg('CROSSCHECK_B1',V,f) for f in W};stable=all(x is not None and b[x] is not None and (p[x]<0)==(b[x]<0) for x in W);result={'status':'COMPLETE_CPU_POSTPROCESS' if stable else 'NUMERICALLY_INCONCLUSIVE','primary_delta':p,'batch1_delta':b,'crosscheck_complete':all(v is not None for v in b.values()),'raw_rows':len(raw)};out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result,sort_keys=True))
