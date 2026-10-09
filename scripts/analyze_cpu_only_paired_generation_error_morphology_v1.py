"""CPU-only paired morphology analysis over already frozen DEV generations."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FAMILIES=("connected components","inside/contains","difference","width","orientation")
MANIFEST=ROOT/"experiments/capability_repair_baseline_v1/paired_final_query_loss_decomposition_v1/TARGET_DEV_60_FINAL_QUERY_MANIFEST.json"
V7=ROOT/"experiments/capability_repair_baseline_v1/corrected_baseline_v1/CAPABILITY_REPAIR_BASELINE_V1_RESULTS.json"
FB=ROOT/"experiments/capability_repair_baseline_v1/family_balanced_loss_control_v1/round_001/evaluation_v1/CAPABILITY_REPAIR_BASELINE_V1_RESULTS.json"
DEV=ROOT/"experiments/targeted_capability_repair_v1/data/TARGET_DEV.jsonl"
OUT=ROOT/"experiments/capability_repair_baseline_v1/cpu_only_paired_generation_error_morphology_v1/RESULT.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def q(v,p):
 v=sorted(v);x=(len(v)-1)*p;a=int(x);b=min(a+1,len(v)-1);return v[a]+(v[b]-v[a])*(x-a)
def lev(a,b):
 d=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  n=[i]
  for j,y in enumerate(b,1):n.append(min(n[-1]+1,d[j]+1,d[j-1]+(x!=y)))
  d=n
 return d[-1]
def grid_tokens(task):
 g=task['test'][0]['output'];return [z for i,row in enumerate(g) for z in ([int(x) for x in row]+([10] if i<len(g)-1 else []))]
def main():
 m=json.loads(MANIFEST.read_text());members=[x for b in m['batches'] for x in b['members']];ids={x['episode_id'] for x in members};family={x['episode_id']:x['family'] for x in members}
 source={x['episode_id']:x for x in (json.loads(y) for y in DEV.read_text().splitlines() if y.strip())};v={x['episode_id']:x for x in json.loads(V7.read_text())['predictions'] if x['episode_id'] in ids};f={x['episode_id']:x for x in json.loads(FB.read_text())['predictions'] if x['episode_id'] in ids}
 if set(v)!=ids or set(f)!=ids or any(not source.get(i) for i in ids):raise RuntimeError('FROZEN_INPUT_COMPLETENESS_FAIL')
 rows=[]
 for i in sorted(ids):
  a,b=v[i],f[i]
  if a['prompt_sha256']!=b['prompt_sha256'] or a['observation_sha256']!=b['observation_sha256']:raise RuntimeError('PREDICTION_IDENTITY_FAIL:'+i)
  target=grid_tokens(source[i]['task']); da=lev(a['content_token_ids'],target)/max(1,len(target));db=lev(b['content_token_ids'],target)/max(1,len(target))
  rows.append({'episode_id':i,'family':family[i],'v7_distance':da,'fb_distance':db,'delta_fb_minus_v7':db-da,'v7_parse_valid':a['parse_valid'],'fb_parse_valid':b['parse_valid'],'v7_exact':a['exact_grid_match'],'fb_exact':b['exact_grid_match'],'target_tokens':len(target),'v7_content_tokens':len(a['content_token_ids']),'fb_content_tokens':len(b['content_token_ids'])})
 if {z:sum(x['family']==z for x in rows) for z in FAMILIES}!={z:12 for z in FAMILIES}:raise RuntimeError('FAMILY_DENOMINATOR_FAIL')
 effect=lambda rs:sum(sum(x['delta_fb_minus_v7'] for x in rs if x['family']==z)/12 for z in FAMILIES)/5
 rng=random.Random(20261009);by={z:[x for x in rows if x['family']==z] for z in FAMILIES};boot=[]
 for _ in range(10000):boot.append(effect([rng.choice(by[z]) for z in FAMILIES for _ in range(12)]))
 lo,hi=q(boot,.025),q(boot,.975);fm={z:sum(x['delta_fb_minus_v7'] for x in by[z])/12 for z in FAMILIES};nonpos=sum(x<=0 for x in fm.values());nonneg=sum(x>=0 for x in fm.values())
 cls='SUBEXACT_OUTPUT_PROGRESS' if hi<0 and nonpos>=4 else 'OUTPUT_QUALITY_DECOUPLING' if lo>=0 and nonneg>=4 else 'MIXED_INCONCLUSIVE'
 out={'protocol_id':'CPU_ONLY_PAIRED_GENERATION_ERROR_MORPHOLOGY_V1','status':'COMPLETE_CPU_ONLY','evidence_scope':'REUSED_TARGET_DEV_DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_ARC_GENERALIZATION','inputs':{'manifest_sha256':sha(MANIFEST),'v7_predictions_sha256':sha(V7),'family_balanced_predictions_sha256':sha(FB),'target_dev_sha256':sha(DEV)},'primary':{'estimand':'family_equal_mean(FB_minus_V7_normalized_content_token_levenshtein)','effect':effect(rows),'bootstrap_95_ci':[lo,hi],'per_family':fm,'classification':cls},'descriptive':{'v7_exact':sum(x['v7_exact'] for x in rows),'fb_exact':sum(x['fb_exact'] for x in rows),'v7_parse_valid':sum(x['v7_parse_valid'] for x in rows),'fb_parse_valid':sum(x['fb_parse_valid'] for x in rows)},'rows':rows}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps(out['primary'],sort_keys=True))
if __name__=='__main__':main()
