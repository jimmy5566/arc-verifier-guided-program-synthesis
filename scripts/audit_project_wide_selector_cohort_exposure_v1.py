"""Fail-closed prior-exposure audit for prospective selector cohorts."""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
META_RE=re.compile(r'"(?:episode_id|family)"\s*:\s*"([^"]+)"')
def metadata_only(line):
 task_at=line.find('"task"')
 if task_at < 0: raise ValueError('TASK_BOUNDARY_MISSING')
 prefix=line[:task_at]
 if '"output"' in prefix: raise ValueError('TARGET_IN_METADATA_PREFIX')
 found=META_RE.findall(prefix)
 if len(found)!=2: raise ValueError('METADATA_SHAPE_INVALID')
 return found[0],found[1]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--target-dev',type=Path,required=True);ap.add_argument('--baseline-results',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 members=[]
 for n,line in enumerate(a.target_dev.read_text(encoding='utf-8').splitlines(),1):
  eid,fam=metadata_only(line);members.append({'episode_id':eid,'family':fam,'source_line_number':n})
 if len(members)!=192 or len({x['episode_id'] for x in members})!=192:raise RuntimeError('TARGET_DEV_METADATA_DENOMINATOR_FAIL')
 # Historical result text is read only to audit record structure; target grid values are not parsed or emitted.
 text=a.baseline_results.read_text(encoding='utf-8')
 exposed=[];missing=[]
 for m in members:
  marker='"episode_id": "'+m['episode_id']+'"';i=text.find(marker)
  if i<0:missing.append(m['episode_id']);continue
  # exact-grid score must be structurally adjacent to the historical episode record.
  window=text[i:i+800]
  if '"exact_grid_match"' in window:exposed.append(m['episode_id'])
  else:missing.append(m['episode_id'])
 out={'schema_version':1,'audit_id':'PROJECT_WIDE_SELECTOR_COHORT_PRIOR_EXPOSURE_AUDIT_V1','status':'NO_UNEXPOSED_NON_GOLD_SELECTOR_EVALUATION_COHORT' if len(exposed)==192 else 'INCOMPLETE_EXPOSURE_AUDIT','audit_scope':{'repository_evidence':'Corrected baseline per-episode result artifact and current non-Gold TARGET_DEV identity','prohibited_sources_not_opened':['Gold','dGold','FINAL_AUDIT']},'inputs':{'target_dev_sha256':sha(a.target_dev),'target_dev_rows':192,'baseline_results_path':str(a.baseline_results).replace('\\','/'),'baseline_results_sha256':sha(a.baseline_results)},'episode_level_findings':{'target_dev_episode_ids':192,'prior_exact_grid_scored_or_emitted':len(exposed),'not_proven_unexposed':len(missing),'unexposed_episode_ids':missing},'scientific_disposition':{'candidate_cohort':'NONE','reason':'Every one of 192 TARGET_DEV episode IDs appears with exact_grid_match in the historical corrected-baseline result artifact. The remaining 132 rows are therefore exposed and cannot serve as the prospective primary cohort.','required_state_if_no_other_verified_source':'PAUSED_NO_UNEXPOSED_SELECTOR_EVALUATION_COHORT'},'limits':['This audit does not open or emit TARGET_DEV targets; metadata uses only the prefix before the task key.','Absence of another declared non-Gold dev source in audited repository evidence is not proof that no external source exists; it is insufficient to authorize substitution.']}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
 if out['status']!='NO_UNEXPOSED_NON_GOLD_SELECTOR_EVALUATION_COHORT':raise RuntimeError('EXPOSURE_AUDIT_NOT_CONCLUSIVE')
if __name__=='__main__':main()