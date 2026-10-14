"""CPU-only freeze of E4F2 counterfactual Count-primary training geometries."""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, os, sys
from pathlib import Path
ROOT=Path(os.environ.get('ARC2_ROOT',Path(__file__).resolve().parents[1]));sys.path[:0]=[str(ROOT),str(ROOT/'src')]
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1';E3=BASE/'deconfounded_count_curriculum_causal_pilot_v1';OUT=E3/'e4f2_distance_overlap_affinity_v1';SOURCE=E3/'TREATMENT_SCHEDULE.json'
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canon(x)+b'\n')
def components(grid):
 seen=set();out=[]
 for y,row in enumerate(grid):
  for x,c in enumerate(row):
   if c==0 or (y,x) in seen:continue
   seen.add((y,x));q=[(y,x)];part=[]
   while q:
    cy,cx=q.pop(0);part.append((cy,cx))
    for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
     if 0<=ny<len(grid) and 0<=nx<len(row) and (ny,nx) not in seen and grid[ny][nx]==c:seen.add((ny,nx));q.append((ny,nx))
   out.append(tuple(sorted(part)))
 return tuple(sorted(out))
def output(count,color):return [[0,0,0],[*([color]*count),*([0]*(3-count))],[0,0,0]]
def geometry(count,color,variant,shift):
 """Each component has >=3 cells; P/N share target-pair feature tuple."""
 y,x=1+(shift%2)*3,1+((shift//2)%2)*3;p0=(y,x);p1=(y+1,x+1)
 if variant=='P':
  cells=[p0,(y,x+1),p1]
  if count>=2: cells += [(y+4,x+4),(y+4,x+5),(y+5,x+5)]
 else:
  cells=[p0,(y,x-1),(y+1,x-1)]
  if count>=2: cells += [p1,(y+1,x+2),(y+2,x+2)]
 if count==3:cells += [(y+7,x),(y+7,x+1),(y+8,x+1)]
 g=[[0]*16 for _ in range(16)]
 for a,b in cells:g[a][b]=color
 cs=components(g)
 if len(cs)!=count or any(len(c)<3 for c in cs):raise RuntimeError('E4F2_GEOMETRY')
 return g,tuple(sorted((p0,p1)))
def task(count,color,variant,shift,source):
 q,pair=geometry(count,color,variant,shift);d=[]
 for j,c in enumerate((1,2,3)):
  dg,_=geometry(c,((color+j)%9) or 1,'P' if (j+shift)%2 else 'N',shift+j+2);d.append({'input':dg,'output':output(c,((color+j)%9) or 1)})
 return {'source_id':source,'train':d,'test':[{'input':q,'output':output(count,color)}]},pair
def pair_features(grid,pair):
 cs=components(grid);ids={v:i for i,c in enumerate(cs) for v in c};sizes={i:len(c) for i,c in enumerate(cs)};(y0,x0),(y1,x1)=pair
 return {'pair':[list(pair[0]),list(pair[1])],'label':'SAME_COMPONENT' if ids[pair[0]]==ids[pair[1]] else 'DIFFERENT_COMPONENT','colour_equal':True,'manhattan':abs(y0-y1)+abs(x0-x1),'chebyshev':max(abs(y0-y1),abs(x0-x1)),'relative_displacement':[abs(y0-y1),abs(x0-x1)],'absolute_coordinates':[list(pair[0]),list(pair[1])],'component_sizes':[sizes[ids[pair[0]]],sizes[ids[pair[1]]]],'topology':'L3_NEAR_COUNTERFACTUAL','pair_order':'CANONICAL_UNORDERED'}
def select(task_value,target_pair):
 grid=task_value['test'][0]['input']
 # P/N counterfactual slot pairs share the full observed feature tuple but
 # reverse only component membership.  One pair per treated query therefore
 # gives exact global marginal matching without injecting a within-query
 # coordinate shortcut.
 return [pair_features(grid,tuple(target_pair))]
def audit(records):
 selected=[p|{'slot':r['slot'],'count':r['count'],'variant':r['variant']} for r in records if r['count']>1 for p in r['selected_pairs']]
 if len(selected)!=32 or sum(x['label']=='SAME_COMPONENT' for x in selected)!=16 or sum(x['label']=='DIFFERENT_COMPONENT' for x in selected)!=16:raise RuntimeError('E4F2_PAIR_BALANCE')
 if any(sum(x['slot']==slot for x in selected)!=1 for slot in {x['slot'] for x in selected}):raise RuntimeError('E4F2_SLOT_MASS')
 if len({(x['slot'],tuple(map(tuple,x['pair']))) for x in selected})!=32:raise RuntimeError('E4F2_DUPLICATE')
 allpairs=[p for r in records if r['count']>1 for p in r['all_pairs']]
 features=('colour_equal','manhattan','chebyshev','relative_displacement','absolute_coordinates','component_sizes','topology','count','pair_order')
 def nonperfect(rows,key):
  bins={}
  for x in rows:bins.setdefault(json.dumps(x[key],sort_keys=True),set()).add(x['label'])
  return any(len(v)>1 for v in bins.values())
 for group_name,group in (('selected',selected),('all_admissible',allpairs)):
  if not all(nonperfect(group,key) for key in features):raise RuntimeError('E4F2_PERFECT_SINGLE_FEATURE:'+group_name)
  tuples={}
  for x in group:
   k=tuple(json.dumps(x[key],sort_keys=True) for key in features);tuples.setdefault(k,set()).add(x['label'])
  if not any(len(v)>1 for v in tuples.values()):raise RuntimeError('E4F2_PERFECT_COMBINED_FEATURE_BASELINE:'+group_name)
 for key in features:
  same=sorted(json.dumps(x[key],sort_keys=True) for x in selected if x['label']=='SAME_COMPONENT');different=sorted(json.dumps(x[key],sort_keys=True) for x in selected if x['label']=='DIFFERENT_COMPONENT')
  if same!=different:raise RuntimeError('E4F2_SELECTED_MARGINAL_MISMATCH:'+key)
 return {'selected_pairs':selected,'selected_count':len(selected),'all_admissible_count':len(allpairs),'shortcut_audit':'PASS_EXACT_SELECTED_MARGINAL_MATCH_AND_NO_LISTED_SINGLE_OR_COMBINED_FEATURE_PERFECT'}
def freeze():
 src=json.loads(SOURCE.read_text())['episodes'];records=[];bycount={1:0,2:0,3:0};successor=[]
 for row in src:
  new=copy.deepcopy(row)
  if row['role']!='PRIMARY_CONNECTED_COMPONENT_COUNT':successor.append(new);continue
  count=len(components(row['task']['test'][0]['input']));bycount[count]+=1;color=next(v for rr in row['task']['test'][0]['input'] for v in rr if v);ordinal=bycount[count]-1;variant='P' if ordinal%2==0 else 'N';t,pair=task(count,color,variant,ordinal//2,'E4F2:'+str(row['slot']));new['task']=t;new['source_id']='E4F2_COUNTERFACTUAL:'+str(row['slot']);successor.append(new)
  grid=t['test'][0]['input'];allp=[pair_features(grid,tuple(p))|{'count':count} for p in itertools.combinations(sorted({(y,x) for y,r in enumerate(grid) for x,v in enumerate(r) if v}),2)]
  from scripts.e4d_object_centric_component_anchor_grounding import connected_components as independent_components
  if tuple(sorted(independent_components(grid)))!=components(grid):raise RuntimeError('E4F2_INDEPENDENT_4_CONNECTIVITY_ORACLE')
  records.append({'slot':row['slot'],'count':count,'variant':variant,'task_sha256':hashlib.sha256(canon(t)).hexdigest(),'selected_pairs':select(t,pair) if count>1 else [],'all_pairs':allp})
 if bycount!={1:16,2:16,3:16} or len(successor)!=384:raise RuntimeError('E4F2_SCHEDULE')
 from training_data.pipeline import task_to_sample
 if sum(task_to_sample(x['task'])['assistant_token_count'] for x in successor)!=27360:raise RuntimeError('E4F2_SUPERVISED_TOKEN_BUDGET')
 report=audit(records)
 schedule={'schema_version':1,'status':'FROZEN_TRAIN_DERIVED_COUNTERFACTUAL_GEOMETRIES_NO_MODEL','source_schedule_sha256':sha(SOURCE),'episodes':successor,'roles':{'PRIMARY_CONNECTED_COMPONENT_COUNT':48,'PRIMARY_OBJECT_EXTRACTION':48,'PROTECTED_REPLAY':288},'counterfactual_primary_slots':[r['slot'] for r in records]}
 spec={'schema_version':1,'status':'FROZEN_E4F2_COMPONENT_AFFINITY_SPEC','treated_slots':32,'untreated_count1_slots':[r['slot'] for r in records if r['count']==1],'auxiliary_coefficient':0.25,'count1_auxiliary_coefficient':0.0,'base_answer_coefficient':1.0,'slot_coefficient':0.25,'assistant_supervised_tokens_total':27360,'pair_loss':'one deterministic unordered pair per Count=2/3 slot, exactly balanced and marginal-matched across labels over the matched 32-slot treatment set','permutation_invariant':True,'records':records,'audit':report}
 write(OUT/'SUCCESSOR_COUNTERFACTUAL_SCHEDULE.json',schedule);write(OUT/'PAIR_AFFINITY_SPEC.json',spec)
 receipt={'status':'PASS_E4F2_COUNTERFACTUAL_TRAIN_FREEZE_AND_PAIR_SHORTCUT_AUDIT_NO_MODEL','model_loaded':False,'gpu_used':False,'targets_opened':False,'sidecar_opened':False,'schedule_sha256':sha(OUT/'SUCCESSOR_COUNTERFACTUAL_SCHEDULE.json'),'pair_spec_sha256':sha(OUT/'PAIR_AFFINITY_SPEC.json'),'count_balance':bycount,'treated_slots':32,'selected_pairs':32,'shortcut_audit':report['shortcut_audit']};write(OUT/'CPU_COUNTERFACTUAL_FREEZE_RECEIPT.json',receipt);return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args();print(json.dumps(freeze(),sort_keys=True))
