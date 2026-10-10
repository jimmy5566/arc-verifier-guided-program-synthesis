#!/usr/bin/env python3
"""CPU-only freeze for MATCHED_ATOMIC_TO_COMPOSITIONAL_TRANSFER_V1.

Creates public ARC-style observations and a local sealed target sidecar.  No
model, tokenizer, torch, optimizer, or GPU dependency is imported.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='MATCHED_ATOMIC_TO_COMPOSITIONAL_TRANSFER_V1'
SEED=20261011
TUPLES=48
RETENTION=24
V7='experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json'
FB='experiments/capability_repair_baseline_v1/family_balanced_loss_control_v1/round_001/evaluation_v1/FINAL_CHECKPOINT_MANIFEST.json'

def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha_bytes(x): return hashlib.sha256(x).hexdigest()
def sha_file(p): return sha_bytes(p.read_bytes())
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_bytes(json.dumps(x,sort_keys=True,indent=2).encode()+b'\n'); os.replace(tmp,p)
def grid(): return [[0]*9 for _ in range(9)]
def clone(g): return [r[:] for r in g]
def put_shape(g,y,x,c):
 for dy,dx in ((0,0),(1,0),(1,1),(2,0)): g[y+dy][x+dx]=c
def components(g,c):
 seen=set(); out=[]
 for y in range(9):
  for x in range(9):
   if g[y][x]!=c or (y,x) in seen: continue
   q=[(y,x)]; seen.add((y,x)); comp=[]
   while q:
    a,b=q.pop(); comp.append((a,b))
    for da,db in ((1,0),(-1,0),(0,1),(0,-1)):
     na,nb=a+da,b+db
     if 0<=na<9 and 0<=nb<9 and g[na][nb]==c and (na,nb) not in seen: seen.add((na,nb));q.append((na,nb))
   out.append(comp)
 return out
def select_component(g,source,target):
 markers=[(y,x) for y in range(9) for x in range(9) if g[y][x]==target]
 if len(markers)!=1: raise ValueError('MARKER_IDENTITY_INVALID')
 my,mx=markers[0]
 matches=[comp for comp in components(g,source) if (my+1,mx) in comp]
 if len(matches)!=1: raise ValueError('SELECTOR_RELATION_INVALID')
 return matches[0]
def transform(g,source,target,kind):
 selected=select_component(g,source,target)
 out=grid()
 if kind=='ATOMIC_SELECT':
  for y,x in selected: out[y][x]=source
 elif kind=='ATOMIC_RECOLOR':
  for y in range(9):
   for x in range(9): out[y][x]=target if g[y][x] in (source,target) else 0
 elif kind=='COMPOSITION_SELECT_RECOLOR':
  for y,x in selected: out[y][x]=target
 else: raise ValueError('UNKNOWN_KIND')
 return out
def example(source,target,variant):
 # Selected component has its unique target-color marker immediately above.
 positions=((2,1,5,5),(3,4,1,1),(4,2,1,5))
 sy,sx,dy,dx=positions[variant]
 g=grid(); put_shape(g,sy,sx,source); put_shape(g,dy,dx,source); g[sy-1][sx]=target
 return g
def task_for(source,target,kind):
 pairs=[]
 for v in (0,1):
  inp=example(source,target,v); pairs.append({'input':inp,'output':transform(inp,source,target,kind)})
 test=example(source,target,2)
 return {'train':pairs,'test':[{'input':test}]},transform(test,source,target,kind)
def retention_task(source,target,variant):
 g=grid(); y,x=((2,2),(3,4),(4,1))[variant%3]; put_shape(g,y,x,source)
 pairs=[]
 for off in (0,1):
  inp=clone(g)
  if off: inp=grid(); put_shape(inp,1+y%3,1+x%4,source)
  pairs.append({'input':inp,'output':[[target if z==source else z for z in r] for r in inp]})
 return {'train':pairs,'test':[{'input':g}]},[[target if z==source else z for z in r] for r in g]
def native(g): return '\n'.join(''.join(str(x) for x in r) for r in g)
def prompt(task):
 out=[]
 for pair in task['train']: out += [f'<|im_start|>user\n{native(pair["input"])}<|im_end|>',f'<|im_start|>assistant\n{native(pair["output"])}<|im_end|>']
 out += [f'<|im_start|>user\n{native(task["test"][0]["input"])}<|im_end|>','<|im_start|>assistant\n']
 return ''.join(out)
def validate_task(task,target,source,target_color,kind):
 for pair in task['train']:
  if transform(pair['input'],source,target_color,kind)!=pair['output']: raise ValueError('TRAIN_TRANSFORM_INVALID')
 if transform(task['test'][0]['input'],source,target_color,kind)!=target: raise ValueError('TEST_TRANSFORM_INVALID')
def historical_exposure_audit(rows):
    candidates=(
        ROOT/'experiments/capability_repair_baseline_v1/unified_native_model_capability_baseline_v1/SYNTHETIC_BENCHMARK_INPUT_MANIFEST_V1.json',
        ROOT/'experiments/capability_repair_baseline_v1/e03_v7_lora_gradient_interference_diagnostic_v1/E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN_MANIFEST_V1.json',
    )
    current_hashes={row['observation_sha256'] for row in rows}; current_ids={row['episode_id'] for row in rows}
    observed_hashes=set(); observed_ids=set(); checked=[]
    for path in candidates:
        if not path.is_file():
            checked.append({'path':str(path.relative_to(ROOT)).replace('\\','/'),'status':'MISSING_NOT_USED'}); continue
        doc=json.loads(path.read_text(encoding='utf-8-sig')); entries=list(doc.get('episodes',[]))
        if not entries:
            for family in doc.get('families',[]): entries += family.get('members',[])
        for entry in entries:
            if isinstance(entry,dict):
                if entry.get('observation_sha256'): observed_hashes.add(entry['observation_sha256'])
                if entry.get('episode_id'): observed_ids.add(entry['episode_id'])
        checked.append({'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':sha_file(path),'entries_checked':len(entries)})
    overlap_hashes=sorted(current_hashes & observed_hashes); overlap_ids=sorted(current_ids & observed_ids)
    if overlap_hashes or overlap_ids: raise ValueError('HISTORICAL_CONTENT_EXPOSURE_OVERLAP')
    return {'sources':checked,'exact_observation_hash_overlap_count':0,'exact_episode_id_overlap_count':0,'passed':True}

def adapter_sha(path):
 x=json.loads(path.read_text(encoding='utf-8-sig')); return next((z.get('sha256') for z in x.get('adapter_files',[]) if z.get('name')=='adapter_model.safetensors'),None)
def prepare(out,sealed,seed=SEED,tuples=TUPLES,retention=RETENTION):
 if tuples!=48 or retention!=24: raise ValueError('FROZEN_COHORT_SIZE_DRIFT')
 rng=random.Random(seed); rows=[]; targets={}; latent=[]
 colors=[1,2,3,4]; targets_colors=[5,6,7,8,9]
 for i in range(tuples):
  source=colors[rng.randrange(len(colors))]; target=targets_colors[rng.randrange(len(targets_colors))]
  latent_tuple={'tuple_id':f'T{i:03d}','source_color':source,'target_color':target,'selector_relation':'UNIQUE_TARGET_COLOR_MARKER_DIRECTLY_ABOVE_SOURCE_COMPONENT','component_shape':'L4','layouts':['2,1;5,5','3,4;1,1','4,2;1,5']}
  latent_tuple['sha256']=sha_bytes(canonical(latent_tuple)); latent.append(latent_tuple)
  for kind in ('ATOMIC_SELECT','ATOMIC_RECOLOR','COMPOSITION_SELECT_RECOLOR'):
   task,answer=task_for(source,target,kind); validate_task(task,answer,source,target,kind)
   eid=f'{PROTOCOL}:TUPLE:{i:03d}:{kind}'; obs={'episode_id':eid,'tuple_id':latent_tuple['tuple_id'],'role':kind,'task':task}
   rows.append({'episode_id':eid,'tuple_id':latent_tuple['tuple_id'],'role':kind,'split':'SYNTHETIC_TRANSFER','target_access':'SEALED_SIDECAR_ONLY','observation':obs,'observation_sha256':sha_bytes(canonical(obs)),'prompt_sha256':sha_bytes(prompt(task).encode())});targets[eid]=answer
 for i in range(retention):
  source=colors[rng.randrange(len(colors))];target=targets_colors[rng.randrange(len(targets_colors))];task,answer=retention_task(source,target,i)
  eid=f'{PROTOCOL}:RETENTION:{i:03d}';obs={'episode_id':eid,'role':'PROTECTED_RECOLOR_RETENTION','task':task}
  rows.append({'episode_id':eid,'tuple_id':None,'role':'PROTECTED_RECOLOR_RETENTION','split':'SYNTHETIC_RETENTION','target_access':'SEALED_SIDECAR_ONLY','observation':obs,'observation_sha256':sha_bytes(canonical(obs)),'prompt_sha256':sha_bytes(prompt(task).encode())});targets[eid]=answer
 if len(rows)!=168 or len({r['episode_id'] for r in rows})!=168 or len({r['observation_sha256'] for r in rows})!=168: raise ValueError('COHORT_UNIQUENESS_INVALID')
 sidecar=sealed/'MATCHED_ATOMIC_TO_COMPOSITIONAL_TRANSFER_V1_TARGET_SIDECAR.json'; dump(sidecar,{'protocol_id':PROTOCOL,'targets':targets})
 v7,fb=ROOT/V7,ROOT/FB
 if not v7.is_file() or not fb.is_file(): raise ValueError('CHECKPOINT_MANIFEST_MISSING')
 # No target values are included in public artifacts.
 batch1_tuples=[f'T{i:03d}' for i in range(0,48,4)]
 batch1_retention=[f'{PROTOCOL}:RETENTION:{i:03d}' for i in range(0,24,4)]
 manifest={'schema_version':1,'protocol_id':PROTOCOL,'status':'CPU_FROZEN_TARGETS_SEALED','generator':{'id':'MATCHED_SELECTOR_RECOLOR_GENERATOR_V1','seed':seed,'source_sha256':sha_file(Path(__file__))},'tuple_count':48,'retention_count':24,'episodes':rows,'latent_parameter_tuples':latent,'fixed_batch1_validation':{'tuple_ids':batch1_tuples,'retention_episode_ids':batch1_retention,'episode_count':42,'selection':'every fourth tuple and retention row in frozen canonical order'},'content_exposure_audit':{'new_generator_namespace':True,'independent_seed':seed,'historical_arc_episode_ids_reused':False,'tuple_parameter_hashes_unique':len({x['sha256'] for x in latent})==48,'public_observation_hashes_unique':True,'historical_exact_content_check':historical_exposure_audit(rows),'overlap_rule':'no exact public observation SHA256 or episode id may overlap checked historical sources; no tuple parameter SHA256 may duplicate within this cohort','Gold_dGold_FINAL_AUDIT_accessed':False},'sealed_target_sidecar':{'local_path':str(sidecar.resolve()),'sha256':sha_file(sidecar),'target_count':168,'not_committed':True}}
 dump(out/'COHORT_MANIFEST.json',manifest)
 protocol={'schema_version':1,'experiment_id':PROTOCOL,'status':'CPU_FROZEN_NOT_AUTHORIZED_FOR_MODEL_EXECUTION','cohort_manifest_path':'COHORT_MANIFEST.json','cohort_manifest_sha256':sha_file(out/'COHORT_MANIFEST.json'),'checkpoints':{'RECONSTRUCTED_FOUNDATION_V2_V7':{'manifest_path':V7,'manifest_sha256':sha_file(v7),'adapter_model_sha256':adapter_sha(v7),'role':'PRIMARY'},'FAMILY_BALANCED':{'manifest_path':FB,'manifest_sha256':sha_file(fb),'adapter_model_sha256':adapter_sha(fb),'role':'FROZEN_HISTORICAL_INTERVENTION_COMPARATOR'}},'inference':{'format':'NATIVE_ARC_OBSERVATION','decoding':'GREEDY_PRIMARY; COMPLETE_OUTPUT_TOP2_SINGLE_FROZEN_TARGET_BLIND_ALTERNATIVE','batch_primary':16,'fallback_ladder':[16,8,4,1],'batch1_validation':'fixed 42-episode subset; parsed outputs and all checkpoint-level metrics must agree exactly or INVALID_NOT_INTERPRETABLE','forbidden':['Beam','DFS','TTT','augmentation','external_selector','Gold','dGold','FINAL_AUDIT']},'metrics':{'primary':'per-checkpoint composition exact rate conditional on BOTH atomic outputs exact, then paired V7-versus-Family-Balanced contrast','secondary':['joint_atomic_prerequisite_exact_rate','unconditional_composition_exact_rate','complete_output_top2_composition_coverage','four_state_tuple_counts','separate_synthetic_protected_retention']},'uncertainty':{'bootstrap':'10000 fixed seed 20261010 stratified paired bootstrap by 4 frozen tuple index strata','minimum_both_atoms_correct_tuples_per_checkpoint':12,'practically_negligible_conditional_failure_bound':0.15},'decision_rule':{'COMPOSITIONAL_TRANSFER_BOTTLENECK_SUPPORTED':'minimum n met and V7 conditional composition-failure bootstrap lower bound >0.15','ATOMIC_PREREQUISITE_LIMITATION_SUPPORTED':'minimum n met, conditional failure bootstrap upper bound <=0.15, and >0.70 of composition failures have at least one atomic prerequisite wrong','MIXED_OR_INCONCLUSIVE':'minimum n unmet, interval crosses 0.15, checkpoint direction unstable, or Batch1 validation changes classification','INVALID_NOT_INTERPRETABLE':'identity, exposure, tuple matching, serialization, target, checkpoint, batching, parser, completeness, or sealed-data failure'},'runtime_cap_seconds':1800,'optimizer_steps':0,'parameter_updates':0,'model_execution_authorized':False,'training_authorized':False}
 dump(out/'PROTOCOL.json',protocol)
 return {'manifest':out/'COHORT_MANIFEST.json','protocol':out/'PROTOCOL.json','sidecar':sidecar}
def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--sealed-root',type=Path,required=True);a.add_argument('--seed',type=int,default=SEED);z=a.parse_args();
 if z.out.exists(): raise SystemExit('OUTPUT_ROOT_MUST_BE_FRESH')
 r=prepare(z.out,z.sealed_root,z.seed);print(json.dumps({k:{'path':str(v),'sha256':sha_file(v)} for k,v in r.items()},sort_keys=True))
if __name__=='__main__': main()
