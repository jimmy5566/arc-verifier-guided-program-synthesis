from __future__ import annotations
import hashlib,json
from itertools import islice, permutations
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DESIGN=ROOT/'experiments/capability_repair_baseline_v1/E04_ORIENTATION_GREENFIELD_GENERATOR_DESIGN_SPEC_V2.json'
PREFREEZE=ROOT/'experiments/capability_repair_baseline_v1/CPU_ONLY_E04_ORIENTATION_GREENFIELD_PREFREEZE_SPECIFICATION_V2.json'
DESIGN_SHA='1fd3592840195c76a101bdb8901dfdd07544ebb183cc1f62030766df24b347f2'
PREFREEZE_SHA='139bd6cb6ecb16a68688d827d224fa7a3282d60c85380ec81a23efdf702b3a13'
RELATIONS=('NORTH','EAST','SOUTH','WEST'); LAYOUTS=('CENTER','NW','NE','SW','SE','MID_WEST','MID_EAST','MID_NORTH','MID_SOUTH','OFFSET_A','OFFSET_B','OFFSET_C')
CONDITIONS=('ROTATION_TARGET','FIXED_TURN_ROTATION_CONTROL','MARKER_BINDING_CONTROL','NO_TRANSFORM_RETENTION_CONTROL')
SCIENTIFIC_FIELDS=('shape_orientation','control_marker_turn','reference_relation','topology','layout_template','color_role_tuple','instance_index')
class PreflightFailure(RuntimeError): pass

def digest(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def fixture_seed(index:int)->int: return 910_000_000+index
def bind_frozen_inputs(design:Path=DESIGN,prefreeze:Path=PREFREEZE)->None:
    if digest(design)!=DESIGN_SHA: raise PreflightFailure('DESIGN_SHA_MISMATCH')
    if digest(prefreeze)!=PREFREEZE_SHA: raise PreflightFailure('PREFREEZE_SHA_MISMATCH')
def semantic_signature(row:dict)->str:
    try: value={k:row[k] for k in SCIENTIFIC_FIELDS}
    except KeyError as e: raise PreflightFailure(f'MISSING_SCIENTIFIC_FIELD:{e.args[0]}')
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def color_tuple(index:int)->tuple[int,int,int,int]:
    if index < 0: raise PreflightFailure('INVALID_COLOR_INDEX')
    return next(islice(permutations(range(1,10),4), index, None))
def enumerate_plan(split:str)->list[dict]:
    if split not in {'TRAIN','VALIDATION'}: raise PreflightFailure('INVALID_SPLIT')
    rows=[]
    for o in range(4):
      for ri,rel in enumerate(RELATIONS):
        reserved=RELATIONS[o]==rel
        if (split=='VALIDATION')!=reserved: continue
        for i in range(12 if reserved else 24):
          row={'split':split,'shape_orientation':o,'control_marker_turn':i%4,'reference_relation':rel,'topology':'single_4_connected_L_TRIOMINO','layout_template':LAYOUTS[i%12],'color_role_tuple':color_tuple(i),'instance_index':i,'seed':810000+104729*(o*4+ri)+7919*i}
          row['semantic_signature_sha256']=semantic_signature(row);rows.append(row)
    return rows
def fail(cond:bool,code:str)->None:
    if not cond: raise PreflightFailure(code)
def validate_plan(train:list[dict],validation:list[dict])->dict:
    fail(len(train)==288,'TRAIN_COUNT');fail(len(validation)==48,'VALIDATION_COUNT')
    for split,rows,count in (('TRAIN',train,24),('VALIDATION',validation,12)):
      for row in rows:
        fail(row.get('split')==split,'WRONG_SPLIT'); fail(tuple(row.get('color_role_tuple',())) in {color_tuple(i) for i in range(count)},'INVALID_COLOR_ROLE')
        fail(len(set(row['color_role_tuple']))==4 and all(1<=x<=9 for x in row['color_role_tuple']),'INVALID_COLOR_ROLE')
      cells={(x['shape_orientation'],x['reference_relation']) for x in rows}
      expected={(o,r) for o in range(4) for r in RELATIONS if (RELATIONS[o]!=r)==(split=='TRAIN')}
      fail(cells==expected,'STRUCTURAL_CELL_MISMATCH')
      for cell in expected:
        sub=[x for x in rows if (x['shape_orientation'],x['reference_relation'])==cell];fail(len(sub)==count,'CELL_COUNT')
        turns=[sum(x['control_marker_turn']==t for x in sub) for t in range(4)];fail(turns==([6]*4 if split=='TRAIN' else [3]*4),'MARKER_IMBALANCE')
        layouts=[sum(x['layout_template']==q for x in sub) for q in LAYOUTS];fail(layouts==([2]*12 if split=='TRAIN' else [1]*12),'LAYOUT_IMBALANCE')
        fail(len({tuple(x['color_role_tuple']) for x in sub})==count,'COLOR_DUPLICATE')
    seeds=[x['seed'] for x in train+validation]; sigs=[x['semantic_signature_sha256'] for x in train+validation]
    fail(len(seeds)==len(set(seeds)),'DUPLICATE_SEED');fail(len(sigs)==len(set(sigs)),'SEMANTIC_OVERLAP')
    fail(enumerate_plan('TRAIN')==train and enumerate_plan('VALIDATION')==validation,'NONDETERMINISTIC_ENUMERATION')
    return {'train_base_tuples':288,'validation_base_tuples':48,'paired_condition_views_per_base_tuple':4,'scientific_grids_rendered':False,'scientific_content_manifests':'NOT_RUN','grid_uniqueness':'NOT_RUN','target_correctness':'NOT_RUN','byte_identical_regeneration':'NOT_RUN'}
def main()->int:
    bind_frozen_inputs();print(json.dumps(validate_plan(enumerate_plan('TRAIN'),enumerate_plan('VALIDATION')),sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
