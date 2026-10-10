from __future__ import annotations
import hashlib,json
from pathlib import Path

RELATIONS=('NORTH','EAST','SOUTH','WEST')
LAYOUTS=('CENTER','NW','NE','SW','SE','MID_WEST','MID_EAST','MID_NORTH','MID_SOUTH','OFFSET_A','OFFSET_B','OFFSET_C')
CONDITIONS=('ROTATION_TARGET','FIXED_TURN_ROTATION_CONTROL','MARKER_BINDING_CONTROL','NO_TRANSFORM_RETENTION_CONTROL')

def latent_signature(row: dict) -> str:
    value={k:row[k] for k in ('split','shape_orientation','control_marker_turn','reference_relation','topology','layout_template','color_role_permutation','instance_index')}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def fixture_seed(index: int) -> int:
    """Separate non-scientific test namespace; never a cohort seed."""
    return 910_000_000 + index

def enumerate_plan(split: str) -> list[dict]:
    if split not in {'TRAIN','VALIDATION'}: raise ValueError(split)
    rows=[]
    for o in range(4):
        for r_i, relation in enumerate(RELATIONS):
            reserved=RELATIONS[o]==relation
            if (split=='VALIDATION') != reserved: continue
            count=12 if reserved else 24
            for i in range(count):
                row={'split':split,'shape_orientation':o,'control_marker_turn':i%4,'reference_relation':relation,'topology':'single_4_connected_L_TRIOMINO','layout_template':LAYOUTS[i%12],'color_role_permutation':f'P{i:02d}','instance_index':i,'seed':810000+104729*(o*4+r_i)+7919*i}
                row['latent_signature_sha256']=latent_signature(row); rows.append(row)
    return rows

def validate_plan(train: list[dict], validation: list[dict]) -> dict:
    assert len(train)==288 and len(validation)==48
    assert len({x['latent_signature_sha256'] for x in train})==288
    assert len({x['latent_signature_sha256'] for x in validation})==48
    assert not ({x['latent_signature_sha256'] for x in train}&{x['latent_signature_sha256'] for x in validation})
    assert len({x['seed'] for x in train+validation})==336
    assert {(x['shape_orientation'],x['reference_relation']) for x in validation}=={(i,RELATIONS[i]) for i in range(4)}
    assert all(sum(x['shape_orientation']==o and x['reference_relation']==r for x in train)==24 for o in range(4) for r in RELATIONS if RELATIONS[o]!=r)
    assert {x['shape_orientation'] for x in train}==set(range(4)) and {x['reference_relation'] for x in train}==set(RELATIONS)
    return {'train_base_tuples':288,'validation_base_tuples':48,'paired_condition_views_per_base_tuple':4,'scientific_grids_rendered':False,'scientific_content_manifests':'NOT_RUN','grid_uniqueness':'NOT_RUN','target_correctness':'NOT_RUN','byte_identical_regeneration':'NOT_RUN'}

def main() -> int:
    train,validation=enumerate_plan('TRAIN'),enumerate_plan('VALIDATION')
    print(json.dumps(validate_plan(train,validation),sort_keys=True));return 0
if __name__=='__main__': raise SystemExit(main())
