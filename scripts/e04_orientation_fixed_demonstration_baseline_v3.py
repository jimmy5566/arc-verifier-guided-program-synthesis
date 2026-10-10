"""CPU-only E04 V3 clean fixed-demonstration baseline constructor."""
from __future__ import annotations
import hashlib,json,tempfile,sys
from collections import defaultdict
from pathlib import Path
ROOT_FOR_IMPORT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT_FOR_IMPORT))
from scripts import e04_orientation_marker_counterfactual_v2 as v2

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3'
DECISION=ROOT/'orchestration/director/responses/DIRECTOR_E04_EVALUATION_CONTAMINATION_RECOVERY_V3_DECISION.json'
CONDITIONS=v2.CONDITIONS; TURNS=v2.TURNS
DEMO_CELLS=((0,'EAST'),(1,'SOUTH'),(2,'WEST'),(3,'NORTH'))
class V3Failure(RuntimeError): pass

def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf-8')
def sha_path(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(canon(x)).hexdigest()
def write_json(path,x): path.write_bytes(json.dumps(x,sort_keys=True,indent=2).encode('utf-8')+b'\n')
def write_jsonl(path,rows): path.write_bytes(b''.join(canon(row)+b'\n' for row in rows))

def add_swatch(grid,color):
    result=[row[:] for row in grid]
    if result[8][8] != result[8][7]: raise V3Failure('SWATCH_POSITION_NOT_BACKGROUND')
    result[8][8]=color
    return result

def selected_bank_bases():
    train=v2.canonical_bases('TRAIN')
    by_cell=defaultdict(list)
    for base in train: by_cell[(base['shape_orientation'],base['reference_relation'])].append(base)
    chosen=[]
    for turn,cell in enumerate(DEMO_CELLS):
        candidates=by_cell.get(cell,[])
        if not candidates: raise V3Failure('TRAIN_DEMONSTRATION_CELL_MISSING')
        chosen.append(min(candidates,key=lambda row:row['canonical_base_id']))
    return chosen

def demo_input(base,turn):
    return add_swatch(v2.input_grid(base,turn),base['color_role_tuple'][3])
def query_input(base,turn):
    return add_swatch(v2.input_grid(base,turn),base['color_role_tuple'][3])
def demonstration_bank(condition):
    if condition not in CONDITIONS: raise V3Failure('INVALID_CONDITION')
    return [{'input':demo_input(base,turn),'output':v2.target_grid(base,turn,condition)} for turn,base in enumerate(selected_bank_bases())]

def prompt(base,turn,condition):
    return {'train':demonstration_bank(condition),'test':[{'input':query_input(base,turn)}]}

def target(base,turn,condition,index):
    return {'row_index':index,'canonical_base_id':base['canonical_base_id'],'condition':condition,'control_marker_turn':turn,
            'reserved_cell':{'shape_orientation':base['shape_orientation'],'reference_relation':base['reference_relation']},
            'target':v2.target_grid(base,turn,condition),'output_color':base['color_role_tuple'][3]}

def no_marker(grid):
    result=[row[:] for row in grid]
    bg=result[7][7]
    for col in range(4): result[0][col]=bg
    return result

def validate_bank():
    bank_bases=selected_bank_bases()
    if len({base['canonical_base_id'] for base in bank_bases})!=4: raise V3Failure('DEMONSTRATION_BANK_DUPLICATE')
    for base,cell in zip(bank_bases,DEMO_CELLS):
        if (base['shape_orientation'],base['reference_relation'])!=cell: raise V3Failure('DEMONSTRATION_SELECTOR_CELL')
        if v2.RELATIONS[base['shape_orientation']]==base['reference_relation']: raise V3Failure('RESERVED_CELL_EXPOSURE')
    inputs=[demo_input(base,turn) for turn,base in enumerate(bank_bases)]
    if len({digest(x) for x in inputs})!=4: raise V3Failure('DEMONSTRATION_INPUT_DUPLICATE')
    for condition in CONDITIONS:
        bank=demonstration_bank(condition)
        for turn,pair in enumerate(bank):
            color=bank_bases[turn]['color_role_tuple'][3]
            if pair['input'][8][8]!=color or not any(color in row for row in pair['output']): raise V3Failure('DEMONSTRATION_SWATCH_TARGET_MISMATCH')
    return bank_bases

def validate_rows(prompts,sidecar):
    validation=v2.canonical_bases('VALIDATION')
    if len(validation)!=48 or len(prompts)!=768 or len(sidecar)!=768: raise V3Failure('V3_ROW_COUNT')
    if [x['row_index'] for x in sidecar]!=list(range(768)): raise V3Failure('SIDECAR_INDEX')
    bank_bases=validate_bank()
    by_condition=[]
    by_base_condition=defaultdict(list)
    for i,(task,truth) in enumerate(zip(prompts,sidecar)):
        if set(task)!={'train','test'} or set(task['test'][0])!={'input'}: raise V3Failure('INPUT_PROMPT_STRUCTURE')
        raw=canon(task).decode('utf-8').lower()
        if any(word in raw for word in ('condition','canonical','split','sha','hash','orientation','reference_relation')): raise V3Failure('INPUT_PROMPT_METADATA_LEAKAGE')
        query=task['test'][0]['input']
        prompt_pairs={(digest(pair['input']),digest(pair['output'])) for pair in task['train']}
        if digest(query) in {item[0] for item in prompt_pairs}: raise V3Failure('QUERY_DEMONSTRATION_INPUT_COLLISION')
        if (digest(query),digest(truth['target'])) in prompt_pairs: raise V3Failure('QUERY_DEMONSTRATION_PAIR_COLLISION')
        if query[8][8]!=truth['output_color'] or not any(truth['output_color'] in row for row in truth['target']): raise V3Failure('QUERY_SWATCH_TARGET_MISMATCH')
        by_base_condition[(truth['canonical_base_id'],truth['condition'])].append((task,truth))
        if truth['control_marker_turn']==0: by_condition.append((truth['canonical_base_id'],truth['condition'],digest(task['train'])))
    for group in by_base_condition.values():
        if {truth['control_marker_turn'] for _,truth in group}!=set(TURNS): raise V3Failure('COUNTERFACTUAL_TURNS_MISSING')
        masks={digest(no_marker(task['test'][0]['input'])) for task,_ in group}
        if len(masks)!=1: raise V3Failure('QUERY_MARKER_COUNTERFACTUAL_FAILURE')
    # fixed, ordered bank per condition for every validation query
    expected={condition:digest(demonstration_bank(condition)) for condition in CONDITIONS}
    for task,truth in zip(prompts,sidecar):
        if digest(task['train'])!=expected[truth['condition']]: raise V3Failure('DEMONSTRATION_BANK_NOT_FIXED')
    per_base=defaultdict(dict)
    for base,condition,bank_hash in by_condition: per_base[base][condition]=bank_hash
    if any(set(row)!=set(CONDITIONS) for row in per_base.values()): raise V3Failure('CONDITION_COVERAGE')
    if any(len(set(row.values()))!=4 for row in per_base.values()): raise V3Failure('CONDITION_TASK_UNOBSERVABLE')
    if len({row['output_color'] for row in sidecar})<2: raise V3Failure('VALIDATION_OUTPUT_COLOR_DIVERSITY')

def freeze(destination=OUT):
    if destination.exists(): raise V3Failure('REFUSE_OVERWRITE_FROZEN_V3')
    if not DECISION.is_file(): raise V3Failure('V3_DECISION_MISSING')
    decision=json.loads(DECISION.read_text(encoding='utf-8'))
    if decision.get('response_id')!='DIRECTOR_E04_EVALUATION_CONTAMINATION_RECOVERY_V3_DECISION': raise V3Failure('V3_DECISION_IDENTITY')
    validate_bank()
    prompts=[]; sidecar=[]
    for base in v2.canonical_bases('VALIDATION'):
        for turn in TURNS:
            for condition in CONDITIONS:
                prompts.append(prompt(base,turn,condition))
                sidecar.append(target(base,turn,condition,len(sidecar)))
    validate_rows(prompts,sidecar)
    destination.mkdir(parents=True)
    write_jsonl(destination/'VALIDATION_INPUT_PROMPTS.jsonl',prompts)
    write_jsonl(destination/'TARGET_SCORER_SIDECAR.jsonl',sidecar)
    bank={'selector':'lexicographically smallest canonical_base_id in each prospectively fixed TRAIN-only cell','cells':[{'turn':t,'shape_orientation':o,'reference_relation':r,'canonical_base_id':b['canonical_base_id']} for t,((o,r),b) in enumerate(zip(DEMO_CELLS,selected_bank_bases()))], 'condition_bank_sha256':{c:digest(demonstration_bank(c)) for c in CONDITIONS}}
    write_json(destination/'FIXED_DEMONSTRATION_BANK.json',bank)
    manifest={'protocol_id':'E04_ORIENTATION_V3_FIXED_INDEPENDENT_DEMONSTRATION_BASELINE','status':'CPU_ONLY_INPUTS_FROZEN_TARGETS_SEPARATE','director_decision_sha256':sha_path(DECISION),'v2_preserved_reference_commit':'50012fd8379dbbef22997d66bf626a911f9e5a2f','canonical_validation_bases':48,'condition_turn_views_per_base':16,'validation_prompts':768,'input_prompts_sha256':sha_path(destination/'VALIDATION_INPUT_PROMPTS.jsonl'),'target_scorer_sidecar_sha256':sha_path(destination/'TARGET_SCORER_SIDECAR.jsonl'),'bank_sha256':sha_path(destination/'FIXED_DEMONSTRATION_BANK.json'),'boundaries':['NO_GOLD','NO_DGOLD','NO_FINAL_AUDIT','NO_MODEL_LOADING_DURING_CPU_FREEZE','NO_GPU_DURING_CPU_FREEZE','NO_TRAINING']}
    write_json(destination/'MANIFEST.json',manifest)
    return manifest

def validate_frozen(destination=OUT):
    prompts=[json.loads(line) for line in (destination/'VALIDATION_INPUT_PROMPTS.jsonl').read_text(encoding='utf-8').splitlines()]
    sidecar=[json.loads(line) for line in (destination/'TARGET_SCORER_SIDECAR.jsonl').read_text(encoding='utf-8').splitlines()]
    validate_rows(prompts,sidecar)
    return {'status':'PASS','model_loaded':False,'gpu_used':False,'optimizer_steps':0,'validation_prompts':len(prompts),'all_mandatory_cpu_gates':'PASS'}

def regeneration(destination=OUT):
    with tempfile.TemporaryDirectory() as tmp:
        temp=Path(tmp)/'v3'; freeze(temp)
        names=('VALIDATION_INPUT_PROMPTS.jsonl','TARGET_SCORER_SIDECAR.jsonl','FIXED_DEMONSTRATION_BANK.json','MANIFEST.json')
        checks={n:sha_path(destination/n)==sha_path(temp/n) for n in names}
    report={'status':'PASS' if all(checks.values()) else 'FAIL','byte_identical':all(checks.values()),'files':checks,'serialization':'UTF-8 canonical JSONL with LF'}
    write_json(destination/'DETERMINISTIC_REGENERATION_REPORT.json',report); return report

if __name__=='__main__':
    print(json.dumps({'manifest':freeze(),'validation':validate_frozen(),'regeneration':regeneration()},sort_keys=True))
