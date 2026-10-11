"""CPU-only E05 V3 three-arm native-Greedy execution contract."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery'
BINDING=OUT/'E05_V3_THREE_ARM_EXECUTION_BINDING_REPAIR.json'
PROMPTS=OUT/'INPUT_PROMPTS.jsonl'
class E05Failure(RuntimeError): pass
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def validate_preimport(binding_path:BINDING.__class__=BINDING):
 b=json.loads(Path(binding_path).read_text()); m=json.loads((OUT/'MANIFEST.json').read_text())
 if b['protocol_id']!=m['protocol_id'] or b['active_hard_cap_seconds']!=10800 or b['physical_batch_size']!=1 or b['decoder']!='native_greedy': raise E05Failure('E05_BINDING_RUNTIME')
 if b['updates']!=0 or b['optimizer'] or b['backward']: raise E05Failure('E05_UPDATE_BOUNDARY')
 arms=b['arms']; ids=[x['arm_id'] for x in arms]
 if ids!=['RAW_V7_NO_E04E_UPDATE','E04E_RUN003_CONTROL_EQUAL_SLOT','E04E_RUN003_TREATMENT_EQUAL_SLOT']: raise E05Failure('E05_ARM_ORDER')
 if arms[0]['adapter_sha256']!='1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3': raise E05Failure('E05_RAW_V7_IDENTITY')
 if [x['adapter_sha256'] for x in arms[1:]]!=['515f5ebfe6685097f78fe260edcc4859edfe17152482a559769f18be8b55da16','3eff5b9debc4b5b3f133e4524bf7244b0a2074a8f0bc0b5595d38ee176db57be']: raise E05Failure('E05_E04E_IDENTITIES')
 r=rows(PROMPTS)
 if len(r)!=768 or sha(PROMPTS)!=m['prompt_sha256']: raise E05Failure('E05_PROMPT_IDENTITY')
 if any(set(x)!={'episode_id','prompt'} for x in r): raise E05Failure('E05_INPUT_SCHEMA')
 if any('target' in json.dumps(x['prompt']).lower() for x in r): raise E05Failure('E05_TARGET_LEAK')
 return {'status':'PASS_E05_V3_THREE_ARM_PREIMPORT','rows':768,'arms':ids,'model_imported':False,'gpu_used':False,'target_sidecar_accessed':False,'optimizer_steps':0}
def validate_raw(path:Path, arm_id:str):
 r=rows(path)
 if len(r)!=768 or [x.get('row_index') for x in r]!=list(range(768)): raise E05Failure('E05_RAW_COVERAGE')
 expected={hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest() for x in rows(PROMPTS)}
 for x in r:
  if x.get('arm_id')!=arm_id or x.get('effective_batch_size')!=1 or x.get('input_sha256') not in expected: raise E05Failure('E05_RAW_BINDING')
 return {'status':'PASS_E05_V3_RAW','rows':768}
if __name__=='__main__': print(json.dumps(validate_preimport(),sort_keys=True))
