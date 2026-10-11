"""No-model validation of the immutable E05 V2 input boundary."""
from __future__ import annotations
import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/capability_repair_baseline_v1/e05_v2_identifiable_prompt_and_asset_identity_repair'
CONDS={'FIXED_TURN_ROTATION_CONTROL','MARKER_BINDING_CONTROL','NO_TRANSFORM_RETENTION_CONTROL','MARKER_CONDITIONED_ROTATION'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 manifest=json.loads((OUT/'MANIFEST.json').read_text()); prompts=[json.loads(x) for x in (OUT/'INPUT_PROMPTS.jsonl').read_text().splitlines()]
 if len(prompts)!=768 or sha(OUT/'INPUT_PROMPTS.jsonl')!=manifest['prompt_sha256']:raise RuntimeError('V2_PROMPT_IDENTITY')
 if any(set(x)!= {'episode_id','prompt'} or 'target' in json.dumps(x['prompt']).lower() for x in prompts):raise RuntimeError('V2_PROMPT_TARGET_LEAK')
 visible={json.dumps(x['prompt'],sort_keys=True,separators=(',',':')) for x in prompts}
 if len(visible)!=768:raise RuntimeError('V2_VISIBLE_PROMPT_COLLISION')
 target_path=OUT/'TARGET_SCORER_SIDECAR.jsonl'
 if not target_path.is_file() or target_path.stat().st_size==0:raise RuntimeError('V2_TARGET_SIDECAR_MISSING')
 # The freezer fixed the target hash atomically.  This input-only validator
 # deliberately does not parse or hash that sealed sidecar.
 result={'status':'PASS_CPU_ONLY_E05_V2_VALIDATION','model_loaded':False,'gpu_used':False,'generation_run':False,'sealed_e04_target_sidecar_opened':False,'input_prompts':768,'unique_visible_prompts':768,'target_sidecar_semantically_opened_for_validation':False,'prompt_sha256':manifest['prompt_sha256'],'target_sidecar_sha256':manifest['target_sidecar_sha256']}
 (OUT/'CPU_FREEZE_VALIDATION.json').write_bytes(json.dumps(result,sort_keys=True,indent=2).encode()+b'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
