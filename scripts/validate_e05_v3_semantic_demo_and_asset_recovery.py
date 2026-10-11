"""Input-only validation for frozen E05 V3; does not parse target sidecar."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 m=json.loads((OUT/'MANIFEST.json').read_text());rows=[json.loads(x) for x in (OUT/'INPUT_PROMPTS.jsonl').read_text().splitlines()]
 if len(rows)!=768 or sha(OUT/'INPUT_PROMPTS.jsonl')!=m['prompt_sha256']:raise RuntimeError('V3_PROMPT_IDENTITY')
 if any(set(x)!={'episode_id','prompt'} or 'target' in json.dumps(x['prompt']).lower() for x in rows):raise RuntimeError('V3_TARGET_LEAK')
 if len({json.dumps(x['prompt'],sort_keys=True,separators=(',',':')) for x in rows})!=768:raise RuntimeError('V3_VISIBLE_PROMPT_COLLISION')
 semantic=json.loads((OUT/'SEMANTIC_IDENTIFIABILITY_AUDIT.json').read_text())
 for c,v in semantic['condition_oracle_replay'].items():
  if v['oracle_replay']!='PASS':raise RuntimeError('V3_DEMO_ORACLE')
 if not (OUT/'TARGET_SCORER_SIDECAR.jsonl').is_file():raise RuntimeError('V3_TARGET_MISSING')
 result={'status':'PASS_CPU_ONLY_E05_V3_VALIDATION','input_prompts':768,'unique_visible_prompts':768,'semantic_oracle_replay':'PASS','target_sidecar_semantically_opened_for_validation':False,'model_loaded':False,'gpu_used':False,'generation_run':False,'prompt_sha256':m['prompt_sha256'],'target_sidecar_sha256':m['target_sidecar_sha256']}
 (OUT/'CPU_FREEZE_VALIDATION.json').write_bytes(json.dumps(result,sort_keys=True,indent=2).encode()+b'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
