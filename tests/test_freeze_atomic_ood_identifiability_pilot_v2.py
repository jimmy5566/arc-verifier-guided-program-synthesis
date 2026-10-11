from __future__ import annotations
import importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v2'
spec=importlib.util.spec_from_file_location('v2',ROOT/'scripts/freeze_atomic_ood_identifiability_pilot_v2.py');m=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(m)
def test_every_frozen_prompt_passes_actual_native_serializer_without_model():
 rows=[json.loads(x) for x in (OUT/'INPUT_PROMPTS.jsonl').read_text().splitlines()]
 assert len(rows)==192
 for row in rows:
  task=row['task'];assert set(task)=={'train','test'} and len(task['test'])==1 and set(task['test'][0])=={'input'}
  assert all(set(pair)=={'input','output'} for pair in task['train'])
  assert m.native_prompt(task).endswith('<|im_start|>assistant\n')
def test_reconstruct_audits_from_frozen_prompt_and_sidecar_bytes():
 prompts=[json.loads(x) for x in (OUT/'INPUT_PROMPTS.jsonl').read_text().splitlines()];targets={x['episode_id']:x['target'] for x in (json.loads(x) for x in (OUT/'SEALED_TARGET_SIDECAR.jsonl').read_text().splitlines())}
 rows=[{'episode_id':x['episode_id'],'cell':x['cell'],'canonical_unit':x['canonical_unit'],'turn':x['turn'],'task':x['task'],'target':targets[x['episode_id']]} for x in prompts]
 rotation,one,collision,blind=m.audits(rows)
 assert rotation['status']==one['status']==collision['status']==blind['status']=='PASS'
 assert rotation['d4_classes']==8 and collision['grid_identity_intersection']==0
def test_frozen_v2_refuses_overwrite():
 p=subprocess.run([sys.executable,str(ROOT/'scripts/freeze_atomic_ood_identifiability_pilot_v2.py')],capture_output=True,text=True)
 assert p.returncode and 'REFUSE_OVERWRITE_FROZEN_ATOMIC_OOD_V2' in p.stderr
