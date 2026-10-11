from __future__ import annotations
import importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v3'
spec=importlib.util.spec_from_file_location('v3',ROOT/'scripts/freeze_atomic_ood_identifiability_pilot_v3.py');m=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(m)
def test_every_frozen_prompt_passes_actual_native_serializer_without_model():
 rows=[json.loads(x) for x in (OUT/'INPUT_PROMPTS.jsonl').read_text().splitlines()]
 assert len(rows)==192
 for row in rows:
  task=row['task'];assert set(task)=={'train','test'} and len(task['test'])==1 and set(task['test'][0])=={'input'}
  assert all(set(pair)=={'input','output'} for pair in task['train'])
  assert m.native_prompt(task).endswith('<|im_start|>assistant\n')
def test_reconstruct_v3_audits_from_frozen_prompt_and_sidecar_bytes():
 prompts=[json.loads(x) for x in (OUT/'INPUT_PROMPTS.jsonl').read_text().splitlines()];targets={x['episode_id']:x['target'] for x in (json.loads(x) for x in (OUT/'SEALED_TARGET_SIDECAR.jsonl').read_text().splitlines())}
 rows=[{'episode_id':x['episode_id'],'cell':x['cell'],'canonical_unit':x['canonical_unit'],'turn':x['turn'],'task':x['task'],'target':targets[x['episode_id']]} for x in prompts]
 rotation,one,collision,blind=m.audits(rows)
 assert rotation['status']==one['status']==collision['status']==blind['status']=='PASS'
 assert rotation['d4_classes']==8 and collision['all_query_historical_primary_exact_intersection']==0 and collision['replacement_historical_primary_d4_intersection']==0
 assert collision['retained_unaffected_units_historical_d4_intersection_descriptive'] >= 0
 assert collision['historical_tasks']==12672 and collision['replacement_units']==[0,5]
def test_only_authorized_units_changed_and_replacements_are_asymmetric_connected():
 s=importlib.util.spec_from_file_location('v2',ROOT/'scripts/freeze_atomic_ood_identifiability_pilot_v2.py');v2=importlib.util.module_from_spec(s);assert s.loader;s.loader.exec_module(v2)
 assert [i for i,(before,after) in enumerate(zip(v2.SHAPES,m.SHAPES)) if before!=after]==[0,5]
 for shape in m.SHAPES:
  assert len(shape)==6 and m.connected(shape) and len({m.turn_shape(shape,t) for t in range(4)})==4 and len(m.d4_orbit(shape))==8
def test_frozen_v3_refuses_overwrite():
 p=subprocess.run([sys.executable,str(ROOT/'scripts/freeze_atomic_ood_identifiability_pilot_v3.py')],capture_output=True,text=True)
 assert p.returncode and 'REFUSE_OVERWRITE_FROZEN_ATOMIC_OOD_V3' in p.stderr
