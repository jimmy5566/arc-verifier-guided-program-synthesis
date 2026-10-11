import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v1'

def test_frozen_artifacts_are_identifiable_and_target_separated():
    assert OUT.exists()
    m=json.loads((OUT/'MANIFEST.json').read_text())
    assert m['status']=='CPU_ONLY_FROZEN_NO_MODEL_NO_INFERENCE' and m['episodes']==192
    assert m['target_sidecar_sealed'] and not m['gpu_used'] and not m['model_loaded']
    rows=[json.loads(x) for x in (OUT/'INPUT_PROMPTS.jsonl').read_text().splitlines()]
    targets=[json.loads(x) for x in (OUT/'SEALED_TARGET_SIDECAR.jsonl').read_text().splitlines()]
    assert len(rows)==len(targets)==192
    assert len({x['episode_id'] for x in rows})==192
    assert all('target' not in x['prompt'] for x in rows)
    assert json.loads((OUT/'ROTATION_DISTINCTNESS_AUDIT.json').read_text())['status']=='PASS'
    assert json.loads((OUT/'ONE_FACTOR_AUDIT.json').read_text())['status']=='PASS'
    assert json.loads((OUT/'COLLISION_AUDIT.json').read_text())['status']=='PASS'
    blind=json.loads((OUT/'BLIND_BASELINE_AUDIT.json').read_text())
    assert blind['marker_blind']['can_solve_all_counterfactuals'] is False
    assert blind['color_blind']['can_solve_all_counterfactuals'] is False

def test_refuses_overwrite_of_frozen_cohort():
    p=subprocess.run([sys.executable,str(ROOT/'scripts/freeze_atomic_ood_identifiability_pilot_v1.py')],capture_output=True,text=True)
    assert p.returncode != 0 and 'REFUSE_OVERWRITE_FROZEN_ATOMIC_OOD_PILOT' in p.stderr
