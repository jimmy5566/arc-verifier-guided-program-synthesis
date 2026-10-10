from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.e04_orientation_marker_counterfactual_v2 import freeze, deterministic_regeneration_report, validate_frozen
if __name__ == '__main__':
    manifest = freeze()
    regen = deterministic_regeneration_report()
    validation = validate_frozen()
    report={'status':'CPU_FREEZE_VALIDATION_PASS','manifest_sha256':__import__('hashlib').sha256((Path(__file__).resolve().parents[1]/'experiments/capability_repair_baseline_v1/e04_orientation_marker_counterfactual_v2/MANIFEST.json').read_bytes()).hexdigest(),'validation':validation,'regeneration':regen,'model_loaded':False,'gpu_used':False,'optimizer_steps':0}
    (Path(__file__).resolve().parents[1]/'experiments/capability_repair_baseline_v1/e04_orientation_marker_counterfactual_v2/CPU_FREEZE_VALIDATION_V1.json').write_bytes(json.dumps(report,sort_keys=True,indent=2).encode('utf-8')+b'\\n')
    print(json.dumps({'manifest':manifest,'report':report},sort_keys=True))
