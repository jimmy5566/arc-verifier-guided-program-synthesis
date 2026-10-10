from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'experiments/capability_repair_baseline_v1/e04_orientation_frozen_cohort_v1/E04_ORIENTATION_V7_ONLY_NO_UPDATE_EVALUATION_PRELAUNCH_PACKAGE_V1.json'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    package=json.loads(P.read_text(encoding='utf-8'))
    assert package['status']=='CPU_ONLY_PRELAUNCH_PACKAGE_REVIEW_REQUIRED'
    assert package['checkpoint']['identity']=='RECONSTRUCTED_FOUNDATION_V2_V7'
    assert package['cohort']['sha256']=='76c6fd57173ec62571f27a312e3f80601c70cdec04cda79432822937e7a37967'
    assert package['cohort']['validation_base_tuples']==48
    assert package['cohort']['validation_condition_views']==192
    assert package['estimands']['uncertainty']=={'method':'reserved-cell-stratified bootstrap at the independent base-tuple level with paired views retained within resamples','seed':20261010,'replicates':10000,'insufficient_conditional_count':'INCONCLUSIVE'}
    proposal=package['runtime_proposal_for_review']
    for key in ('optimizer_steps','parameter_updates','backward','training','gradient_computation','Gold_access','dGold_access','FINAL_AUDIT_access'):
        assert proposal[key] in (0,False)
    assert proposal['authorization'].startswith('No model loading')
    for entry in package['bound_files']:
        path=ROOT/entry['path']; assert path.is_file() and sha(path)==entry['sha256'],entry['path']
    print('E04_V7_NO_UPDATE_PRELAUNCH_PACKAGE_CPU_TEST_PASS')
if __name__=='__main__':main()
