from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TABLE={'treatment_control_count_min':3,'treatment_control_count_lower95_gt':0,'treatment_v7_count_min':2,'treatment_v7_count_lower95_gt':0,'object_extraction_no_harm_both':True,'composition_no_harm_both':True,'retention_counts':{'FIXED_TURN_ROTATION_CONTROL':8,'MARKER_BINDING_CONTROL':8,'NO_TRANSFORM_RETENTION':24},'bootstrap_replicates':10000,'all_gates_must_pass':True}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--validate-only',action='store_true');x=a.parse_args();b=json.loads(x.binding.read_text());p=ROOT/b['inputs']['decision_table']['path'];
 if not p.is_file() or sha(p)!=b['inputs']['decision_table']['sha256']:raise SystemExit('IC_V2_SCORER:DECISION_TABLE_IDENTITY')
 t=json.loads(p.read_text())
 if t!=TABLE:raise SystemExit('IC_V2_SCORER:DECISION_TABLE_CONTENT')
 if not x.validate_only:raise SystemExit('IC_V2_SCORER:SCORING_REQUIRES_RAW_FREEZE_AND_SEALED_TRANSITION')
 print(json.dumps({'status':'PASS_V2_SCORER_DECISION_TABLE_CLOSURE','all_gates_must_pass':True,'sealed_sidecar_opened':False,'model_loaded':False,'gpu_used':False},sort_keys=True))
if __name__=='__main__':main()