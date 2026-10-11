"""CPU postprocessor for the frozen E04-D sufficient statistics."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from scripts import e04_d_gradient_statistics as s

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--raw',type=Path,required=True);a.add_argument('--config',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
 raw=json.loads(x.raw.read_text());c=json.loads(x.config.read_text())
 if raw.get('status')!='COMPLETE_NO_UPDATE' or any(raw.get(k)!=0 for k in ('optimizer_steps','parameter_updates','generation_calls')) or raw.get('final_audit_opened') is not False:raise SystemExit('E04D_PROCESS_BOUNDARY_INVALID')
 if raw.get('config_sha256')!=sha(x.config):raise SystemExit('E04D_CONFIG_IDENTITY_INVALID')
 rows=raw.get('pairs',[]); expected={r['pair_id'] for r in [json.loads(z) for z in Path(c['cohort_path']).read_text().splitlines() if z]}
 if len(rows)!=48 or {r.get('pair_id') for r in rows}!=expected:raise SystemExit('E04D_RAW_COHORT_INVALID')
 repeats=raw.get('repeat_cosines',{});
 if set(repeats)!={'FIXED_TURN_ROTATION_CONTROL','MARKER_BINDING_CONTROL','NO_TRANSFORM_RETENTION_CONTROL'} or not all(-1<=float(v)<=1 for v in repeats.values()):raise SystemExit('E04D_REPEAT_INVALID')
 scales=raw.get('repeat_scale_variation',{});
 if set(scales)!=set(repeats) or any(set(v)!={'first_norm','second_norm','absolute_difference','relative_difference'} or any(float(v[k])<0 for k in v) for v in scales.values()):raise SystemExit('E04D_REPEAT_SCALE_INVALID')
 result=s.classify(rows,repeats);result.update({'status':'COMPLETE_NO_UPDATE','protocol_id':c['protocol_id'],'raw_sha256':sha(x.raw),'config_sha256':sha(x.config),'repeat_scale_variation':scales,'interpretation_limit':'repeat scale variation is reported separately and does not enter cosine sign classification; local alignment is not causal proof of the E04-C marker regression'})
 x.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
if __name__=='__main__':main()
