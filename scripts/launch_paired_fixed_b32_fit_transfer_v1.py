"""Ordered launch chain; intentionally requires an externally frozen authorization."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--authorization',type=Path,required=True);ap.add_argument('--train',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError('FRESH_OUTPUT_REQUIRED')
 cfg=json.loads(a.config.read_text());auth=json.loads(a.authorization.read_text())
 if auth.get('decision')!='CONTINUE_CONTROLLER' or auth.get('bindings',{}).get('config_sha256')!=sha(a.config):raise RuntimeError('AUTHORIZATION_BINDING_FAIL')
 manifests=[Path(v['checkpoint_manifest_path']) for v in cfg['inputs']['conditions'].values()]
 verify=[sys.executable,str(ROOT/'scripts/verify_paired_fixed_b32_runtime_identity.py')]
 for m in manifests:verify += ['--checkpoint-manifest',str(m)]
 subprocess.run(verify,check=True)
 receipt=a.out/'LAUNCH_CAP_RECEIPT.json'
 command=[sys.executable,str(ROOT/'scripts/run_paired_fixed_b32_fit_transfer_v1.py'),'--config',str(a.config),'--authorization',str(a.authorization),'--train',str(a.train),'--out',str(a.out)]
 subprocess.run([sys.executable,str(ROOT/'scripts/arc2_hard_cap_launcher.py'),'--cap-seconds','1800','--receipt',str(receipt),'--',*command],check=True)
 terminal=json.loads((a.out/'TERMINAL_RECEIPT.json').read_text())
 if terminal.get('status')!='COMPLETE_NO_UPDATE':raise RuntimeError('INCOMPLETE_NOT_INTERPRETABLE')
 subprocess.run([sys.executable,str(ROOT/'scripts/analyze_paired_fixed_b32_fit_transfer_v1.py'),'--raw',str(a.out/'RAW_PER_ROW_NLL.jsonl'),'--dev',str(ROOT/cfg['inputs']['frozen_dev_family_evidence_path']),'--out',str(a.out/'ANALYSIS.json')],check=True)
if __name__=='__main__':main()
