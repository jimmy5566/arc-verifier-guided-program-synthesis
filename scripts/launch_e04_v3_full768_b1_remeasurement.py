"""Launch exactly one Director-authorized E04 full768 physical-B1 job."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BRANCH='infra/arc2-dual-agent-runpod-orchestrator-v1'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def remote_script(binding:dict,binding_commit:str,binding_path:str,binding_sha:str,ssh_target:str='')->str:
 out,source,lock=binding['output_root'],binding['source_commit'],binding['output_root']+'.launch_lock'
 return f'''set -euo pipefail
repo=/root/arc-runtime-3090-gpu-benchmark-v1/arc2
cd "$repo"
git fetch origin {BRANCH}
git checkout --detach {source}
test "$(git rev-parse HEAD)" = "{source}"
test -z "$(git status --porcelain)"
test ! -e "{out}"
test ! -e "{lock}"
if pgrep -af 'run_e04_v3_full768_b1_remeasurement|arc2_hard_cap_launcher.*full768' >/tmp/e04-full768-active; then cat /tmp/e04-full768-active; exit 41; fi
mkdir -p "$(dirname "{out}")"
mkdir "{lock}"
git show "{binding_commit}:{binding_path}" > "{lock}/LAUNCH_BINDING.json.tmp"
test "$(sha256sum "{lock}/LAUNCH_BINDING.json.tmp" | awk '{{print $1}}')" = "{binding_sha}"
mv "{lock}/LAUNCH_BINDING.json.tmp" "{lock}/LAUNCH_BINDING.json"
python3 scripts/run_e04_v3_full768_b1_remeasurement.py --self-test > "{lock}/PREIMPORT_PREFLIGHT.json"
nohup python3 scripts/arc2_hard_cap_launcher.py --cap-seconds 1800 --receipt "{out}/LAUNCH_CAP_RECEIPT.json" -- python3 scripts/run_e04_v3_full768_b1_remeasurement.py --config experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_FULL768_B1_NO_UPDATE_REMEASUREMENT_V1_CONFIG.json --binding "{lock}/LAUNCH_BINDING.json" --output-root "{out}" > "{lock}/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '{{"protocol_id":"%s","source_commit":"%s","binding_sha256":"%s","remote_pid":%s,"output_root":"%s","nonce":"%s","ssh_target":"%s","expected_terminal_receipt":"%s/TERMINAL_RECEIPT.json","primary_process":{{"host":"RUNPOD","role":"remote_launcher","pid":%s}}}}\\n' "E04_V3_FULL768_B1_NO_UPDATE_REMEASUREMENT_V1" "{source}" "{binding_sha}" "$pid" "{out}" "{binding['nonce']}" "{ssh_target}" "{out}" "$pid" > "{lock}/LAUNCH_RECEIPT.json"
cat "{lock}/LAUNCH_RECEIPT.json"
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-commit',required=True);p.add_argument('--ssh-target');p.add_argument('--launch',action='store_true');a=p.parse_args();b=json.loads(a.binding.read_text(encoding='utf-8'));rel=a.binding.resolve().relative_to(ROOT).as_posix();cmd=remote_script(b,a.binding_commit,rel,sha(a.binding),a.ssh_target or '')
 if not a.launch: print(cmd);return
 if not a.ssh_target:raise SystemExit('SSH_TARGET_REQUIRED')
 raise SystemExit(subprocess.run(['ssh','-F','NUL','-tt',a.ssh_target],input=cmd+'\nexit\n',text=True,check=False).returncode)
if __name__=='__main__':main()
