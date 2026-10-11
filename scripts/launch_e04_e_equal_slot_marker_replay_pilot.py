"""GitHub-synchronized detached launcher for the one E04-E V2 parent job."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BRANCH='infra/arc2-dual-agent-runpod-orchestrator-v1'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def remote_script(b:dict,binding_commit:str,binding_path:str,binding_sha:str)->str:
 out=b['output_root'];lock=out+'.launch_lock';source=binding_commit
 return f'''set -euo pipefail
repo=/root/arc-runtime-3090-gpu-benchmark-v1/arc2
cd "$repo"
test "${{GH_TOKEN:+present}}" = "present"
git fetch origin {BRANCH}
git checkout --detach {source}
test "$(git rev-parse HEAD)" = "{source}"
test -z "$(git status --porcelain)"
test ! -e "{out}"
test ! -e "{lock}"
if pgrep -af 'run_e04_e_equal_slot_marker_replay_pilot|arc2_hard_cap_launcher.*e04_e' >/tmp/e04e-active; then cat /tmp/e04e-active; exit 41; fi
mkdir -p "$(dirname "{out}")";mkdir "{lock}"
git show "{binding_commit}:{binding_path}" > "{lock}/LAUNCH_BINDING.json.tmp"
test "$(sha256sum "{lock}/LAUNCH_BINDING.json.tmp" | awk '{{print $1}}')" = "{binding_sha}"
mv "{lock}/LAUNCH_BINDING.json.tmp" "{lock}/LAUNCH_BINDING.json"
nohup python3 scripts/arc2_hard_cap_launcher.py --cap-seconds 7200 --receipt "{out}/LAUNCH_CAP_RECEIPT.json" -- python3 scripts/run_e04_e_equal_slot_marker_replay_pilot.py --binding "{lock}/LAUNCH_BINDING.json" --output-root "{out}" --launch-commit "{binding_commit}" > "{lock}/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '{{"protocol_id":"%s","launch_commit":"%s","worker_source_commit":"%s","binding_sha256":"%s","remote_pid":%s,"output_root":"%s","nonce":"%s","expected_terminal_receipt":"%s/TERMINAL_RECEIPT.json","gh_token_present":true}}\\n' "{b['protocol_id']}" "{source}" "{b['worker_source_commit']}" "{binding_sha}" "$pid" "{out}" "{b['nonce']}" "{out}" > "{lock}/LAUNCH_RECEIPT.json"
cat "{lock}/LAUNCH_RECEIPT.json"'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-commit',required=True);p.add_argument('--ssh-target');p.add_argument('--launch',action='store_true');a=p.parse_args();b=json.loads(a.binding.read_text());rel=a.binding.resolve().relative_to(ROOT).as_posix();cmd=remote_script(b,a.binding_commit,rel,sha(a.binding))
 if not a.launch:print(cmd);return
 if not a.ssh_target:raise SystemExit('SSH_TARGET_REQUIRED_FOR_COMMAND_CONTROL')
 r=subprocess.run(['ssh','-F','NUL','-tt',a.ssh_target],input=cmd+'\nexit\n',text=True,check=False)
 if r.returncode:raise SystemExit(r.returncode)
if __name__=='__main__':main()
