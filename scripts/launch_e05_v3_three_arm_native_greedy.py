"""Render/launch the one authorized E05 V3 detached job through external binding."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; BRANCH='infra/arc2-dual-agent-runpod-orchestrator-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def render(binding:dict,binding_commit:str,binding_path:str,binding_sha:str):
 out=binding['output_root']; lock=out+'.launch_lock'; src=binding['source_commit']
 return f'''set -euo pipefail
repo=/root/arc-runtime-3090-gpu-benchmark-v1/arc2
cd "$repo"
git fetch origin {BRANCH}
git checkout --detach {src}
test "$(git rev-parse HEAD)" = "{src}"
test -z "$(git status --porcelain)"
test ! -e "{out}"
test ! -e "{lock}"
! pgrep -af 'run_e05_v3_three_arm_native_greedy|arc2_hard_cap_launcher.*e05_v3' > /tmp/e05-active
mkdir -p "$(dirname "{out}")"
mkdir "{lock}"
git show "{binding_commit}:{binding_path}" > "{lock}/LAUNCH_BINDING.json.tmp"
test "$(sha256sum "{lock}/LAUNCH_BINDING.json.tmp" | awk '{{print $1}}')" = "{binding_sha}"
mv "{lock}/LAUNCH_BINDING.json.tmp" "{lock}/LAUNCH_BINDING.json"
nohup python3 scripts/arc2_hard_cap_launcher.py --cap-seconds 10800 --receipt "{out}/LAUNCH_CAP_RECEIPT.json" -- python3 scripts/run_e05_v3_three_arm_native_greedy.py --config experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/E05_V3_RUNTIME_CONFIG.json --binding "{lock}/LAUNCH_BINDING.json" --output-root "{out}" > "{lock}/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '{{"protocol_id":"%s","source_commit":"%s","binding_sha256":"%s","remote_pid":%s,"output_root":"%s","nonce":"%s"}}\\n' "{binding['protocol_id']}" "{src}" "{binding_sha}" "$pid" "{out}" "{binding['nonce']}" > "{lock}/LAUNCH_RECEIPT.json"
cat "{lock}/LAUNCH_RECEIPT.json"
'''
def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--binding-commit',required=True);a.add_argument('--launch',action='store_true');a.add_argument('--ssh-target');x=a.parse_args(); b=json.loads(x.binding.read_text())
 rel=x.binding.resolve().relative_to(ROOT).as_posix();script=render(b,x.binding_commit,rel,sha(x.binding))
 if not x.launch:print(script);return
 if not x.ssh_target:raise SystemExit('SSH_TARGET_REQUIRED')
 r=subprocess.run(['ssh','-F','NUL','-tt',x.ssh_target],input=script+'\nexit\n',text=True);raise SystemExit(r.returncode)
if __name__=='__main__':main()
