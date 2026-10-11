"""GitHub-synchronized detached launcher for the one E04-E V2 parent job."""
from __future__ import annotations
import argparse,base64,hashlib,json,secrets,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BRANCH='infra/arc2-dual-agent-runpod-orchestrator-v1'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def forced_pty_shell(target:str,script:str)->str:
 """Use the established RunPod command-control envelope; never transfer assets."""
 nonce=secrets.token_hex(16); completion=json.dumps({'arc2_remote_nonce':nonce},sort_keys=True)
 encoded=base64.b64encode((script+f"\nprintf '%s\\n' '{completion}'\n").encode()).decode()
 line=f"echo {encoded} | base64 -d | bash; printf '__ARC2_REMOTE_END__\\n'; exit"
 payload=f"\x1b[200~{line}\x1b[201~\r".encode()
 completed=subprocess.run(['ssh','-F','NUL','-tt','-o','BatchMode=yes','-o','ConnectTimeout=20',target],input=payload,capture_output=True,check=False,timeout=45)
 lines=completed.stdout.splitlines();marker=b'__ARC2_REMOTE_END__';done=completion.encode()
 if completed.returncode or marker not in [x.strip() for x in lines] or done not in lines:raise SystemExit(f'E04E_REMOTE_COMMAND_CONTROL_FAILED:{completed.returncode}')
 return completed.stdout.decode('utf-8',errors='replace')
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
gpu_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
test "$gpu_total" -ge 24000
test "$gpu_free" -ge 20000
python3 -c 'import torch, transformers, peft, bitsandbytes; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()'
mkdir -p "$(dirname "{out}")";mkdir "{lock}"
git show "{binding_commit}:{binding_path}" > "{lock}/LAUNCH_BINDING.json.tmp"
test "$(sha256sum "{lock}/LAUNCH_BINDING.json.tmp" | awk '{{print $1}}')" = "{binding_sha}"
mv "{lock}/LAUNCH_BINDING.json.tmp" "{lock}/LAUNCH_BINDING.json"
nohup python3 scripts/arc2_hard_cap_launcher.py --cap-seconds 7200 --receipt "{out}/LAUNCH_CAP_RECEIPT.json" -- python3 scripts/run_e04_e_equal_slot_marker_replay_pilot.py --binding "{lock}/LAUNCH_BINDING.json" --output-root "{out}" --launch-commit "{binding_commit}" > "{lock}/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '{{"protocol_id":"%s","launch_commit":"%s","worker_source_commit":"%s","binding_sha256":"%s","remote_pid":%s,"output_root":"%s","nonce":"%s","expected_terminal_receipt":"%s/TERMINAL_RECEIPT.json","gpu_total_mib":%s,"gpu_free_mib":%s,"gh_token_present":true}}\\n' "{b['protocol_id']}" "{source}" "{b['worker_source_commit']}" "{binding_sha}" "$pid" "{out}" "{b['nonce']}" "{out}" "$gpu_total" "$gpu_free" > "{lock}/LAUNCH_RECEIPT.json"
cat "{lock}/LAUNCH_RECEIPT.json"'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-commit',required=True);p.add_argument('--ssh-target');p.add_argument('--launch',action='store_true');a=p.parse_args();b=json.loads(a.binding.read_text());rel=a.binding.resolve().relative_to(ROOT).as_posix();cmd=remote_script(b,a.binding_commit,rel,sha(a.binding))
 if not a.launch:print(cmd);return
 if not a.ssh_target:raise SystemExit('SSH_TARGET_REQUIRED_FOR_COMMAND_CONTROL')
 print(forced_pty_shell(a.ssh_target,cmd))
if __name__=='__main__':main()
