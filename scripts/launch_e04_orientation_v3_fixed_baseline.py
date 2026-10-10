"""Launch exactly one E04 V3 detached job after an immutable binding is published.

The binding is deliberately stored outside the exact source checkout to avoid a
self-hash cycle: it binds an earlier immutable code commit and is materialized
from a later Git object into a nonce-specific external lock directory.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "infra/arc2-dual-agent-runpod-orchestrator-v1"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def remote_script(binding: dict, *, binding_commit: str, binding_repo_path: str, binding_sha256: str) -> str:
    out = binding["output_root"]
    source = binding["source_commit"]
    lock = out + ".launch_lock"
    return f"""set -euo pipefail
repo=/root/arc-runtime-3090-gpu-benchmark-v1/arc2
cd "$repo"
git fetch origin {BRANCH}
git checkout --detach {source}
test "$(git rev-parse HEAD)" = "{source}"
test -z "$(git status --porcelain)"
test ! -e "{out}"
test ! -e "{lock}"
if pgrep -af 'run_e04_orientation_v3_fixed_baseline|arc2_hard_cap_launcher.*e04_orientation' >/tmp/e04-active; then
  cat /tmp/e04-active
  exit 41
fi
mkdir -p "$(dirname "{out}")"
mkdir "{lock}"
git show "{binding_commit}:{binding_repo_path}" > "{lock}/LAUNCH_BINDING.json.tmp"
test "$(sha256sum "{lock}/LAUNCH_BINDING.json.tmp" | awk '{{print $1}}')" = "{binding_sha256}"
mv "{lock}/LAUNCH_BINDING.json.tmp" "{lock}/LAUNCH_BINDING.json"
nohup python3 scripts/arc2_hard_cap_launcher.py --cap-seconds 1800 --receipt "{out}/LAUNCH_CAP_RECEIPT.json" -- python3 scripts/run_e04_orientation_v3_fixed_baseline.py --config experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_NATIVE_GREEDY_RUNTIME_CONFIG_V1.json --binding "{lock}/LAUNCH_BINDING.json" --output-root "{out}" > "{lock}/launcher.log" 2>&1 < /dev/null &
pid=$!
printf '{{"protocol_id":"%s","source_commit":"%s","binding_sha256":"%s","remote_pid":%s,"output_root":"%s","nonce":"%s"}}\\n' "E04_ORIENTATION_V3_FIXED_INDEPENDENT_DEMONSTRATION_BASELINE" "{source}" "{binding_sha256}" "$pid" "{out}" "{binding["nonce"]}" > "{lock}/LAUNCH_RECEIPT.json"
cat "{lock}/LAUNCH_RECEIPT.json"
"""

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-commit", required=True)
    parser.add_argument("--ssh-target")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text(encoding="utf-8-sig"))
    relative = args.binding.resolve().relative_to(ROOT).as_posix()
    command = remote_script(binding, binding_commit=args.binding_commit, binding_repo_path=relative, binding_sha256=sha(args.binding))
    if not args.launch:
        print(command)
        return
    if not args.ssh_target:
        raise SystemExit("SSH_TARGET_REQUIRED_FOR_LAUNCH")
    completed = subprocess.run(["ssh", "-F", "NUL", "-tt", args.ssh_target], input=command + "\nexit\n",
                               text=True, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)

if __name__ == "__main__":
    main()
