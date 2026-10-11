import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.launch_e05_v3_three_arm_native_greedy import render,sha
ROOT=Path(__file__).resolve().parents[1]; B=ROOT/'experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/E05_V3_RUNTIME_LAUNCH_BINDING_RUN_001.json'
def main():
 s=render(json.loads(B.read_text()),'60f375a9b1bb9acbd0afca395ad33249bc9475db',B.relative_to(ROOT).as_posix(),sha(B))
 assert 'git checkout --detach 0fbd20d8613f035ddaa18a9544ab3c781975e22d' in s and '--cap-seconds 10800' in s and 'run_e05_v3_three_arm_native_greedy.py' in s and 'pgrep -af' in s and '[?2004' not in s
 print('PASS_E05_V3_LAUNCHER_RENDER')
if __name__=='__main__':main()
