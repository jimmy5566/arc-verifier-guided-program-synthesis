from __future__ import annotations
import argparse,json
from pathlib import Path
def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 s='''import os, subprocess, sys
from pathlib import Path
root=next(Path("/kaggle/input").rglob("run_soar_single_gpu_parallel_smoke.py")).parents[1]
native=next(Path("/kaggle/input").rglob("native_frozen30.json")); challenge=next(Path("/kaggle/input").rglob("arc-agi_training_challenges.json")); out=Path("/kaggle/working/artifacts/soar_single_gpu_parallel_smoke")
cmd=[sys.executable,str(root/"scripts/run_soar_single_gpu_parallel_smoke.py"),"--challenge-path",str(challenge),"--native-frozen",str(native),"--input-root","/kaggle/input","--output-root",str(out)]
print({"event":"SOAR_SINGLE_GPU_GATE_START","cmd":cmd,"gpus":subprocess.check_output(["nvidia-smi","-L"],text=True).splitlines()})
if subprocess.run(cmd,text=True).returncode: raise RuntimeError("single GPU gate/smoke failed")
print({"event":"SOAR_SINGLE_GPU_SMOKE_COMPLETE","files":[str(x) for x in out.rglob("*.json")]})
'''
 n={"cells":[{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.splitlines(keepends=True)}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"kaggle":{"accelerator":"nvidiaL4","isGpuEnabled":True,"isInternetEnabled":False,"language":"python","sourceType":"notebook"}},"nbformat":4,"nbformat_minor":4};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(n),encoding='utf-8')
if __name__=='__main__':main()
