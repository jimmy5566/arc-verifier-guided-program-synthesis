"""External wall-clock watchdog for a bounded ARC2 subprocess."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');os.replace(tmp,path)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cap-seconds',type=int,required=True);ap.add_argument('--receipt',type=Path,required=True);ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    if not a.command or a.command[0]!='--':raise SystemExit('COMMAND_AFTER_DOUBLE_DASH_REQUIRED')
    command=a.command[1:];started=time.monotonic();p=subprocess.Popen(command)
    status='COMPLETE';code=None
    while code is None:
        code=p.poll()
        if code is None and time.monotonic()-started>a.cap_seconds:
            p.terminate();
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            status='RUNTIME_CAP_REACHED';code=p.returncode
        elif code is None:time.sleep(.25)
    atomic(a.receipt,{'schema_version':1,'status':status,'cap_seconds':a.cap_seconds,'wall_seconds':time.monotonic()-started,'exit_code':code,'command':command})
    if status!='COMPLETE' or code!=0:raise SystemExit(1)
if __name__=='__main__':main()
