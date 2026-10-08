"""Thin persistent ARC2 workflow runner.

Only schedules the four high-level workflow states.  Controller owns science;
Director is called synchronously by Controller; Supervisor remains a receipt
utility for detached RunPod jobs.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATES={"ACTIVE","WAITING_REMOTE_JOB","PAUSED","TERMINAL"}
TERMINAL_RECEIPTS={"SUCCESS","TRAIN_FAILED","OOM","INFRA_FAILED","INTERRUPTED"}
def now()->str:return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def save(p:Path,v:dict[str,Any])->None:
 t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n");os.replace(t,p)
def load(p:Path)->dict[str,Any]:
 v=json.loads(p.read_text(encoding="utf-8"))
 if v.get("schema_version")!=1 or v.get("status") not in STATES:raise RuntimeError("INVALID_ARC2_WORKFLOW_STATE")
 return v
def agents()->list[dict[str,Any]]:
 r=subprocess.run(["herdr","agent","list"],capture_output=True,text=True,encoding="utf-8",errors="strict",timeout=15)
 if r.returncode:raise RuntimeError("HERDR_AGENT_LIST_FAILED")
 return json.loads(r.stdout)["result"]["agents"]
def controller_target(requested:str)->tuple[str,str]:
 a=agents()
 for x in a:
  if x.get("agent")==requested or x.get("name")==requested:return str(x["agent"]),str(x.get("agent_status"))
 focused=[x for x in a if x.get("agent")=="codex" and x.get("focused")]
 if len(focused)==1:return "codex",str(focused[0].get("agent_status"))
 raise RuntimeError("CONTROLLER_AGENT_UNRESOLVED")
def prompt_controller(target:str,state:Path,timeout:int)->None:
 text=(f"ARC2 workflow runner: read {state.resolve()} and resume the current high-level stage. "
 "Execute as much scientifically valid work as possible, persist the next high-level state before returning, "
 "and do not start GPU/model work without the workflow's valid authorization.")
 cmd=["herdr","agent","prompt",target,text,"--wait","--until","idle","--until","done","--until","blocked","--timeout",str(timeout*1000)]
 r=subprocess.run(cmd,capture_output=True,text=True,encoding="utf-8",errors="strict",timeout=timeout+15)
 if r.returncode:raise RuntimeError("CONTROLLER_TURN_FAILED")
def remote_terminal(state:dict[str,Any],receipt_root:Path)->bool:
 job=state.get("active_remote_job") or {}; path=job.get("terminal_receipt_path") if isinstance(job,dict) else None
 candidates=[Path(path)] if path else list(receipt_root.glob("ROUND_*/*_TERMINAL_RECEIPT.json"))
 for p in candidates:
  if p.is_file():
   try:r=json.loads(p.read_text(encoding="utf-8"))
   except json.JSONDecodeError:continue
   if r.get("status") in TERMINAL_RECEIPTS:
    state.update({"status":"ACTIVE","active_remote_job":None,"last_completed_action":"REMOTE_JOB_TERMINAL_RECEIPT_AVAILABLE","next_action":"PROCESS_REMOTE_RECEIPT","last_updated_at":now()});return True
 return False
def step(state_path:Path,controller:str,receipt_root:Path,timeout:int)->str:
 state=load(state_path); status=state["status"]
 if status in {"PAUSED","TERMINAL"}:return status
 if status=="WAITING_REMOTE_JOB":
  if remote_terminal(state,receipt_root):save(state_path,state);return "REMOTE_TERMINAL"
  return "REMOTE_PENDING"
 target,agent_status=controller_target(controller)
 if agent_status=="working":return "CONTROLLER_BUSY"
 prompt_controller(target,state_path,timeout);return "CONTROLLER_TURN"
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--state",type=Path,required=True);p.add_argument("--receipt-root",type=Path,required=True);p.add_argument("--controller-agent",default="arc-controller");p.add_argument("--turn-timeout-seconds",type=int,default=180);p.add_argument("--remote-poll-seconds",type=float,default=300);p.add_argument("--once",action="store_true");a=p.parse_args()
 while True:
  result=step(a.state,a.controller_agent,a.receipt_root,a.turn_timeout_seconds)
  if a.once or result in {"PAUSED","TERMINAL"}:return 0
  time.sleep(a.remote_poll_seconds if result=="REMOTE_PENDING" else 1)
if __name__=="__main__":
 try:raise SystemExit(main())
 except RuntimeError as e:print(str(e),file=sys.stderr);raise SystemExit(2)
