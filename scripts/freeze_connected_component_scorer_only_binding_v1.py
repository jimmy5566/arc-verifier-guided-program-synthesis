#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--predecessor',type=Path,required=True);p.add_argument('--raw-root',required=True);p.add_argument('--nonce',required=True);p.add_argument('--output-root',required=True);p.add_argument('--control-root',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();b=copy.deepcopy(json.loads(a.predecessor.read_text()));e=b['execution'];
 b['status']='FROZEN_SCORER_ONLY_OWNER_AUTH_REQUIRED';b['runtime_source_commit']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();e.update({'nonce':a.nonce,'output_root':a.output_root,'control_root':a.control_root,'free_running_generations':0,'raw_rows_created':0,'immutable_raw_root':a.raw_root,'immutable_raw_sha256':'5b118d593ab605b738e6ac1b95a6437a6d784ab8e8268c0a14a4e6af7311f137','immutable_raw_freeze_receipt_sha256':'b41e91621ef8276518e13567c0d203a6cbc8054d753e9c54e54a514af8c90255'});b['runnable'].update({'scorer_only_worker':'scripts/run_connected_component_scorer_only_v1.py','scorer_only_launcher':'scripts/launch_connected_component_scorer_only_v1.py'});paths=set(b['source_hashes'])|{'scripts/score_connected_component_fit_generation_diagnostic_v1.py','scripts/run_connected_component_scorer_only_v1.py','scripts/launch_connected_component_scorer_only_v1.py','scripts/freeze_connected_component_scorer_only_binding_v1.py'};b['source_hashes']={x:sha(ROOT/x) for x in sorted(paths)};b['authorization']={'owner_one_shot_required':True,'not_authorized_by_this_binding':True,'permitted_launches':1,'retry':False};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(b,sort_keys=True,indent=2)+'\n');print(sha(a.out))
if __name__=='__main__':main()
