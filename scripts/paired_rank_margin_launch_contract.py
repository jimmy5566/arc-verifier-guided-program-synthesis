"""Filesystem-only launch/receipt guards for the future paired no-update job."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path

CAP_SECONDS=900
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path:Path,value:dict)->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf8',newline='\n');os.replace(tmp,path)
def require_launch(*,output:Path,receipt:Path,cap_seconds:int,expected_hashes:dict[str,str],actual_paths:dict[str,Path])->None:
 if cap_seconds!=CAP_SECONDS:raise RuntimeError('RUNTIME_CAP_DRIFT')
 if output.exists() or receipt.exists() or output.parent.exists():raise RuntimeError('FRESH_OUTPUT_OR_DUPLICATE_JOB_REQUIRED')
 if set(expected_hashes)!=set(actual_paths) or any(sha(actual_paths[k])!=v for k,v in expected_hashes.items()):raise RuntimeError('LAUNCH_IDENTITY_MISMATCH')
def failure_receipt(*,receipt:Path,reason:str)->None:
 if receipt.exists():raise RuntimeError('RECEIPT_NON_OVERWRITE_REQUIRED')
 atomic(receipt,{'status':'FAILED_NO_UPDATE','reason':reason,'optimizer_steps':0,'generation':False,'training':False,'backward':False})

def require_checkpoint_records(discovery:dict, required:tuple[str,str])->dict:
 records=discovery.get('checkpoint_records',{})
 chosen={key:records.get(key) for key in required}
 if any(not isinstance(v,dict) or not v.get('manifest_path') or not v.get('manifest_sha256') or not v.get('adapter_model_sha256') for v in chosen.values()):raise RuntimeError('CHECKPOINT_DISCOVERY_BINDING_INVALID')
 return chosen
