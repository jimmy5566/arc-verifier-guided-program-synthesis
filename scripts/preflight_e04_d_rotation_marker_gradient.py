"""CPU-only gate for E04-D source, config, cohort, and one-job binding."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--emit',type=Path);x=a.parse_args();b=json.loads(x.binding.read_text());
 if b.get('protocol_id')!='E04_D_ROTATION_MARKER_LOCAL_GRADIENT_DIAGNOSTIC_V1' or b.get('execution_authorized') is not True or b.get('jobs')!=1 or b.get('retry') is not False or b.get('runtime_cap_seconds')!=1800:raise SystemExit('E04D_BINDING_CONTRACT')
 if not str(b.get('nonce','')) or not str(b.get('output_root','')).startswith('/workspace/arc2/e04_d_rotation_marker_local_gradient_diagnostic_v1/'):raise SystemExit('E04D_OUTPUT_BINDING')
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();src=b.get('approved_source_commit','')
 if subprocess.call(['git','merge-base','--is-ancestor',src,head],cwd=ROOT)!=0:raise SystemExit('E04D_APPROVED_SOURCE_NOT_ANCESTOR')
 if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise SystemExit('E04D_TRACKED_SOURCE_DIRTY')
 for rel,digest in b.get('bound_files',{}).items():
  p=(ROOT/rel).resolve()
  if not str(p).startswith(str(ROOT.resolve())) or not p.is_file() or sha(p)!=digest:raise SystemExit('E04D_BOUND_FILE_HASH:'+rel)
 response=ROOT/b['director_response_path']
 if sha(response)!=b['director_response_sha256'] or json.loads(response.read_text()).get('post_repair_execution',{}).get('maximum_detached_jobs')!=1:raise SystemExit('E04D_DIRECTOR_AUTHORIZATION')
 out={'status':'PASS_NO_MODEL_IMPORT','binding_sha256':sha(x.binding),'approved_source_commit':src,'runtime_head':head,'bound_files':len(b['bound_files']),'model_loaded':False,'gpu_used':False,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False}
 if x.emit:x.emit.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
 print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
