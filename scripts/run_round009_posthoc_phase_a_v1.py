"""Phase A target-blind generator.  Real execution is directive-gated."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(t,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--surface',choices=('NOVEL','REPLAY','PROTECTED'),required=True);p.add_argument('--input-manifest',type=Path,required=True);p.add_argument('--prompt-input-jsonl',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--dummy-worker',action='store_true');a=p.parse_args()
 b=json.loads(a.binding.read_text(encoding='utf-8-sig'));m=json.loads(a.input_manifest.read_text(encoding='utf-8-sig'))
 if b.get('phase')!='A_GENERATION' or m.get('contains_targets') is not False:raise RuntimeError('PHASE_A_INPUT_CONTRACT_INVALID')
 if a.output.exists() and any(a.output.iterdir()):raise RuntimeError('PHASE_A_OUTPUT_ROOT_NOT_FRESH')
 if not a.execute:raise RuntimeError('PHASE_A_EXECUTE_REQUIRED')
 rows=[json.loads(x) for x in a.prompt_input_jsonl.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
 expected={x['sample_id'] for x in m['items']}; got={x['sample_id'] for x in rows}
 if got!=expected or any('labels' in x or 'target' in x or 'gold' in x for x in rows):raise RuntimeError('PHASE_A_PROMPT_PROJECTION_INVALID')
 started=time.monotonic(); predictions=[]
 if a.dummy_worker:
  generated={x['sample_id']:[0,15] for x in rows}
 else:
  if not b.get('execution_authorized'):raise RuntimeError('PHASE_A_NOT_AUTHORIZED')
  import torch
  from peft import PeftModel
  from transformers import AutoModelForCausalLM
  model=AutoModelForCausalLM.from_pretrained(b['model_path'],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa');model=PeftModel.from_pretrained(model,b['checkpoint']['path'],is_trainable=False);model.eval();generated={}
  for x in rows:
   ids=torch.tensor([x['prompt_ids']],device='cuda:0');out=model.generate(input_ids=ids,max_new_tokens=int(b['decode']['max_new_tokens']),do_sample=False,num_beams=1,eos_token_id=15,pad_token_id=13);generated[x['sample_id']]=[int(v) for v in out[0,len(x['prompt_ids']):].detach().cpu().tolist()]
 for x in rows:
  tokens=generated[x['sample_id']];predictions.append({'surface':a.surface,'sample_id':x['sample_id'],'prompt_sha256':x['prompt_sha256'],'generated_token_ids':tokens,'prediction_sha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),'parse_status':'UNSCORED_SEALED_PHASE_B'})
 manifest={'schema_version':1,'phase':'A_GENERATION','status':'SUCCESS','prediction_count':len(predictions),'predictions':predictions,'contains_references':False,'model_loaded':not a.dummy_worker,'generation_started':True}
 write(a.output/'PHASE_A_PREDICTION_MANIFEST.json',manifest);mh=sha(a.output/'PHASE_A_PREDICTION_MANIFEST.json');receipt={'schema_version':1,'phase':'A_GENERATION','status':'SUCCESS','nonce':b['nonce'],'prediction_manifest_sha256':mh,'prediction_count':len(predictions),'wall_seconds':time.monotonic()-started,'model_loaded':not a.dummy_worker,'generation_started':True,'final_audit_opened':False};write(a.receipt,receipt);print(json.dumps(receipt,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
