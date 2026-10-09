import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));WEAK=('connected components','inside/contains','difference','width','orientation');IGN=-100;PAD=13
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n');os.replace(t,p)
def rows(path,manifest):
 from training_data.pipeline import task_to_sample
 d={x['episode_id']:x for x in map(json.loads,filter(str,path.read_text().splitlines()))};o=[]
 for b in manifest['batches']:
  g=[]
  for m in b['members']:
   r=d[m['episode_id']]
   if r['split']!='TARGET_DEV' or r['family'] not in WEAK:raise RuntimeError('DEV_BINDING_FAIL')
   s=task_to_sample({'source_id':r['episode_id'],**r['task']});g.append({'episode_id':r['episode_id'],'family':r['family'],'ids':s['input_ids'],'labels':s['labels']})
  o.append(g)
 return o
def selftest(cfg,dev):
 m=json.loads((ROOT/cfg['inputs']['manifest_path']).read_text());g=rows(dev,m);flat=[x for b in g for x in b];c={f:sum(x['family']==f for x in flat) for f in WEAK}
 if c!={f:12 for f in WEAK}:raise RuntimeError('FAMILY_DENOMINATOR_FAIL')
 return {'status':'PASS_CPU_ONLY','rows':len(flat),'batches':len(g),'families':c}
def collate(g,torch):
 w=max(map(lambda x:len(x['ids']),g));ids=[];ma=[];la=[]
 for x in g:
  n=w-len(x['ids']);ids.append(x['ids']+[PAD]*n);ma.append([1]*len(x['ids'])+[0]*n);la.append(x['labels']+[IGN]*n)
 return torch.tensor(ids,device='cuda:0'),torch.tensor(ma,device='cuda:0'),torch.tensor(la,device='cuda:0')
def score(model,groups,torch,F,ck,mode,deadline):
 out=[]
 with torch.no_grad():
  for bi,g in enumerate(groups):
   if time.monotonic()>deadline:raise RuntimeError('RUNTIME_CAP_REACHED')
   ids,ma,la=collate(g,torch);z=model(input_ids=ids,attention_mask=ma,use_cache=False).logits.float();loss=F.cross_entropy(z[:,:-1,:].reshape(-1,z.shape[-1]),la[:,1:].reshape(-1),ignore_index=IGN,reduction='none').view_as(la[:,1:])
   for i,x in enumerate(g):
    v=loss[i][la[i,1:]!=IGN].tolist();out.append({'checkpoint_id':ck,'mode_id':mode,'episode_id':x['episode_id'],'family':x['family'],'row_nll_sum':float(sum(v)),'supervised_token_count':len(v)})
 return out
def main():
 a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True);a.add_argument('--dev',type=Path,required=True);a.add_argument('--out',type=Path);a.add_argument('--self-test',action='store_true');q=a.parse_args();cfg=json.loads(q.config.read_text())
 if q.self_test:print(json.dumps(selftest(cfg,q.dev),sort_keys=True));return
 if q.out is None or q.out.exists():raise RuntimeError('FRESH_OUTPUT_REQUIRED')
 import torch,torch.nn.functional as F
 from transformers import AutoModelForCausalLM
 from peft import PeftModel
 manifest=json.loads((ROOT/cfg['inputs']['manifest_path']).read_text());gs=rows(q.dev,manifest);by_id={x['episode_id']:x for g in gs for x in g};cross=[by_id[x] for x in manifest['batch1_crosscheck']];raw=[];deadline=time.monotonic()+900
 for ck,c in cfg['inputs']['conditions'].items():
  m=json.loads((ROOT/c['manifest']).read_text());model=AutoModelForCausalLM.from_pretrained(m['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0');model=PeftModel.from_pretrained(model,m['adapter_path'],is_trainable=False);model.eval();raw+=score(model,gs,torch,F,ck,'PRIMARY_B32',deadline)+score(model,[[x] for x in cross],torch,F,ck,'CROSSCHECK_B1',deadline);del model;torch.cuda.empty_cache()
 q.out.mkdir(parents=True);rp=q.out/'RAW.jsonl';rp.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in raw));atomic(q.out/'TERMINAL_RECEIPT.json',{'status':'COMPLETE_NO_UPDATE','raw_sha256':sha(rp),'raw_rows':len(raw),'optimizer_steps':0,'generation_performed':False,'final_audit_opened':False})
if __name__=='__main__':main()
