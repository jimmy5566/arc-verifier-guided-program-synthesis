"""Protocol-specific one-job worker for the accepted E1 pilot.

`--preflight` imports no model library.  `--execute` is intentionally the only
GPU path and is gated by the successor binding and launcher.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
# Script entrypoints start with scripts/ on sys.path; runtime modules live in src/.
import sys
for _import_root in (ROOT, ROOT/'src'):
 if str(_import_root) not in sys.path:
  sys.path.insert(0,str(_import_root))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(canon(x)+b'\n');os.replace(q,p)
def write_jsonl(p, values):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(b''.join(canon(x)+b'\n' for x in values));os.replace(q,p)
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def fail(x): raise RuntimeError('E1_'+x)

def _serializer_task(item):
 """Return a non-mutating serializer view for one frozen schedule row.

 The protected E04 replay bytes predate ``source_id``.  ``task_to_sample``
 requires that metadata field, but it does not place it in model messages.
 Keep the immutable schedule untouched and derive a stable identifier from the
 already-frozen canonical base identity.
 """
 task=copy.deepcopy(item['task'])
 task.setdefault('source_id',f"SERIALIZER_VIEW:{item['canonical_base_id']}")
 return task

def validate_training_schedules(binding):
 """Exercise the exact production serializer over every slot in both arms."""
 from training_data.pipeline import task_to_sample
 base=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
 report={}
 for arm,name in [('CONTROL_NARROW_STRUCTURE','CONTROL_SCHEDULE.json'),('TREATMENT_STRUCTURE_DIVERSE','TREATMENT_SCHEDULE.json')]:
  schedule=json.loads((base/name).read_text())['episodes']
  if len(schedule)!=384 or [row.get('slot') for row in schedule]!=list(range(384)): fail('SCHEDULE_SLOTS:'+arm)
  samples=[]
  for index,item in enumerate(schedule):
   try: samples.append(task_to_sample(_serializer_task(item)))
   except Exception as exc: raise RuntimeError(f'E1_SCHEDULE_SERIALIZATION:{arm}:{index}:{type(exc).__name__}:{exc}') from exc
  transition=schedule[96]
  if transition.get('role')!='PROTECTED_REPLAY' or 'source_id' in transition['task']: fail('FIRST_REPLAY_TRANSITION:'+arm)
  report[arm]={'slots':len(samples),'first_replay_slot':96,'first_replay_sample_id':samples[96]['sample_id'],'supervised_tokens':sum(x['assistant_token_count'] for x in samples),'transformer_tokens':sum(x['sequence_length'] for x in samples)}
 if report['CONTROL_NARROW_STRUCTURE']['supervised_tokens']!=report['TREATMENT_STRUCTURE_DIVERSE']['supervised_tokens']: fail('SCHEDULE_SUPERVISED_TOKEN_MISMATCH')
 if report['CONTROL_NARROW_STRUCTURE']['transformer_tokens']!=report['TREATMENT_STRUCTURE_DIVERSE']['transformer_tokens']: fail('SCHEDULE_TRANSFORMER_TOKEN_MISMATCH')
 return report

def preflight(binding_path:Path, output:Path|None=None, require_fresh:bool=False):
 b=json.loads(binding_path.read_text())
 required={'protocol_id','scientific_source_commit','inputs','v7','worker_path','worker_sha256','launcher_path','launcher_sha256','score_entrypoint_path','score_entrypoint_sha256','output','runtime','authorization_required','evaluation','dataset_paths','scientific_construct_validation_path','scientific_construct_validation_sha256'}
 if not required.issubset(b): fail('BINDING_SCHEMA')
 if b['worker_path']!='scripts/run_connected_components_object_extraction_pilot_v1.py' or b['worker_sha256']!=sha(Path(__file__)): fail('WORKER_IDENTITY')
 for rel,expected in b['inputs'].items():
  if sha(ROOT/rel)!=expected: fail('INPUT_IDENTITY:'+rel)
 construct=ROOT/b['scientific_construct_validation_path']
 if sha(construct)!=b['scientific_construct_validation_sha256']: fail('SCIENTIFIC_CONSTRUCT_IDENTITY')
 if json.loads(construct.read_text()).get('status')!='PASS_CPU_ONLY_INDEPENDENT_SEMANTIC_ORACLE_VALIDATION': fail('SCIENTIFIC_CONSTRUCT_STATUS')
 if sha(ROOT/b['v7']['checkpoint_manifest_path'])!=b['v7']['checkpoint_manifest_sha256']: fail('V7_MANIFEST')
 validate_training_schedules(b)
 total_primary=0;total_b1=0
 for tag,arm in [('V7',None),('CONTROL_NARROW_STRUCTURE','CONTROL_NARROW_STRUCTURE'),('TREATMENT_STRUCTURE_DIVERSE','TREATMENT_STRUCTURE_DIVERSE')]:
  items=_datasets(b,arm);total_primary+=len(items);total_b1+=len(_select_b1_items(items,b,tag))
 if total_primary!=int(b['evaluation']['primary_evaluations']): fail('PRIMARY_EVALUATION_COUNT')
 if total_b1!=int(b['evaluation']['batch1_repeat_evaluations']): fail('B1_EVALUATION_COUNT')
 if output is not None:
  if require_fresh and output.exists(): fail('FRESH_OUTPUT_REQUIRED')
  if b['output']['remote_output_root']!=str(output).replace('\\','/'): fail('OUTPUT_ROOT')
 return b

def _messages(task):
 from inference.nvarc_native import serialize_grid
 out=[]
 for pair in task['train']: out += [{'role':'user','content':serialize_grid(pair['input'])},{'role':'assistant','content':serialize_grid(pair['output'])}]
 out.append({'role':'user','content':serialize_grid(task['test'][0]['input'])}); return out

def _load_v7(binding, trainable=False):
 import torch
 from peft import PeftModel
 from transformers import AutoModelForCausalLM
 m=json.loads((ROOT/binding['v7']['checkpoint_manifest_path']).read_text());
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): fail('CUDA_BF16')
 base=AutoModelForCausalLM.from_pretrained(m['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0')
 model=PeftModel.from_pretrained(base,m['adapter_path'],is_trainable=trainable)
 if trainable: model.enable_input_require_grads();model.gradient_checkpointing_enable()
 else: model.eval()
 return model,torch,m

def _train(binding, arm, schedule_path, root, deadline):
 from training_data.pipeline import task_to_sample
 from scripts.e04_e_equal_slot_loss import backward_equal_slot_term, backward_weighted_slot_term
 from scripts.decision_token_supervision import count_terminal_label_index, matched_weights
 import bitsandbytes as bnb
 model,torch,manifest=_load_v7(binding,trainable=True);torch.manual_seed(binding['training']['torch_seed'])
 opt=bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad],lr=binding['training']['learning_rate']); schedule=json.loads(schedule_path.read_text())['episodes'];steps=0
 for offset in range(0,384,4):
  if time.monotonic()>deadline: fail('TRAIN_CAP')
  opt.zero_grad(set_to_none=True); loss=0.0
  for item in schedule[offset:offset+4]:
   sample=task_to_sample(_serializer_task(item));ids=torch.tensor([sample['input_ids']],device='cuda:0');labels=torch.tensor([sample['labels']],device='cuda:0');out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
   allocation=binding.get('training',{}).get('loss_allocation',{'kind':'UNIFORM_SEQUENCE_LOSS'})
   if allocation.get('kind')=='COUNT_TERMINAL_TOKEN_REWEIGHTED' and item.get('role')=='PRIMARY_CONNECTED_COMPONENT_COUNT':
    decision_index,_=count_terminal_label_index(item['task']);_,treatment=matched_weights(sample['labels'],decision_index,float(allocation['decision_token_multiplier']));weights=[0.0]*len(sample['labels'])
    for index,weight in treatment.items(): weights[index]=weight
    term,_,_=backward_weighted_slot_term(out.logits,labels,weights)
   elif allocation.get('kind') in {'UNIFORM_SEQUENCE_LOSS','COUNT_TERMINAL_TOKEN_REWEIGHTED'}:
    term,_=backward_equal_slot_term(out.logits,labels)
   else: fail('LOSS_ALLOCATION_KIND')
   loss+=float(term);del out,term
  if not torch.isfinite(torch.tensor(loss)): fail('NONFINITE_LOSS')
  opt.step();steps+=1
 if steps!=96: fail('TRAIN_STEPS')
 adapter=root/'adapter';model.save_pretrained(adapter);return model,torch,manifest,sha(adapter/'adapter_model.safetensors')

def _datasets(binding, arm=None):
 base=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'; m=base/'measurement_contract_v1'; result=[]
 defaults={'OOD':'OOD_INPUT_PROMPTS.json','RETENTION':'RETENTION_INPUT_PROMPTS.json','COMPOSITION':'COMPOSITION_SENTINEL_INPUT_PROMPTS.json'}
 configured=binding.get('dataset_paths',{})
 for split,name in defaults.items():
  path=ROOT/configured[split] if split in configured else base/name
  for i,row in enumerate(json.loads(path.read_text())): result.append({'split':split,'record_id':f'{split}:{i}','task':row['task']})
 for row in json.loads((m/'PROTECTED_CAPABILITY_INPUT_PROMPTS.json').read_text())['records']: result.append({'split':'PROTECTED','record_id':'PROTECTED:'+row['protected_id'],'task':row['prompt']})
 if arm:
  for row in json.loads((m/'TRAIN_FIT_INPUT_PROMPTS.json').read_text())['records']:
   if row['arm']==arm: result.append({'split':'TRAIN_FIT','record_id':f'TRAIN_FIT:{arm}:{row["source_slot"]}','task':row['task']})
 return result

def _select_b1_items(items,binding,checkpoint_tag):
 requested=binding['evaluation']['b1_subset_by_checkpoint'].get(checkpoint_tag)
 if not isinstance(requested,list) or not requested: fail('B1_SUBSET_MISSING:'+checkpoint_tag)
 if len(requested)!=len(set(requested)): fail('B1_SUBSET_DUPLICATE:'+checkpoint_tag)
 available={item['record_id']:item for item in items}
 if len(available)!=len(items): fail('DATASET_RECORD_ID_DUPLICATE:'+checkpoint_tag)
 missing=[record_id for record_id in requested if record_id not in available]
 if missing: fail('B1_SUBSET_UNKNOWN:'+checkpoint_tag+':'+','.join(missing))
 selected=[available[record_id] for record_id in requested]
 declared=binding['evaluation'].get('batch1_gate_records_by_checkpoint',{}).get(checkpoint_tag)
 if declared is not None and int(declared)!=len(selected): fail('B1_SUBSET_COUNT:'+checkpoint_tag)
 return selected

def _generate(model,torch,manifest,items,binding,out,deadline,tag,force_batch_size=None):
 from inference.nvarc_native import NVARCNativeProvider,checkpoint_native_tokenizer,parse_native_grid
 config=ROOT/'experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json'
 c=json.loads(config.read_text()); config_dir=ROOT/'configs'/c['runtime_config_dir_name']
 tok,_=checkpoint_native_tokenizer(Path(manifest['base_path']),config_dir)
 provider=NVARCNativeProvider(model_path=Path(manifest['base_path']),tokenizer_config_dir=config_dir,device='cuda:0');provider.model,provider.tokenizer=model,tok
 ordered=sorted(items,key=lambda x:(len(canon(x['task'])),x['record_id']))
 ladder=[int(x) for x in binding['evaluation']['fallback_ladder']]
 effective=int(force_batch_size or ladder[0]); raw=[]; events=[]; i=0
 while i<len(ordered):
  if time.monotonic()>deadline: fail('EVALUATION_CAP')
  batch=ordered[i:i+effective]
  try:
   got=provider.generate_many([_messages(x['task']) for x in batch],max_new_tokens=int(binding['evaluation']['max_new_tokens']),context_window=int(binding['runtime']['context_window']),seeds=[int(binding['training']['torch_seed'])+i+j for j in range(len(batch))])
  except RuntimeError as e:
   if 'out of memory' not in str(e).lower() or effective==1: raise
   pos=ladder.index(effective)
   if pos+1>=len(ladder): raise
   events.append({'at_record':i,'from':effective,'to':ladder[pos+1],'reason':'CUDA_OOM'})
   effective=ladder[pos+1];torch.cuda.empty_cache();continue
  for item,x in zip(batch,got,strict=True):
   grid=parse_native_grid(x.text)
   raw.append({'record_id':item['record_id'],'split':item['split'],'prompt_sha256':hashlib.sha256(canon(item['task'])).hexdigest(),'generated_token_ids':[int(v) for v in x.token_ids],'parsed_grid':None if grid is None else [list(r) for r in grid],'parser_valid':grid is not None,'effective_batch_size':effective})
  i+=len(batch)
 path=out/f'{tag}_RAW.jsonl';write_jsonl(path,raw)
 write(out/f'{tag}_GENERATION_RECEIPT.json',{'requested_batch_size':ladder[0],'effective_batch_sizes':sorted({x['effective_batch_size'] for x in raw}),'fallback_events':events,'raw_sha256':sha(path),'rows':len(raw),'target_accessed':False})
 return sha(path)

def execute(binding_path,output):
 started=time.monotonic()
 try:
  b=preflight(binding_path,output,require_fresh=True);output.mkdir(parents=True);deadline=started+b['runtime']['combined_cap_seconds'];write(output/'PREFLIGHT_RECEIPT.json',{'status':'PASS_CPU_TO_GPU_GATE','model_loaded':False,'targets_accessed':False})
  v7,torch,manifest=_load_v7(b,trainable=False); v7_items=_datasets(b); raw={'V7':_generate(v7,torch,manifest,v7_items,b,output,deadline,'V7'),'V7_B1':_generate(v7,torch,manifest,_select_b1_items(v7_items,b,'V7'),b,output,deadline,'V7_B1',force_batch_size=1)};del v7
  for arm,file in [('CONTROL_NARROW_STRUCTURE','CONTROL_SCHEDULE.json'),('TREATMENT_STRUCTURE_DIVERSE','TREATMENT_SCHEDULE.json')]:
   armroot=output/arm;model,torch,manifest,adapter=_train(b,arm,ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'/file,armroot,deadline);arm_items=_datasets(b,arm);raw[arm]=_generate(model,torch,manifest,arm_items,b,output,deadline,arm);raw[arm+'_B1']=_generate(model,torch,manifest,_select_b1_items(arm_items,b,arm),b,output,deadline,arm+'_B1',force_batch_size=1);write(armroot/'ADAPTER_RECEIPT.json',{'adapter_sha256':adapter,'optimizer_steps':96});del model
  write(output/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':raw,'targets_accessed':False,'wall_seconds':time.monotonic()-started});write(output/'TERMINAL_RECEIPT.json',{'status':'COMPLETE_PENDING_SEPARATE_SCORE','raw_sha256':raw,'targets_accessed':False})
 except Exception as e:
  output.mkdir(parents=True,exist_ok=True)
  write(output/'TERMINAL_RECEIPT.json',{'status':'FAILED_OR_PARTIAL','error':f'{type(e).__name__}:{e}','targets_accessed':False,'wall_seconds':time.monotonic()-started});raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True);a.add_argument('--preflight',action='store_true');a.add_argument('--execute',action='store_true');z=a.parse_args()
 if z.preflight: print(json.dumps({'status':'PASS_E1_WORKER_CPU_PREFLIGHT','binding':preflight(z.binding,None),'model_loaded':False,'gpu_used':False,'targets_accessed':False},sort_keys=True))
 elif z.execute: execute(z.binding,z.output_root)
 else: raise SystemExit('choose --preflight or --execute')
