#!/usr/bin/env python3
"""CPU-only production-shaped integration for the V7 executable successor."""
from __future__ import annotations
import hashlib, importlib.util, json, tempfile, sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v7exec", ROOT / "scripts" / "v7_component_diagnostic_executable_v2.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class FakeRuntime:
 def __init__(self, raw_root, *, oom=False, b1_mismatch=False, teacher_drift=False, dependency_error=False, malformed=False):
  self.raw_root=Path(raw_root); self.oom=oom; self.b1_mismatch=b1_mismatch; self.teacher_drift=teacher_drift; self.dependency_error=dependency_error; self.malformed=malformed; self.calls=[]; self.sidecar_opened=False; self.teacher_forward_events=[]
 def check_dependencies(self):
  if self.dependency_error: raise RuntimeError('DEPENDENCY_MISSING')
 def verify_adapter(self,path,expected):
  if digest(path)!=expected: raise RuntimeError('V7_ADAPTER_IDENTITY_MISMATCH')
 def verify_tokenizer(self,descriptor): assert descriptor['files']
 def empty_cache(self): pass
 def generate(self,records,batch_size,max_new_tokens):
  self.calls.append(('generation',batch_size,len(records)))
  if self.oom and batch_size==32:
   self.oom=False; raise RuntimeError('CUDA out of memory')
  if self.malformed: return [{'generated_token_ids':'bad','parser_valid':False} for _ in records]
  out=[]
  for row in records:
   number=int(row['cohort_record_id'].rsplit(':',1)[1]); tokens=[2, 3]
   if self.b1_mismatch and batch_size==1: tokens=[9,3]
   out.append({'generated_token_ids':tokens,'parsed_grid':[[number % 4]],'parser_valid':True})
  return out
 def open_sidecar(self,descriptor):
  assert (self.raw_root/'RAW_FROZEN.jsonl').exists(), 'SIDECAR_OPENED_BEFORE_RAW'
  self.sidecar_opened=True
  return {f'R:{i:02d}':{'target_token_ids':[2,3],'target_grid':[[i%4]]} for i in range(72)}
 def teacher_forced(self,raw_rows,sidecar,batch_size):
  self.calls.append(('teacher',batch_size,len(raw_rows)))
  if batch_size==32: assert len(raw_rows)==72, 'PRIMARY_TEACHER_MUST_BE_PHYSICAL_BATCH32_PATH'
  if batch_size==1: assert len(raw_rows)==12, 'B1_TEACHER_MUST_BE_PHYSICAL_B1_PATH'
  self.teacher_forward_events.append({'event':'PHYSICAL_TEACHER_FORWARD','requested_batch_size':batch_size,'records':len(raw_rows)})
  first=[0.0,4.0,0.0,1.0] if self.teacher_drift and batch_size==1 else [0.0,0.0,4.0,1.0]
  return {row['cohort_record_id']:[first,[0.0,0.0,1.0,4.0]] for row in raw_rows}

def fixture(base):
 base=Path(base); base.mkdir(parents=True,exist_ok=True); protocol=base/'protocol.json'; cohort=base/'cohort.json'; adapter=base/'adapter.bin'; sidecar=base/'sidecar.json'; subset=base/'batch1.json'
 protocol.write_text('{}'); adapter.write_bytes(b'V7-ADAPTER'); sidecar.write_text('{}'); native=base/'native'; native.mkdir(); (native/'chat_template.j2').write_text('native-template')
 records=[{'cohort_record_id':f'R:{i:02d}','prompt_task_sha256':f'{i:064x}','task':{'train':[],'test':[{'input':[[i%4]]}]}} for i in range(72)]
 cohort.write_text(json.dumps({'records':records},sort_keys=True))
 subset.write_text(json.dumps({'record_ids':[f'R:{i:02d}' for i in range(0,72,6)]},sort_keys=True))
 source='scripts/v7_component_diagnostic_executable_v2.py'
 binding={'protocol_sha256':digest(protocol),'protocol':{'path':str(protocol),'sha256':digest(protocol)},'cohort':{'path':str(cohort),'sha256':digest(cohort)},'batch1_subset':{'path':str(subset),'sha256':digest(subset)},'teacher_forced_tolerance':{'rank_abs_delta_max':0,'top_count_delta_max':0,'margin_abs_delta_max':0.02,'nll_abs_delta_max':0.005},'sidecar':{'path':str(sidecar),'remote_path':str(sidecar),'sha256':digest(sidecar)},'tokenizer':{'runtime_path':str(base),'files':{'protocol.json':digest(protocol)}},'checkpoint':{'adapter_model_path':str(adapter),'adapter_model_sha256':digest(adapter)},'execution':{'checkpoint':'V7','retry':False,'optimizer_steps':0,'backward_calls':0,'hard_cap_seconds':3600,'requested_batch_size':32,'fallback_ladder':[32,16,8,4,1],'mandatory_watchdog':True},'runtime':{'base_path':str(base),'adapter_dir':str(base),'native_config_dir':str(native),'native_config_tree_sha256':M.tree_sha256(native),'native_token_contract':{'eos_token_id':15,'pad_token_id':13,'padding':'LEFT_EXPLICIT','num_beams':1}},'source_hashes':{source:digest(ROOT/source)}}
 binding_path=base/'binding.json'; binding_path.write_text(json.dumps(binding,sort_keys=True)); auth=base/'auth.json'; auth.write_text(json.dumps({'status':'VALID_UNCONSUMED','binding_sha256':digest(binding_path)}))
 return binding_path,auth,adapter

def expect(code, fn):
 try: fn()
 except RuntimeError as exc: assert code in str(exc), (code,exc)
 else: raise AssertionError(code+' not raised')

def successful_run(base, **kwargs):
 binding,auth,_=fixture(base); out=Path(base)/'out'; control=Path(base)/'control'; runtime=FakeRuntime(out,**kwargs)
 result=M.launch(binding,runtime,auth,out,control)
 return result,runtime,out,control,binding,auth

def main():
 with tempfile.TemporaryDirectory() as tmp:
  result,runtime,out,control,binding,auth=successful_run(Path(tmp)/'pass',oom=True)
  assert result['status']=='COMPLETE' and runtime.sidecar_opened
  assert result['teacher_forced_forwards']==72 and result['batch1_teacher_forced_forwards']==12 and result['batch1_generations']==12
  assert ('generation',32,32) in runtime.calls and any(kind=='generation' and batch==16 for kind,batch,_ in runtime.calls) and ('generation',1,12) in runtime.calls
  assert ('teacher',32,72) in runtime.calls and ('teacher',1,12) in runtime.calls
  assert [item['requested_batch_size'] for item in result['teacher_forward_events']]==[32,1]
  row=json.loads((out/'SCORED_METRICS.jsonl').read_text().splitlines()[0]); assert row['teacher_forced']['token_weighted_nll'] > 0 and row['teacher_forced']['mean_margin'] == 3.0
  assert json.loads((out/'RAW_FREEZE_RECEIPT.json').read_text())['fallback_events']==[{'from':32,'to':16,'reason':'CUDA_OOM'}]

  result,_,_,_,_,_=successful_run(Path(tmp)/'b1fail',b1_mismatch=True); assert result['status']=='INVALID_NOT_INTERPRETABLE'
  result,_,_,_,_,_=successful_run(Path(tmp)/'teacherfail',teacher_drift=True); assert result['status']=='INVALID_NOT_INTERPRETABLE'
  assert M.taxonomy([[1,0,1]],[[1,1,0]],True)['primary']=='CONNECTED_COMPONENT_COUNT'
  assert M.taxonomy([[0,0,1]],[[1,0,0]],True)['primary']=='SELECTED_OBJECT_IDENTITY'
  assert M.taxonomy([[1,0,0],[1,0,0]],[[1,1,0],[0,0,0]],True)['primary']=='GEOMETRY'
  assert M.taxonomy([[2]],[[1]],True)['primary']=='COLOR'
  binding,auth,_=fixture(Path(tmp)/'badsource'); obj=json.loads(binding.read_text()); obj['source_hashes']['scripts/v7_component_diagnostic_executable_v2.py']='0'*64; binding.write_text(json.dumps(obj)); expect('SOURCE_IDENTITY_MISMATCH',lambda:M.validate_binding(binding))
  binding,auth,_=fixture(Path(tmp)/'badcohort'); obj=json.loads(binding.read_text()); obj['cohort']['sha256']='0'*64; binding.write_text(json.dumps(obj)); expect('COHORT_IDENTITY_MISMATCH',lambda:M.validate_binding(binding))
  binding,auth,_=fixture(Path(tmp)/'badadapter'); obj=json.loads(binding.read_text()); obj['checkpoint']['adapter_model_sha256']='0'*64; binding.write_text(json.dumps(obj)); auth.write_text(json.dumps({'status':'VALID_UNCONSUMED','binding_sha256':digest(binding)})); expect('V7_ADAPTER_IDENTITY_MISMATCH',lambda:M.launch(binding,FakeRuntime(Path(tmp)/'badadapter'/'out'),auth,Path(tmp)/'badadapter'/'out',Path(tmp)/'badadapter'/'control'))
  binding,auth,_=fixture(Path(tmp)/'dependency'); expect('DEPENDENCY_MISSING',lambda:M.launch(binding,FakeRuntime(Path(tmp)/'dependency'/'out',dependency_error=True),auth,Path(tmp)/'dependency'/'out',Path(tmp)/'dependency'/'control'))
  binding,auth,_=fixture(Path(tmp)/'root'); (Path(tmp)/'root'/'out').mkdir(); expect('FRESH_OUTPUT_ROOT_REQUIRED',lambda:M.launch(binding,FakeRuntime(Path(tmp)/'root'/'out'),auth,Path(tmp)/'root'/'out',Path(tmp)/'root'/'control'))
  binding,auth,_=fixture(Path(tmp)/'malformed'); out=Path(tmp)/'malformed'/'out';control=Path(tmp)/'malformed'/'control'; expect('MALFORMED_GENERATION_TOKENS',lambda:M.launch(binding,FakeRuntime(out,malformed=True),auth,out,control)); assert json.loads((control/'TERMINAL_RECEIPT.json').read_text())['status']=='FAILED_NO_UPDATE'; expect('DUPLICATE_OR_CONSUMED_ONE_SHOT',lambda:M.launch(binding,FakeRuntime(Path(tmp)/'malformed'/'other',malformed=True),auth,Path(tmp)/'malformed'/'other',Path(tmp)/'malformed'/'other_control'))
  binding,auth,_=fixture(Path(tmp)/'timeout'); out=Path(tmp)/'timeout'/'out';control=Path(tmp)/'timeout'/'control'
  with mock.patch.object(M.time,'monotonic',side_effect=[0,4000]): expect('EXECUTION_TIMEOUT',lambda:M.launch(binding,FakeRuntime(out),auth,out,control))
  assert json.loads((control/'TERMINAL_RECEIPT.json').read_text())['status']=='FAILED_NO_UPDATE'
  watchdog=Path(tmp)/'watchdog.json'; assert M.enforce_hard_cap([sys.executable,'-c','import time; time.sleep(1)'],watchdog,seconds=.01) is None; assert json.loads(watchdog.read_text())['status']=='FAILED_NO_UPDATE_TIMEOUT'
  # Native prompt/token fixture: no JSON transport or ad-hoc chat prompt may enter the executor.
  messages=M.native_prompt_messages({'task':{'train':[{'input':[[1,2]],'output':[[2,1]]}],'test':[{'input':[[3,4]]}]}})
  assert messages==[{'role':'user','content':'12'},{'role':'assistant','content':'21'},{'role':'user','content':'34'}]
  class PromptTokenizer:
   def apply_chat_template(self, got, **kwargs):
    assert got==messages and kwargs['add_generation_prompt'] and kwargs['tokenize']
    class Row:
     def tolist(self): return [14,11,1,2,15,14,12,2,1,15,14,11,3,4,15,14,12,10]
    return {'input_ids':[Row()]}
  assert M.native_prompt_token_ids(PromptTokenizer(), {'task':{'train':[{'input':[[1,2]],'output':[[2,1]]}],'test':[{'input':[[3,4]]}]}})[0:2]==[14,11]
  # Production --launch is a watchdog parent; direct child execution is rejected without its environment marker.
  binding,auth,_=fixture(Path(tmp)/'childguard'); command=M.production_child_command(type('Args',(),{'binding':binding,'authorization':auth,'output_root':Path(tmp)/'childguard'/'out','control_root':Path(tmp)/'childguard'/'control'})())
  assert '--execute-claimed' in command and '--watch-command' not in Path(M.__file__).read_text()
  with mock.patch.object(sys,'argv',[str(Path(M.__file__)),'--binding',str(binding),'--authorization',str(auth),'--output-root',str(Path(tmp)/'childguard'/'out'),'--control-root',str(Path(tmp)/'childguard'/'control'),'--execute-claimed']):
   expect('DIRECT_CHILD_EXECUTION_REFUSED', M.main)
 print('PASS_V7_EXECUTABLE_V2_PRODUCTION_SHAPED_CPU_INTEGRATION')

if __name__=='__main__': main()
