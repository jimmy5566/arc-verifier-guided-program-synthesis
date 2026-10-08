from __future__ import annotations
import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'scripts/run_round009_posthoc_phase_a_v1.py'; B=ROOT/'scripts/run_round009_posthoc_phase_b_v1.py'; L=ROOT/'scripts/run_round009_posthoc_stage_launcher_v1.py'
def dump(p,x):p.write_text(json.dumps(x),encoding='utf-8')
class StageLauncherTests(unittest.TestCase):
 def test_dummy_phase_workers_and_launch_lock(self):
  with tempfile.TemporaryDirectory() as raw:
   r=Path(raw); source=r/'source';source.mkdir(); subprocess.run(['git','init','-q',str(source)],check=True);subprocess.run(['git','-C',str(source),'config','user.email','x@y'],check=True);subprocess.run(['git','-C',str(source),'config','user.name','x'],check=True);(source/'x').write_text('x');subprocess.run(['git','-C',str(source),'add','.'],check=True);subprocess.run(['git','-C',str(source),'commit','-qm','x'],check=True);head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
   directive=r/'directive.json';dump(directive,{'directive_id':'D','protocol_id':'P','round_id':'R','evaluation_stage_authorized':False});dh=hashlib.sha256(directive.read_bytes()).hexdigest()
   a=r/'a.json';dump(a,{'phase':'A_GENERATION','protocol_id':'P','round_id':'R','authorization_directive_id':'D','authorization_directive_sha256':dh,'nonce':'a','argv':['run'],'execution_checkout_commit':head,'execution_authorized':True,'output_root':str(r/'aout')});lock=r/'lock.json';dump(lock,{'binding_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'execution_checkout_commit':head,'phase':'A_GENERATION','nonce':'a'})
   self.assertEqual(0,subprocess.run([sys.executable,str(L),'--phase','a','--binding',str(a),'--directive',str(directive),'--source-root',str(source),'--runtime-lock',str(lock),'--dry-run'],capture_output=True,text=True).returncode)
   im=r/'im.json';dump(im,{'contains_targets':False,'items':[{'sample_id':'s'}]});prompts=r/'prompts.jsonl';prompts.write_text(json.dumps({'sample_id':'s','prompt_ids':[14,10],'prompt_sha256':'p'})+'\n')
   ar=r/'ar.json';self.assertEqual(0,subprocess.run([sys.executable,str(A),'--binding',str(a),'--surface','NOVEL','--input-manifest',str(im),'--prompt-input-jsonl',str(prompts),'--output',str(r/'pa'),'--receipt',str(ar),'--execute','--dummy-worker'],capture_output=True,text=True).returncode)
   def rows(surface,n):return [{'surface':surface,'manifest_item_id':f'{surface}{i}','canonical_reference_output_grid_sha256':'x'} for i in range(n)]
   mapping=r/'mapping.json';dump(mapping,{'surfaces':{'NOVEL':rows('NOVEL',128),'REPLAY':rows('REPLAY',64),'PROTECTED':rows('PROTECTED',64)}});pred=r/'pred.json';dump(pred,{'predictions':[{'surface':x['surface'],'sample_id':x['manifest_item_id'],'prompt_sha256':'p','generated_token_ids':[0,15],'prediction_sha256':'q','parse_status':'OK'} for z in json.loads(mapping.read_text())['surfaces'].values() for x in z]})
   b=r/'b.json';dump(b,{'phase':'B_SEALED_SCORING','requires_phase_a_terminal_immutable':True,'execution_authorized':True,'nonce':'b'});br=r/'br.json';proc=subprocess.run([sys.executable,str(B),'--binding',str(b),'--prediction-manifest',str(pred),'--phase-a-receipt',str(ar),'--sealed-mapping',str(mapping),'--output',str(r/'pb'),'--receipt',str(br),'--execute'],capture_output=True,text=True);self.assertEqual(0,proc.returncode,proc.stderr)
 def test_launcher_rejects_checkout_substitution(self):
  with tempfile.TemporaryDirectory() as raw:
   r=Path(raw);source=r/'source';source.mkdir();subprocess.run(['git','init','-q',str(source)],check=True);subprocess.run(['git','-C',str(source),'config','user.email','x@y'],check=True);subprocess.run(['git','-C',str(source),'config','user.name','x'],check=True);(source/'x').write_text('x');subprocess.run(['git','-C',str(source),'add','.'],check=True);subprocess.run(['git','-C',str(source),'commit','-qm','x'],check=True)
   d=r/'d';dump(d,{'directive_id':'D','protocol_id':'P','round_id':'R'});dh=hashlib.sha256(d.read_bytes()).hexdigest();a=r/'a';dump(a,{'phase':'A_GENERATION','protocol_id':'P','round_id':'R','authorization_directive_id':'D','authorization_directive_sha256':dh,'nonce':'n','output_root':str(r/'o')});lock=r/'l';dump(lock,{'binding_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'execution_checkout_commit':'0'*40,'phase':'A_GENERATION','nonce':'n'})
   x=subprocess.run([sys.executable,str(L),'--phase','a','--binding',str(a),'--directive',str(d),'--source-root',str(source),'--runtime-lock',str(lock),'--dry-run'],capture_output=True,text=True);self.assertNotEqual(0,x.returncode);self.assertIn('LIVE_CHECKOUT_MISMATCH',x.stderr)
if __name__=='__main__':unittest.main()
