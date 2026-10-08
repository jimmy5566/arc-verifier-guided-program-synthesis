from __future__ import annotations
import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'scripts/run_round009_posthoc_phase_a_v1.py'; B=ROOT/'scripts/run_round009_posthoc_phase_b_v1.py'; L=ROOT/'scripts/run_round009_posthoc_stage_launcher_v1.py'; LOCK=ROOT/'scripts/freeze_round009_posthoc_runtime_lock_v1.py'
def dump(p,x):p.write_text(json.dumps(x),encoding='utf-8')
class StageLauncherTests(unittest.TestCase):
 def test_dummy_phase_preflights_and_launch_lock(self):
  with tempfile.TemporaryDirectory() as raw:
   r=Path(raw); source=r/'source';source.mkdir(); subprocess.run(['git','init','-q',str(source)],check=True);subprocess.run(['git','-C',str(source),'config','user.email','x@y'],check=True);subprocess.run(['git','-C',str(source),'config','user.name','x'],check=True);(source/'x').write_text('x');subprocess.run(['git','-C',str(source),'add','.'],check=True);subprocess.run(['git','-C',str(source),'commit','-qm','x'],check=True);head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
   directive=r/'directive.json';dump(directive,{'protocol_id':'P','round_id':'R','evaluation_stage_authorized':False})
   a=r/'a.json';dump(a,{'phase':'A_GENERATION','protocol_id':'P','round_id':'R','nonce':'a','argv':['run'], 'execution_checkout_commit':head,'output_root':str(r/'aout')})
   lock=r/'lock.json';dump(lock,{'binding_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'execution_checkout_commit':head,'phase':'A_GENERATION','nonce':'a'})
   self.assertEqual(0,subprocess.run([sys.executable,str(L),'--phase','a','--binding',str(a),'--directive',str(directive),'--source-root',str(source),'--runtime-lock',str(lock),'--dry-run'],capture_output=True,text=True).returncode)
   ar=r/'ar.json'; self.assertEqual(0,subprocess.run([sys.executable,str(A),'--binding',str(a),'--output',str(r/'pa'),'--receipt',str(ar),'--preflight'],capture_output=True,text=True).returncode)
   mapping=r/'mapping.json';dump(mapping,{'surfaces':{'NOVEL':[]}});pred=r/'pred.json';dump(pred,{'predictions':[{'sample_id':'s','prompt_sha256':'a','prediction_sha256':'b','parse_status':'OK'}]})
   b=r/'b.json';dump(b,{'phase':'B_SEALED_SCORING','requires_phase_a_terminal_immutable':True})
   self.assertEqual(0,subprocess.run([sys.executable,str(B),'--binding',str(b),'--prediction-manifest',str(pred),'--phase-a-receipt',str(ar),'--sealed-mapping',str(mapping),'--output',str(r/'pb'),'--receipt',str(r/'br'),'--preflight'],capture_output=True,text=True).returncode)
 def test_launcher_rejects_checkout_substitution(self):
  with tempfile.TemporaryDirectory() as raw:
   r=Path(raw); source=r/'source';source.mkdir();subprocess.run(['git','init','-q',str(source)],check=True);subprocess.run(['git','-C',str(source),'config','user.email','x@y'],check=True);subprocess.run(['git','-C',str(source),'config','user.name','x'],check=True);(source/'x').write_text('x');subprocess.run(['git','-C',str(source),'add','.'],check=True);subprocess.run(['git','-C',str(source),'commit','-qm','x'],check=True)
   d=r/'d';dump(d,{'protocol_id':'P','round_id':'R'});a=r/'a';dump(a,{'phase':'A_GENERATION','protocol_id':'P','round_id':'R','nonce':'n','output_root':str(r/'o')});lock=r/'l';dump(lock,{'binding_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'execution_checkout_commit':'0'*40,'phase':'A_GENERATION','nonce':'n'})
   x=subprocess.run([sys.executable,str(L),'--phase','a','--binding',str(a),'--directive',str(d),'--source-root',str(source),'--runtime-lock',str(lock),'--dry-run'],capture_output=True,text=True)
   self.assertNotEqual(0,x.returncode);self.assertIn('LIVE_CHECKOUT_MISMATCH',x.stderr)
if __name__=='__main__':unittest.main()
