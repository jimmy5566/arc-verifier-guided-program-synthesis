import hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('preflight',Path('scripts/preflight_forward_targeted_repair_v1.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class PreflightTest(unittest.TestCase):
 def fixture(self):
  tmp=tempfile.TemporaryDirectory(); root=Path(tmp.name); base=root/'base';adapter=root/'adapter';base.mkdir();adapter.mkdir()
  (base/'config.json').write_text('base');(adapter/'adapter_model.safetensors').write_bytes(b'adapter')
  train=root/'train.jsonl';train.write_text('{"split":"TRAIN"}\n')
  def entry(p,r): return {'name':str(p.relative_to(r)),'bytes':p.stat().st_size,'sha256':m.sha256(p)}
  manifest={'base_path':str(base),'adapter_path':str(adapter),'base_files':[entry(base/'config.json',base)],'adapter_files':[entry(adapter/'adapter_model.safetensors',adapter)]}
  mp=root/'manifest.json';mp.write_text(json.dumps(manifest))
  config={'round_id':'test','final_audit_opened':False,'retention_sentinel_usage':'EVALUATION_ONLY_NOT_TRAINING_DATA','role_weights':{'ATOMIC_REPAIR':.45,'COMPOSITION_REPAIR':.30,'RETENTION_TRAIN':.25},'train_path':str(train),'train_sha256':m.sha256(train),'checkpoint_manifest_path':str(mp),'checkpoint_manifest_sha256':m.sha256(mp)}
  cp=root/'config.json';cp.write_text(json.dumps(config));return tmp,root,cp
 def test_valid(self):
  tmp,root,cp=self.fixture()
  with tmp:self.assertEqual(m.validate(cp,root/'fresh')['round_id'],'test')
 def test_modified_adapter_rejected(self):
  tmp,root,cp=self.fixture()
  with tmp:
   (root/'adapter'/'adapter_model.safetensors').write_bytes(b'changed')
   with self.assertRaisesRegex(RuntimeError,'ADAPTER_SIZE_MISMATCH|ADAPTER_SHA256_MISMATCH'):m.validate(cp,root/'fresh')
 def test_existing_output_rejected(self):
  tmp,root,cp=self.fixture()
  with tmp:
   (root/'used').mkdir()
   with self.assertRaisesRegex(RuntimeError,'FRESH_OUTPUT_REQUIRED'):m.validate(cp,root/'used')
if __name__=='__main__':unittest.main()
