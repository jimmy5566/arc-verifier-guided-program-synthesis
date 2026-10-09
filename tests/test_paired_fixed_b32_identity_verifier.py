import hashlib, importlib.util, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
S=importlib.util.spec_from_file_location('v',ROOT/'scripts/verify_paired_fixed_b32_runtime_identity.py');V=importlib.util.module_from_spec(S);S.loader.exec_module(V)
class IdentityVerifierTest(unittest.TestCase):
 def test_missing_hash_and_extra_weight_fail(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for name,data in [('model-00001-of-00001.safetensors',b'a'),('config.json',b'{}'),('tokenizer_config.json',b'{}')]: (p/name).write_bytes(data)
   records=[{'name':x.name,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in p.iterdir()]
   V.verify_files(p,records,'BASE')
   (p/'model-00002-of-00002.safetensors').write_bytes(b'b')
   with self.assertRaises(RuntimeError):V.verify_files(p,records,'BASE')
   (p/'model-00002-of-00002.safetensors').unlink()
   records[0]['sha256']='0'*64
   with self.assertRaises(RuntimeError):V.verify_files(p,records,'BASE')
 def test_adapter_requires_expected_weight_and_config_and_rejects_extra(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'adapter_model.safetensors').write_bytes(b'a');(p/'adapter_config.json').write_bytes(b'{}')
   records=[{'name':x.name,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in p.iterdir()]
   V.verify_files(p,records,'ADAPTER')
   (p/'adapter_model.bin').write_bytes(b'b')
   with self.assertRaises(RuntimeError):V.verify_files(p,records,'ADAPTER')
   with self.assertRaises(RuntimeError):V.verify_files(p,records[:1],'ADAPTER')
if __name__=='__main__':unittest.main()
