from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from scripts.launch_paired_v7_family_balanced_rank_margin_v1 import consume_reviewed_binding,execute_bounded
from scripts.paired_rank_margin_launch_contract import acquire_live_lock,failure_receipt,release_live_lock,require_checkpoint_records,require_launch,sha
class LaunchContractTests(unittest.TestCase):
 def test_identity_cap_and_fresh_output(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); inp=root/'i';inp.write_text('x');out=root/'run'/'result.json';rec=root/'run'/'receipt.json'
   require_launch(output=out,receipt=rec,cap_seconds=900,expected_hashes={'i':sha(inp)},actual_paths={'i':inp})
   (root/'run').mkdir()
   require_launch(output=out,receipt=rec,cap_seconds=900,expected_hashes={'i':sha(inp)},actual_paths={'i':inp})
   out.write_text('done')
   with self.assertRaisesRegex(RuntimeError,'FRESH'):
    require_launch(output=out,receipt=rec,cap_seconds=900,expected_hashes={'i':sha(inp)},actual_paths={'i':inp})
 def test_live_lock_rejects_duplicate_and_releases(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'stable-parent'/'.nonce.live.lock';acquire_live_lock(p)
   with self.assertRaisesRegex(RuntimeError,'DUPLICATE'):acquire_live_lock(p)
   release_live_lock(p);acquire_live_lock(p)
 def test_failure_receipt_is_atomic_and_nonoverwrite(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'r.json';failure_receipt(receipt=p,reason='X');self.assertEqual(__import__('json').loads(p.read_text())['status'],'FAILED_NO_UPDATE')
   with self.assertRaisesRegex(RuntimeError,'NON_OVERWRITE'):failure_receipt(receipt=p,reason='X')
 def test_checkpoint_discovery_requires_manifests_and_adapters(self):
  d={'checkpoint_records':{'V7':{'manifest_path':'m','manifest_sha256':'a','adapter_model_sha256':'b'},'FB':{'manifest_path':'m','manifest_sha256':'c','adapter_model_sha256':'d'}}}
  self.assertEqual(set(require_checkpoint_records(d,('V7','FB'))),{'V7','FB'})
 def test_timeout_receipt_preserves_compact_partial_evidence_and_releases_lock(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);partial=root/'output'/'PARTIAL_BATCH_EVIDENCE';partial.mkdir(parents=True);(partial/'x.json').write_text('{}');lock=root/'.n.live.lock';acquire_live_lock(lock);receipt=root/'TERMINAL_RECEIPT.json'
   process=Mock(pid=123);process.wait.side_effect=[__import__('subprocess').TimeoutExpired('worker',1),None]
   with patch('scripts.launch_paired_v7_family_balanced_rank_margin_v1.subprocess.Popen',return_value=process),patch('scripts.launch_paired_v7_family_balanced_rank_margin_v1.os.killpg',create=True):
    self.assertEqual(execute_bounded(['worker'],receipt=receipt,cap_seconds=1,partial_evidence=partial,live_lock=lock),124)
   saved=json.loads(receipt.read_text());self.assertEqual(saved['partial_batch_journal_count'],1);self.assertFalse(lock.exists())
 def test_mocked_worker_completion_preserves_existing_terminal_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);receipt=root/'TERMINAL_RECEIPT.json';receipt.write_text('{"status":"COMPLETE_NO_UPDATE"}');process=Mock();process.wait.return_value=0
   with patch('scripts.launch_paired_v7_family_balanced_rank_margin_v1.subprocess.Popen',return_value=process):
    self.assertEqual(execute_bounded(['worker'],receipt=receipt,cap_seconds=1,partial_evidence=root/'partial',live_lock=root/'lock'),0)
   self.assertEqual(json.loads(receipt.read_text())['status'],'COMPLETE_NO_UPDATE')
 def test_exact_reviewed_binding_is_consumed_and_source_parity_is_enforced(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);files={}
   for key in ('contract_sha256','runtime_contract_sha256','launch_contract_sha256','worker_sha256','launcher_sha256','postprocessor_sha256'):
    p=root/'scripts'/f'{key}.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(key);files[key]=str(p.relative_to(root))
   input_paths={}
   for key in ('raw_predictions','input_manifest','v7_manifest','family_balanced_manifest'):
    p=root/f'{key}.json';p.write_text(key);input_paths[key]=str(p.relative_to(root))
   sidecar=root/'sidecar.json';sidecar.write_text('sidecar')
   hashes={key:sha(root/value) for key,value in input_paths.items()};hashes['sealed_sidecar']=sha(sidecar)
   config={'implementation_files':files,'implementation_identity':{key:sha(root/value) for key,value in files.items()},'runtime_inputs':{'expected_hashes':hashes|{'sealed_sidecar':'ignored'},**{f'{key}_path':value for key,value in input_paths.items()}},'sealed_sidecar_sha256':sha(sidecar),'launch_identity':{'nonce':'n'*16},'checkpoints':{'V7':{}},'batch1_subset_episode_ids':[str(i) for i in range(12)]}
   config_sha=hashlib.sha256(json.dumps(config).encode()).hexdigest();output=root/'remote-output';receipt=output/'TERMINAL_RECEIPT.json'
   binding={'protocol_id':'PAIRED_V7_FAMILY_BALANCED_CORRECT_TOKEN_RANK_MARGIN_V1','source_commit':'source','origin_ref':'origin/test','runtime_cap_seconds':900,'nonce':'n'*16,'input_paths':{key:str(root/value) for key,value in input_paths.items()}|{'sealed_sidecar':str(sidecar)},'expected_hashes':hashes,'checkpoints':config['checkpoints'],'batch1_subset_episode_ids':config['batch1_subset_episode_ids'],'output':str(output),'receipt':str(receipt),'implementation_identity':{'config_sha256':config_sha,**config['implementation_identity']},'no_update':True,'forbidden':[]}
   bp=root/'binding.json';bp.write_text(json.dumps(binding));digest=sha(bp)
   with patch('scripts.launch_paired_v7_family_balanced_rank_margin_v1.git_head',return_value='runtime'),patch('scripts.launch_paired_v7_family_balanced_rank_margin_v1.git_ref',return_value='runtime'),patch('scripts.launch_paired_v7_family_balanced_rank_margin_v1.blob_sha_at_commit',return_value=digest),patch('scripts.launch_paired_v7_family_balanced_rank_margin_v1.is_ancestor',return_value=True),patch('scripts.launch_paired_v7_family_balanced_rank_margin_v1.require_clean_tracked_checkout'):
    self.assertEqual(consume_reviewed_binding(root=root,binding_path=bp,binding_sha256=digest,config=config,config_sha256=config_sha,sidecar=sidecar,output=output,receipt=receipt,expected_commit='runtime')['nonce'],'n'*16)
   with patch('scripts.launch_paired_v7_family_balanced_rank_margin_v1.git_head',return_value='wrong'),patch('scripts.launch_paired_v7_family_balanced_rank_margin_v1.git_ref',return_value='runtime'):
    with self.assertRaisesRegex(RuntimeError,'SOURCE_PARITY'):consume_reviewed_binding(root=root,binding_path=bp,binding_sha256=digest,config=config,config_sha256=config_sha,sidecar=sidecar,output=output,receipt=receipt,expected_commit='runtime')
if __name__=='__main__':unittest.main()
