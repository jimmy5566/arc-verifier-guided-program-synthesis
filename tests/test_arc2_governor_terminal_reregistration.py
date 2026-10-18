"""No live services: exact receipt reconciliation for terminal registration race."""
import copy,json,tempfile,unittest
from pathlib import Path
from test_arc2_governor_thin_router import governor as g

class TerminalRegistrationTests(unittest.TestCase):
    def fixture(self,root):
        receipt={'job_id':'OLD','status':'COMPLETE','execution_source_commit':'a'*40,
                 'receipt_origin':'LOCAL_SUPERVISOR_PIPELINE_OBSERVATION_V1',
                 'observed_process_identity':{}}
        path=root/'receipt.json';g.atomic(path,receipt)
        old={'job_id':'OLD','controller_role':'selector','source_commit':'a'*40}
        payload={'round_id':'OLD','controller_role':'selector','bound_remote_job':True,
                 'terminal_status':'COMPLETE','terminal_receipt':str(path),
                 'terminal_receipt_sha256':g.sha256_file(path)}
        item={'event_id':'terminal_event','status':'COMPLETE','payload':payload}
        state={'remote_jobs':{'OLD':old},'pending_events':[item]}
        new={'job_id':'NEW','controller_role':'selector','source_commit':'b'*40}
        return state,new,path,receipt
    def test_exact_completed_receipt_retires_only_stale_entry(self):
        with tempfile.TemporaryDirectory() as td:
            state,new,path,_=self.fixture(Path(td));event=copy.deepcopy(state['pending_events'])
            other={'job_id':'ORACLE','controller_role':'oracle'};state['remote_jobs']['ORACLE']=other
            result=g.retire_completed_role_jobs(state,new)
            self.assertEqual(result[0]['job_id'],'OLD');self.assertEqual(state['remote_jobs'],{'ORACLE':other})
            self.assertEqual(state['pending_events'],event);self.assertTrue(path.exists())
    def test_live_unknown_uncompleted_wrong_hash_or_source_still_protected(self):
        with tempfile.TemporaryDirectory() as td:
            for mutation in ['live','unknown','uncompleted','hash','source','role']:
                state,new,path,receipt=self.fixture(Path(td));p=state['pending_events'][0]['payload']
                if mutation=='live':receipt['observed_process_identity']={'pid':123}
                if mutation=='unknown':receipt['status']='UNKNOWN'
                if mutation=='source':receipt['execution_source_commit']='c'*40
                if mutation in ['live','unknown','source']:g.atomic(path,receipt);p['terminal_receipt_sha256']=g.sha256_file(path)
                if mutation=='uncompleted':state['pending_events'][0]['status']='CLAIMED'
                if mutation=='hash':p['terminal_receipt_sha256']='0'*64
                if mutation=='role':p['controller_role']='oracle'
                self.assertEqual(g.retire_completed_role_jobs(state,new),[],mutation)
                self.assertIn('OLD',state['remote_jobs'])
                with self.assertRaisesRegex(RuntimeError,'ACTIVE_REMOTE_JOB_PROTECTED'):g.protect_remote_owner(state,new)
    def test_same_manifest_is_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            state,_,_,_=self.fixture(Path(td));same=state['remote_jobs']['OLD']
            self.assertEqual(g.retire_completed_role_jobs(state,same),[])
            self.assertIn('OLD',state['remote_jobs'])

if __name__=='__main__':unittest.main()
