import copy,hashlib,importlib.util,json,pathlib,sys
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import pytest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'orchestration/supervisor'))
import kaggle_jobs as k
import arc2_supervisor as s

def manifest():
    nonce='a'*32;out='/kaggle/working/arc2_paired48_'+nonce
    return {'backend':'kaggle','job_id':'PAIR48_CPU_A001','kernel_ref':'jimmy5566/arc2-paired48-preflight-'+nonce[:12],'kernel_version':1,'unique_one_push_slug':True,'controller_role':'selector','completion_recipient':'controller','is_private':True,'kaggle_cli':k.CLI,'nonce':nonce,'source_commit':'b'*40,'kernel_code_sha256':'c'*64,'source_manifest_sha256':'d'*64,'output_root':out,'terminal_receipt':out+'/TERMINAL_RECEIPT.json','deadline_utc':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()}
def identity(v):return {'kernel_ref':v['kernel_ref'],'kernel_version':1,'is_private':True,'code_sha256':v['kernel_code_sha256']}
def worker(v):return {**{x:v[x] for x in ['job_id','kernel_ref','kernel_version','nonce','source_commit','source_manifest_sha256']},'status':'COMPLETE'}
def test_manifest_has_no_fake_transport_and_restricts_private_research():
    v=manifest();assert k.validate(v)==v
    for field,value in [('ssh_target','fake'),('primary_process',{'pid':1}),('kernel_ref','jimmy5566/arc2-fixed-4-4-d1-release'),('is_private',False),('kernel_version',2)]:
        with pytest.raises(AssertionError):k.validate({**v,field:value})
def test_platform_and_worker_identity_conflicts_fail_closed():
    v=manifest();assert k.bound_terminal(v,worker(v),identity(v))['status']=='COMPLETE'
    for field,value in [('kernel_version',2),('is_private',False),('code_sha256','f'*64)]:
        with pytest.raises(AssertionError):k.bound_terminal(v,worker(v),{**identity(v),field:value})
    with pytest.raises(AssertionError):k.bound_terminal(v,{**worker(v),'nonce':'e'*32},identity(v))
def test_real_CLI_enum_status_and_bound_output(tmp_path):
    v=manifest()
    def runner(cmd,**kw):
        if 'status' in cmd:return SimpleNamespace(returncode=0,stdout='job has status "KernelWorkerStatus.COMPLETE"')
        if cmd[0]==k.PYTHON:return SimpleNamespace(returncode=0,stdout=json.dumps(identity(v)))
        out=pathlib.Path(cmd[cmd.index('-p')+1]);(out/'TERMINAL_RECEIPT.json').write_text(json.dumps(worker(v)));return SimpleNamespace(returncode=0,stdout='')
    r=k.reconcile(v,tmp_path,runner);assert r['status']=='COMPLETE' and r['kernel_version']==1 and r['scientific_acceptance']=='NOT_INFERRED_FROM_PROCESS_RECEIPT'
def test_missing_output_is_not_success_and_deadline_is_UNKNOWN(tmp_path):
    v=manifest()
    def runner(cmd,**kw):return SimpleNamespace(returncode=0,stdout='job has status "KernelWorkerStatus.COMPLETE"' if 'status' in cmd else json.dumps(identity(v)) if cmd[0]==k.PYTHON else '')
    assert k.reconcile(v,tmp_path,runner) is None
    assert k.reconcile(v,tmp_path,runner,datetime.now(timezone.utc)+timedelta(days=1))['status']=='UNKNOWN'
def test_API_error_no_false_completion_then_UNKNOWN(tmp_path):
    v=manifest();run=lambda *a,**kw:SimpleNamespace(returncode=1,stdout='')
    assert k.reconcile(v,tmp_path,run) is None
    assert k.reconcile(v,tmp_path,run,datetime.now(timezone.utc)+timedelta(days=1))['status']=='UNKNOWN'
def test_governor_validator_and_supervisor_ingest_idempotent_route(tmp_path):
    v=manifest();registry=tmp_path/'registry';registry.mkdir();p=registry/'job.json';p.write_text(json.dumps(v))
    spec=importlib.util.spec_from_file_location('governor_kaggle_test',ROOT/'scripts/arc2_governor.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    assert g._read_remote_job_manifest(str(p))[1]==v
    manifests=s.remote_job_manifests(registry);assert len(manifests)==1 and manifests[0]['backend']=='kaggle' and 'ssh_target' not in manifests[0]
    receipt=tmp_path/'terminal.json';receipt.write_text(json.dumps({'status':'COMPLETE'}));transport={**manifests[0],'local_receipt':str(receipt),'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()}
    monitor=s.Supervisor(tmp_path/'state.json',tmp_path/'notifications',tmp_path/'events');assert monitor.reconcile_bound_remote_jobs([transport])==[v['job_id']];assert monitor.reconcile_bound_remote_jobs([transport])==[]
    event=json.loads(next((tmp_path/'events').glob('*.json')).read_text());assert event['payload']['controller_role']=='selector' and event['recipient']=='controller'

def test_prelaunch_registration_preserves_claim_and_checks_live_identity(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('governor_prelaunch_test',ROOT/'scripts/arc2_governor.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    v=manifest();p=tmp_path/'job.json';p.write_text(json.dumps(v))
    item={'event_id':'same','status':'CLAIMED','recipient':'controller','payload':{'controller_role':'selector'}}
    state={'disposition':'CONTINUE_CONTROLLER','pending_events':[item],'agent_routes':{'selector':{}},'remote_jobs':{},'active_event':'same'}
    monkeypatch.setattr(g,'load',lambda _:state);monkeypatch.setattr(g,'save',lambda *args:None)
    called=[];monkeypatch.setattr(g,'event_caller_pane',lambda *args:called.append(True))
    g.register_remote_job.__wrapped__(tmp_path/'state.json',str(p))
    assert called==[True] and item['status']=='CLAIMED' and state['active_event']=='same' and state['disposition']=='CONTINUE_CONTROLLER'
    assert (tmp_path/'remote_jobs'/ (v['job_id']+'.json')).is_file()
    def reject(*args):raise RuntimeError('REAL_IDENTITY_MISMATCH')
    monkeypatch.setattr(g,'event_caller_pane',reject)
    with pytest.raises(RuntimeError,match='REAL_IDENTITY_MISMATCH'):g.register_remote_job.__wrapped__(tmp_path/'state.json',str(p))

def test_platform_slug_alias_requires_unchanged_bytes_and_live_proof(tmp_path,monkeypatch):
    old=manifest();new={**old,'platform_kernel_ref':'jimmy5566/arc2-paired48-platform-title'}
    ident={**identity(old),'kernel_ref':new['platform_kernel_ref']}
    proof={'requested_kernel_ref':old['kernel_ref'],'nonce':old['nonce'],'job_id':old['job_id'],'platform_identity':ident}
    p=tmp_path/'proof.json';p.write_text(json.dumps(proof));new['platform_alias_receipt']={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    monkeypatch.setattr(k.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout=json.dumps(ident)))
    k.verify_prelaunch_alias(old,new);assert k.bound_terminal(new,worker(old),ident)['status']=='COMPLETE'
    for key,value in [('nonce','f'*32),('source_commit','e'*40),('kernel_code_sha256','f'*64)]:
        with pytest.raises(AssertionError):k.verify_prelaunch_alias(old,{**new,key:value})
    monkeypatch.setattr(k.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout=json.dumps({**ident,'kernel_version':2})))
    with pytest.raises(AssertionError):k.verify_prelaunch_alias(old,new)
