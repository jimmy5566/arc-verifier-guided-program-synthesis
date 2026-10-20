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

def test_controlled_stop_keeps_exact_status_in_same_role_callback(tmp_path):
    v=manifest();p=tmp_path/'terminal.json';p.write_text(json.dumps({'status':'CONTROLLED_STOP','terminal_status':'CONTROLLED_STOP'}))
    monitor=s.Supervisor(tmp_path/'state.json',tmp_path/'notifications',tmp_path/'events')
    assert monitor.reconcile_bound_remote_jobs([{**v,'manifest_path':str(tmp_path/'job.json'),'local_receipt':str(p),'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}])==[v['job_id']]
    e=json.loads(next((tmp_path/'events').glob('*.json')).read_bytes())
    assert e['payload']['terminal_status']=='CONTROLLED_STOP' and e['payload']['controller_role']=='selector'
def test_platform_and_worker_identity_conflicts_fail_closed():
    v=manifest();assert k.bound_terminal(v,worker(v),identity(v))['status']=='COMPLETE'
    for field,value in [('kernel_version',2),('is_private',False),('code_sha256','f'*64)]:
        with pytest.raises(AssertionError):k.bound_terminal(v,worker(v),{**identity(v),field:value})
    with pytest.raises(AssertionError):k.bound_terminal(v,{**worker(v),'nonce':'e'*32},identity(v))
def test_real_CLI_enum_status_and_bound_output(tmp_path):
    v=manifest()
    def runner(cmd,**kw):
        if 'status' in cmd:return SimpleNamespace(returncode=0,stdout='job has status "KernelWorkerStatus.COMPLETE"')
        if '-c' in cmd:return SimpleNamespace(returncode=0,stdout=json.dumps(identity(v)))
        out=pathlib.Path(cmd[cmd.index('--output')+1]);(out/'TERMINAL_RECEIPT.json').write_text(json.dumps(worker(v)));return SimpleNamespace(returncode=0,stdout='')
    r=k.reconcile(v,tmp_path,runner);assert r['status']=='COMPLETE' and r['kernel_version']==1 and r['scientific_acceptance']=='NOT_INFERRED_FROM_PROCESS_RECEIPT'
def test_missing_output_is_not_success_and_deadline_is_UNKNOWN(tmp_path):
    v=manifest()
    def runner(cmd,**kw):return SimpleNamespace(returncode=0,stdout='job has status "KernelWorkerStatus.COMPLETE"' if 'status' in cmd else json.dumps(identity(v)) if '-c' in cmd else '')
    assert k.reconcile(v,tmp_path,runner)['status']=='UNKNOWN'
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

def notebook_manifest():
    return {**manifest(),'kernel_ref':'jimmy5566/arc2-l4x4-runtime-template','kernel_id':133930602,'kernel_type':'notebook','prior_kernel_version':4,'kernel_version':5,'prior_kernel_code_sha256':'95a0642d616f01f18e7b57875373cfcdce7d3317bdd89482cd30038673d51a30','unique_one_push_slug':False,'single_version_push':True,'owner_amendment_sha256':'3375fe13721877fb755248d513b536047348000f580b458698afad60548042a8'}

def test_exact_owner_notebook_version_extension_is_narrow():
    v=notebook_manifest();k.validate(v)
    for key,value in [('kernel_id',1),('kernel_ref','jimmy5566/arc2-fixed-4-4-d1-release'),('kernel_version',4),('kernel_version',1),('is_private',False),('owner_amendment_sha256','0'*64)]:
        with pytest.raises(AssertionError):k.validate({**v,key:value})
    with pytest.raises(AssertionError):k.validate({**manifest(),'kernel_version':5})

def test_notebook_stale_newer_or_wrong_id_never_uses_cached_output(tmp_path):
    v=notebook_manifest();i={**identity(v),'kernel_version':5,'kernel_id':133930602};k.bound_terminal(v,worker(v),i)
    for bad in [{**i,'kernel_version':4},{**i,'kernel_version':6},{**i,'kernel_id':123}]:
        with pytest.raises(AssertionError):k.bound_terminal(v,worker(v),bad)

def test_actual_notebook_identity_code_allows_only_JSON_serialization(monkeypatch,capsys):
    from contextlib import contextmanager
    from kaggle.api import kaggle_api_extended
    v=notebook_manifest();notebook={'nbformat':4,'cells':[{'source':'print(123)\n','cell_type':'code'}],'metadata':{}}
    @contextmanager
    def client():yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=SimpleNamespace(get_kernel=lambda req:SimpleNamespace(metadata=SimpleNamespace(ref=v['kernel_ref'],current_version_number=5,is_private=True,id=133930602),blob=SimpleNamespace(source=json.dumps(notebook,indent=2))))))
    monkeypatch.setattr(kaggle_api_extended,'KaggleApi',lambda:SimpleNamespace(authenticate=lambda:None,build_kaggle_client=client))
    exec(k.identity_command(v)[2],{})
    actual=json.loads(capsys.readouterr().out);expected=hashlib.sha256(json.dumps(notebook,sort_keys=True).encode()).hexdigest()
    assert actual['code_sha256']==expected and actual['raw_platform_source_sha256']!=expected
    changed={**notebook,'cells':[{'source':'print(124)\n','cell_type':'code'}]}
    assert hashlib.sha256(json.dumps(changed,sort_keys=True).encode()).hexdigest()!=expected

def recovery_fixture(tmp_path):
    v=notebook_manifest();prior=tmp_path/'UNKNOWN.json';prior.write_text(json.dumps({'status':'UNKNOWN','job_id':v['job_id'],'execution_source_commit':v['source_commit']}))
    oldsha=hashlib.sha256(prior.read_bytes()).hexdigest();event={'event_id':'1'*64,'status':'CLAIMED','payload':{'round_id':v['job_id'],'bound_remote_job':True,'terminal_status':'UNKNOWN','terminal_receipt':str(prior),'terminal_receipt_sha256':oldsha}}
    i=tmp_path/'identity.json';i.write_text(json.dumps({**identity(v),'kernel_version':5,'kernel_id':133930602}))
    p=tmp_path/'status.json';p.write_text(json.dumps({'status':'KernelWorkerStatus.RUNNING'}))
    proof={'source_event_id':event['event_id'],'no_new_execution':True,'prior_unknown_sha256':oldsha,**{x:v[x] for x in ['job_id','nonce','source_commit','kernel_version']},'identity':{'path':str(i),'sha256':hashlib.sha256(i.read_bytes()).hexdigest()},'status':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}
    return v,event,proof

def test_monitor_recovery_is_same_claimed_UNKNOWN_job_only(tmp_path):
    v,e,p=recovery_fixture(tmp_path);assert k.validate_monitor_recovery(v,p,e)['job_id']==v['job_id']
    for key,value in [('no_new_execution',False),('nonce','f'*32),('source_event_id','2'*64),('kernel_version',6)]:
        with pytest.raises(AssertionError):k.validate_monitor_recovery(v,{**p,key:value},e)
    with pytest.raises(AssertionError):k.validate_monitor_recovery(v,p,{**e,'status':'COMPLETE'})
    pathlib.Path(p['identity']['path']).write_text('{}')
    with pytest.raises(AssertionError):k.validate_monitor_recovery(v,p,e)

def test_existing_notebook_opaque_ERROR_without_worker_is_not_terminal(tmp_path):
    v=notebook_manifest()
    def runner(cmd,**kw):return SimpleNamespace(returncode=0,stdout='job has status "ERROR"' if 'status' in cmd else json.dumps({**identity(v),'kernel_version':5,'kernel_id':133930602}) if '-c' in cmd else '')
    assert k.reconcile(v,tmp_path,runner) is None
    assert k.reconcile(v,tmp_path,runner,datetime.now(timezone.utc)+timedelta(days=1))['status']=='UNKNOWN'

def test_monitor_resume_preserves_UNKNOWN_and_emits_one_new_terminal(tmp_path,monkeypatch):
    v,e,p=recovery_fixture(tmp_path);registry=tmp_path/'registry';registry.mkdir();(registry/'job.json').write_text(json.dumps(v))
    proof=tmp_path/'proof.json';proof.write_text(json.dumps(p));control=k.validate_monitor_recovery(v,p,e);control.update(proof_path=str(proof),proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),prior_unknown_path=e['payload']['terminal_receipt'])
    (registry/'monitor_recovery').mkdir();(registry/'monitor_recovery'/(v['job_id']+'.json')).write_text(json.dumps(control))
    cache=tmp_path/'cache';old=cache/'remote_jobs'/v['job_id']/'TERMINAL_RECEIPT.json';old.parent.mkdir(parents=True);old.write_bytes(pathlib.Path(e['payload']['terminal_receipt']).read_bytes())
    monitor=s.Supervisor(tmp_path/'state.json',tmp_path/'notifications',tmp_path/'events');monitor.state['rounds'][v['job_id']]={'terminal_receipt_hash':e['payload']['terminal_receipt_sha256'],'terminal_status':'UNKNOWN','controller_notification_sent':True}
    monkeypatch.setattr(k,'reconcile',lambda *a:None)
    assert s.fetch_remote_job_receipts(registry,cache)==[]
    monkeypatch.setattr(k,'reconcile',lambda *a:{**worker(v),'receipt_origin':'LOCAL_SUPERVISOR_KAGGLE_OBSERVATION_V1'})
    records=s.fetch_remote_job_receipts(registry,cache);assert len(records)==1
    assert monitor.reconcile_bound_remote_jobs(records)==[v['job_id']]
    assert monitor.reconcile_bound_remote_jobs(records)==[]
    assert hashlib.sha256(old.read_bytes()).hexdigest()==e['payload']['terminal_receipt_sha256']
    assert len(list((tmp_path/'events').glob('*.json')))==1
    assert monitor.state['rounds'][v['job_id']]['observation_history'][0]['terminal_status']=='UNKNOWN'

def test_governor_same_event_WAIT_REMOTE_recovery_no_new_task(tmp_path):
    from test_arc2_governor_thin_router import governor as g,initial_state,args_for_completion
    v,e,p=recovery_fixture(tmp_path);e.update(kind='REMOTE_TERMINAL',recipient='controller',producer='supervisor')
    state=tmp_path/'workflow.json';g.atomic(state,initial_state(pending_events=[e],remote_jobs={v['job_id']:v},remote_job=v))
    job=tmp_path/'job.json';job.write_text(json.dumps(v));proof=tmp_path/'proof.json';proof.write_text(json.dumps(p))
    args=args_for_completion(e['event_id'],'WAIT_REMOTE',remote_job_file=str(job),remote_monitor_recovery=str(proof),remote_monitor_recovery_sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
    g.complete_event(state,args);saved=g.load(state)
    assert saved['pending_events'][0]['status']=='COMPLETE' and len(saved['pending_events'])==1
    assert saved['remote_jobs'][v['job_id']]==v
    assert (tmp_path/'remote_jobs/monitor_recovery'/(v['job_id']+'.json')).exists()

def test_notebook_same_role_callback_exact_version(tmp_path):
    v=notebook_manifest()
    def runner(cmd,**kw):
        if 'status' in cmd:return SimpleNamespace(returncode=0,stdout='job has status "KernelWorkerStatus.COMPLETE"')
        return SimpleNamespace(returncode=0,stdout=json.dumps({**identity(v),'kernel_version':5,'kernel_id':133930602}))
    out=tmp_path/'kernel_output';out.mkdir();(out/'TERMINAL_RECEIPT.json').write_text(json.dumps(worker(v)))
    r=k.reconcile(v,tmp_path,runner);assert r['status']=='COMPLETE' and r['kernel_version']==5
    p=tmp_path/'observed.json';p.write_text(json.dumps(r));monitor=s.Supervisor(tmp_path/'state.json',tmp_path/'notifications',tmp_path/'events')
    assert monitor.reconcile_bound_remote_jobs([{**v,'manifest_path':str(tmp_path/'manifest.json'),'local_receipt':str(p),'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}])==[v['job_id']]
    e=json.loads(next((tmp_path/'events').glob('*.json')).read_bytes());assert e['payload']['controller_role']=='selector'

def test_claimed_verified_kaggle_terminal_retires_only_stale_live_binding(tmp_path):
    spec=importlib.util.spec_from_file_location('governor_retire_test',ROOT/'scripts/arc2_governor.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    old=manifest();w=tmp_path/'worker.json';w.write_text(json.dumps({**worker(old),'status':'INFRA_FAILED'}))
    receipt={'job_id':old['job_id'],'execution_source_commit':old['source_commit'],'status':'INFRA_FAILED','receipt_origin':'LOCAL_SUPERVISOR_KAGGLE_OBSERVATION_V1','worker_terminal_local':str(w),'worker_terminal_sha256':hashlib.sha256(w.read_bytes()).hexdigest(),'platform_identity':identity(old)}
    p=tmp_path/'receipt.json';p.write_text(json.dumps(receipt))
    item={'status':'CLAIMED','recipient':'controller','payload':{'controller_role':'selector','round_id':old['job_id'],'bound_remote_job':True,'terminal_status':'INFRA_FAILED','terminal_receipt':str(p),'terminal_receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}
    state={'agent_routes':{'selector':{}},'remote_jobs':{old['job_id']:old},'remote_job':old}
    g.retire_claimed_kaggle_terminal(state,item);assert state['remote_jobs']=={} and state['remote_job'] is None and item['verified_terminal_binding_retired']['job_id']==old['job_id']
    state['remote_jobs']={old['job_id']:old};w.write_text(json.dumps({**worker(old),'status':'COMPLETE'}))
    with pytest.raises(RuntimeError,match='SHA256_MISMATCH'):g.retire_claimed_kaggle_terminal(state,item)
    assert old['job_id'] in state['remote_jobs']


def test_existing_bound_terminal_skips_download_despite_missing_supplement(tmp_path):
    v=manifest();out=tmp_path/'kernel_output';out.mkdir()
    w={**worker(v),'artifacts':{'PREFLIGHT_DEPENDENCY_LOG.txt':{'sha256':'e'*64,'bytes':8}}}
    (out/'TERMINAL_RECEIPT.json').write_text(json.dumps(w))
    calls=[]
    def runner(cmd,**kw):
        calls.append(cmd)
        if 'status' in cmd:return SimpleNamespace(returncode=0,stdout='has status "KernelWorkerStatus.COMPLETE"')
        if '-c' in cmd:return SimpleNamespace(returncode=0,stdout=json.dumps(identity(v)))
        raise AssertionError('A_VALID_CACHED_TERMINAL_MUST_NOT_REDOWNLOAD')
    result=k.reconcile(v,tmp_path,runner)
    assert result['status']=='COMPLETE' and len(calls)==2
    assert result['artifact_recovery']['missing']==['PREFLIGHT_DEPENDENCY_LOG.txt']
    assert result['scientific_acceptance']=='NOT_INFERRED_FROM_PROCESS_RECEIPT'


def test_download_timeout_after_receipt_written_still_routes(tmp_path):
    import subprocess
    v=manifest()
    def runner(cmd,**kw):
        if 'status' in cmd:return SimpleNamespace(returncode=0,stdout='has status "COMPLETE"')
        if '-c' in cmd:return SimpleNamespace(returncode=0,stdout=json.dumps(identity(v)))
        out=pathlib.Path(cmd[cmd.index('--output')+1]);(out/'TERMINAL_RECEIPT.json').write_text(json.dumps(worker(v)))
        raise subprocess.TimeoutExpired('receipt-fetch',kw['timeout'])
    assert k.reconcile(v,tmp_path,runner)['status']=='COMPLETE'


def test_bad_cached_nonce_never_refetched_or_accepted(tmp_path):
    v=manifest();out=tmp_path/'kernel_output';out.mkdir();(out/'TERMINAL_RECEIPT.json').write_text(json.dumps({**worker(v),'nonce':'f'*32}))
    def runner(cmd,**kw):
        if 'status' in cmd:return SimpleNamespace(returncode=0,stdout='has status "COMPLETE"')
        assert '-c' in cmd
        return SimpleNamespace(returncode=0,stdout=json.dumps(identity(v)))
    r=k.reconcile(v,tmp_path,runner);assert r['status']=='UNKNOWN' and r['reason'].startswith('AssertionError')
    assert (tmp_path/'OBSERVATION_ERROR.json').is_file()


def test_cached_corrupt_artifact_is_not_scientific_success(tmp_path):
    v=manifest();out=tmp_path/'kernel_output';out.mkdir();(out/'PREFLIGHT.json').write_text('{}')
    w={**worker(v),'artifacts':{'PREFLIGHT.json':{'sha256':'e'*64,'bytes':2}}};(out/'TERMINAL_RECEIPT.json').write_text(json.dumps(w))
    def runner(cmd,**kw):return SimpleNamespace(returncode=0,stdout='has status "COMPLETE"' if 'status' in cmd else json.dumps(identity(v)))
    r=k.reconcile(v,tmp_path,runner)
    assert r['artifact_recovery']['hash_mismatch']==['PREFLIGHT.json']
    assert r['scientific_acceptance']=='NOT_INFERRED_FROM_PROCESS_RECEIPT'


def test_named_fetch_stops_before_dependency_pages(tmp_path):
    import kaggle_receipt_fetch as fetch
    v=manifest();name='arc2_paired48_'+v['nonce']+'/TERMINAL_RECEIPT.json';calls=[]
    class Client:
        def __enter__(self):return SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=self))
        def __exit__(self,*args):pass
        def list_kernel_session_output(self,req):
            calls.append(req)
            assert len(calls)==1
            return SimpleNamespace(files=[SimpleNamespace(file_name=name,url='not-logged')],next_page_token='many-dependencies')
    class API:
        def build_kaggle_client(self):return Client()
        def _set_paging(self,r,size,token):r.page_size=size;r.page_token=token
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):yield json.dumps(worker(v)).encode()
    result=fetch.fetch_named(API(),v['kernel_ref'],v['nonce'],tmp_path,['TERMINAL_RECEIPT.json'],SimpleNamespace,lambda *a,**kw:Response())
    assert result['pages']==1 and result['missing']==[] and len(calls)==1


def test_named_fetch_rejects_path_escape(tmp_path):
    import kaggle_receipt_fetch as fetch
    v=manifest()
    with pytest.raises(AssertionError):fetch.fetch_named(None,v['kernel_ref'],v['nonce'],tmp_path,['../secret.json'],None,None)
