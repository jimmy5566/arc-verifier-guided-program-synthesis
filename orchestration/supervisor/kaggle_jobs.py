"""Typed private Kaggle job reconciliation; metadata/output only, never launch."""
from __future__ import annotations
import hashlib,json,pathlib,re,subprocess,sys
from datetime import datetime,timezone

CLI='C:/Users/asus/AppData/Local/Programs/Python/Python312/Scripts/kaggle.exe'
PYTHON='C:/Users/asus/AppData/Local/Programs/Python/Python312/python.exe'

def validate(v):
    assert v.get('backend')=='kaggle','KAGGLE_BACKEND_REQUIRED'
    assert not ({'ssh_target','primary_process'}&v.keys()),'NO_FAKE_SSH_OR_PID'
    assert re.fullmatch(r'[A-Za-z0-9_-]+',v['job_id'])
    if v['kernel_ref']=='jimmy5566/arc2-l4x4-runtime-template':
        assert v['kernel_id']==133930602 and v['kernel_type']=='notebook'
        assert type(v['prior_kernel_version']) is int
        if v.get('notebook_policy_sha256'):
            assert v['notebook_policy_sha256']=='130f06795eb8b763f77a169aa7c443b8771aaebab685f93f3af34517d31fbac9'
            assert not any(v.get(k) for k in ['allocation_id','standing_policy_sha256','execution_policy','startup_only'])
            assert v['prior_kernel_version']>=10 and v['kernel_version']==v['prior_kernel_version']+1
            assert re.fullmatch(r'[0-9a-f]{64}',v['prior_kernel_code_sha256'])
            assert v['platform_session_timeout_seconds']==7200 and v['parent_python_wall_cap_seconds']==6900
            assert v['initialization_wall_cap_seconds']==600 and v['cleanup_reserve_wall_seconds']==180
            assert v['allocated_device_count']==4 and v['cumulative_budget_enforced'] is False
            assert v['completion_context']['notebook_policy_sha256']==v['notebook_policy_sha256']
            assert v['completion_context']['durable_payload_byte_verification_required'] is True
            if v.get('launch_registration_utc'):
                registered=datetime.fromisoformat(v['launch_registration_utc'])
                visible=datetime.fromisoformat(v['launch_visibility_deadline_utc'])
                assert registered.tzinfo is not None and 0<(visible-registered).total_seconds()<=180
        elif v.get('startup_only'):
            assert v['startup_only'] is True
            if v['startup_authority_sha256']=='9a5af87d81fe672769b87e154e1e00e983166bf32531563877b77c8daf3e2ef7':
                assert v['execution_purpose']=='PASSIVE_EARLIEST_ENTRY_INITIALIZATION_ONLY_V2'
                assert v['standing_policy_sha256']==v['startup_authority_sha256']
                assert v['entry_telemetry_contract']=='WORKER_ENTRY_PRE_VALIDATION_V1'
                assert v['completion_context']['standing_policy_sha256']==v['standing_policy_sha256']
                assert v['completion_context']['entry_telemetry_contract']==v['entry_telemetry_contract']
                assert v['completion_context']['standing_policy_path']
            else:
                assert v['execution_purpose']=='PASSIVE_INITIALIZATION_ONLY_V1'
                assert v['startup_authority_sha256']=='c8785e6536643a782ea08b0fd1868740371bda5005231a4e8f4728da3b7b5ba4'
                assert not v.get('standing_policy_sha256')
            assert not v.get('allocation_id') and not v.get('execution_policy')
            assert v['prior_kernel_version']>=7 and v['kernel_version']==v['prior_kernel_version']+1
            assert re.fullmatch(r'[0-9a-f]{64}',v['prior_kernel_code_sha256'])
            assert v['engineering_cap_device_seconds']==1800 and v['allocated_device_count']==4
            assert v['platform_session_timeout_seconds']==450 and v['parent_python_wall_cap_seconds']==300
            assert v['initialization_wall_cap_seconds']==240 and v['cleanup_reserve_wall_seconds']==30
            assert v['fit_calls_authorized']==v['generation_calls_authorized']==0
            assert v['completion_context']['phase']==v['execution_purpose']
            assert v['completion_context']['startup_authority_sha256']==v['startup_authority_sha256']
            assert v['completion_context']['engineering_reservation_device_seconds']==1800
            assert v['engineering_ledger_path'] and v['completion_context']['engineering_ledger_path']
        elif v.get('allocation_id'):
            assert v['allocation_id']=='PAIRED48_OWNER_EXPLICIT_RESET_V1'
            assert v['owner_time_reset_sha256']=='1fd2ed429029e91fc24970bf28d7e1d5dc1a68b2ec3681b964327a75893389a3'
            assert v['prior_kernel_version']>=5 and v['kernel_version']==v['prior_kernel_version']+1
            assert re.fullmatch(r'[0-9a-f]{64}',v['prior_kernel_code_sha256'])
            assert v['allocation_cap_device_seconds']==7200 and 0<v['allocation_remaining_device_seconds']<=7200
            assert v['completion_context']['allocation_id']==v['allocation_id'] and v['completion_context']['allocation_ledger_path']
        elif (v['prior_kernel_version'],v['kernel_version'])==(4,5):
            assert v['owner_amendment_sha256']=='3375fe13721877fb755248d513b536047348000f580b458698afad60548042a8'
            assert v['prior_kernel_code_sha256']=='95a0642d616f01f18e7b57875373cfcdce7d3317bdd89482cd30038673d51a30'
        else:
            assert (v['prior_kernel_version'],v['kernel_version'])==(5,6)
            assert v['owner_directive_sha256']=='0a34754f556145f1d433519887fab63302fdd0ce85cc9f076ea2d8960c6beeda'
            assert v['prior_kernel_code_sha256']=='06dbaa6042626417b85b1caa0c8c5f71b53e654232fb2c5fc43b4e1d1b96747d'
        assert v['unique_one_push_slug'] is False and v['single_version_push'] is True
    else:
        assert re.fullmatch(r'jimmy5566/arc2-paired48-[a-z0-9-]+',v['kernel_ref']),'RESEARCH_SLUG_ONLY'
        assert v['kernel_version']==1 and v['unique_one_push_slug'] is True
        if v.get('kernel_type'):
            assert v['kernel_type']=='notebook'
            assert v['source_hash_contract']=='NOTEBOOK_JSON_SORT_KEYS_V1'
    assert v['controller_role']=='selector' and v.get('completion_recipient')=='controller'
    assert v['is_private'] is True and v['kaggle_cli']==CLI
    assert re.fullmatch(r'[0-9a-f]{32}',v['nonce']) and re.fullmatch(r'[0-9a-f]{40}',v['source_commit'])
    for key in ['kernel_code_sha256','source_manifest_sha256']:assert re.fullmatch(r'[0-9a-f]{64}',v[key])
    assert v['output_root']=='/kaggle/working/arc2_paired48_'+v['nonce']
    assert v['terminal_receipt']==v['output_root']+'/TERMINAL_RECEIPT.json'
    assert datetime.fromisoformat(v['deadline_utc']).tzinfo is not None
    if v.get('execution_policy'):
        assert v['execution_policy']=='CHECKPOINTED_PROGRESS_UNTIL_HARD_CAP_V1'
        assert v['owner_execution_policy_sha256']=='02bf3f2e7fa562ced8bbae82d2eb33f2b0989c207650f75430d9a285fc0c4eac'
        assert v['completion_context']['execution_policy']==v['execution_policy']
        assert v['completion_context']['owner_execution_policy_sha256']==v['owner_execution_policy_sha256']
    if v.get('standing_policy_sha256'):
        assert v['standing_policy_sha256']=='9a5af87d81fe672769b87e154e1e00e983166bf32531563877b77c8daf3e2ef7'
        assert v['entry_telemetry_contract']=='WORKER_ENTRY_PRE_VALIDATION_V1'
        assert v['completion_context']['standing_policy_sha256']==v['standing_policy_sha256']
        assert v['completion_context']['entry_telemetry_contract']==v['entry_telemetry_contract']
        if not v.get('startup_only'):
            assert v['standing_scope']=='MISSING42_SCIENTIFIC_CONTINUATION_V1'
            assert v['completion_context']['standing_scope']==v['standing_scope']
            assert v.get('execution_policy')=='CHECKPOINTED_PROGRESS_UNTIL_HARD_CAP_V1'
            assert 0<v['allocation_remaining_device_seconds']<=4794.056
            assert re.fullmatch(r'[0-9a-f]{64}',v['prior_startup_readiness_sha256'])
            assert v['completion_context']['prior_startup_readiness_sha256']==v['prior_startup_readiness_sha256']
    if v.get('platform_kernel_ref'):
        assert re.fullmatch(r'jimmy5566/arc2-paired48-[a-z0-9-]+',v['platform_kernel_ref'])
        assert re.fullmatch(r'[0-9a-f]{64}',v['platform_alias_receipt']['sha256'])
    return v

def identity_command(v):
    # SDK handles existing OAuth internally; no credential values are accessed or emitted.
    code="""import hashlib,json
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
a=KaggleApi();a.authenticate();r=ApiGetKernelRequest();r.user_name='jimmy5566';r.kernel_slug=SLUG
with a.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_kernel(r)
payload={'kernel_ref':q.metadata.ref,'kernel_version':q.metadata.current_version_number,'is_private':q.metadata.is_private,'code_sha256':hashlib.sha256(q.blob.source.encode()).hexdigest()}
bound_id=BOUND_ID
if bound_id is not None:
 payload['kernel_id']=int(q.metadata.id)
if bound_id is not None or NOTEBOOK:
 payload['raw_platform_source_sha256']=payload['code_sha256']
 # The API reformats notebook JSON. Restore the exact uploaded CLI JSON
 # serialization; all cell strings, outputs, metadata and format stay bound.
 notebook=json.loads(q.blob.source)
 payload['code_sha256']=hashlib.sha256(json.dumps(notebook,sort_keys=True).encode()).hexdigest()
print(json.dumps(payload))
""".replace('SLUG',repr(v.get('platform_kernel_ref',v['kernel_ref']).split('/')[1])).replace('BOUND_ID',repr(v.get('kernel_id'))).replace('NOTEBOOK',repr(v.get('kernel_type')=='notebook'))
    return [PYTHON,'-c',code]

def bound_terminal(v,worker,identity):
    assert identity['kernel_ref']==v.get('platform_kernel_ref',v['kernel_ref']) and identity['kernel_version']==v['kernel_version'] and identity['is_private'] is True,'PLATFORM_VERSION_OR_PRIVACY_MISMATCH'
    assert identity['code_sha256']==v['kernel_code_sha256'],'PLATFORM_CODE_MISMATCH'
    if v.get('kernel_id'):assert identity['kernel_id']==v['kernel_id'],'PLATFORM_NOTEBOOK_ID_MISMATCH'
    for k in ['job_id','kernel_ref','kernel_version','nonce','source_commit','source_manifest_sha256']:
        assert worker[k]==v[k],'WORKER_IDENTITY_MISMATCH:'+k
    if v.get('notebook_policy_sha256'):
        assert worker['notebook_policy_sha256']==v['notebook_policy_sha256'],'WORKER_NOTEBOOK_POLICY_MISMATCH'
    if v.get('allocation_id'):
        for k in ['allocation_id','owner_time_reset_sha256']:assert worker[k]==v[k],'WORKER_ALLOCATION_MISMATCH:'+k
    if v.get('execution_policy'):
        for k in ['execution_policy','owner_execution_policy_sha256']:assert worker[k]==v[k],'WORKER_EXECUTION_POLICY_MISMATCH:'+k
    if v.get('startup_only'):
        for k in ['execution_purpose','startup_authority_sha256']:assert worker[k]==v[k],'WORKER_STARTUP_PHASE_MISMATCH:'+k
        assert worker['engineering_cap_device_seconds']==1800
        assert worker['fit_calls_authorized']==worker['generation_calls_authorized']==0
        if v.get('standing_policy_sha256'):
            for k in ['standing_policy_sha256','entry_telemetry_contract']:assert worker[k]==v[k],'WORKER_STANDING_AUTHORITY_MISMATCH:'+k
    elif v.get('standing_policy_sha256'):
        for k in ['standing_policy_sha256','entry_telemetry_contract','standing_scope']:assert worker[k]==v[k],'WORKER_STANDING_SCIENCE_MISMATCH:'+k
    assert worker['status'] in ['COMPLETE','INFRA_FAILED','CONTROLLED_STOP','FAILED_OR_PARTIAL_NO_RETRY','UNKNOWN']
    return worker

def verify_prelaunch_alias(previous,current):
    """Only canonical platform naming may change; uploaded code/run remain exact."""
    extra={'platform_kernel_ref','platform_alias_receipt'}
    assert not previous.keys() & extra,'ALIAS_ALREADY_BOUND'
    assert {k:v for k,v in current.items() if k not in extra}==previous,'NON_ALIAS_BINDING_CHANGE'
    proof=current['platform_alias_receipt'];raw=pathlib.Path(proof['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==proof['sha256']
    observed=json.loads(raw)
    assert observed['requested_kernel_ref']==previous['kernel_ref'] and observed['nonce']==previous['nonce'] and observed['job_id']==previous['job_id']
    expected={'kernel_ref':current['platform_kernel_ref'],'kernel_version':1,'is_private':True,'code_sha256':previous['kernel_code_sha256']}
    assert observed['platform_identity']==expected
    live=subprocess.run(identity_command(current),capture_output=True,text=True,timeout=60)
    assert live.returncode==0 and json.loads(live.stdout.strip().splitlines()[-1])==expected,'LIVE_ALIAS_IDENTITY_MISMATCH'

def validate_monitor_recovery(v, proof, item):
    """Resume observation only, after an UNKNOWN callback contradicted by RUNNING."""
    validate(v)
    assert v['kernel_ref']=='jimmy5566/arc2-l4x4-runtime-template'
    payload=item['payload']
    assert item['status']=='CLAIMED' and payload['bound_remote_job'] is True
    assert payload['round_id']==v['job_id'] and payload['terminal_status']=='UNKNOWN'
    assert proof['source_event_id']==item['event_id'] and proof['no_new_execution'] is True
    assert proof['prior_unknown_sha256']==payload['terminal_receipt_sha256']
    prior=pathlib.Path(payload['terminal_receipt']).read_bytes()
    assert hashlib.sha256(prior).hexdigest()==proof['prior_unknown_sha256']
    old=json.loads(prior)
    assert old['status']=='UNKNOWN' and old['job_id']==v['job_id'] and not old.get('worker_terminal_local')
    assert old['execution_source_commit']==v['source_commit']
    for key in ['job_id','nonce','source_commit','kernel_version']:assert proof[key]==v[key]
    observed={}
    for kind in ['identity','status']:
        bound=proof[kind];raw=pathlib.Path(bound['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==bound['sha256']
        observed[kind]=json.loads(raw)
    i=observed['identity']
    assert i['kernel_ref']==v['kernel_ref'] and i['kernel_id']==v['kernel_id'] and i['kernel_version']==v['kernel_version'] and i['is_private'] is True
    assert i['code_sha256']==v['kernel_code_sha256']
    assert observed['status']['status']=='KernelWorkerStatus.RUNNING'
    return {'job_id':v['job_id'],'nonce':v['nonce'],'source_event_id':item['event_id'],'prior_unknown_sha256':proof['prior_unknown_sha256'],'manifest_identity_sha256':hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()}

def monitor_recovery(manifest, registry):
    path=pathlib.Path(registry)/'monitor_recovery'/(manifest['job_id']+'.json')
    if not path.exists():return None
    control=json.loads(path.read_bytes());remote={k:x for k,x in manifest.items() if k!='manifest_path'}
    assert control['manifest_identity_sha256']==hashlib.sha256(json.dumps(remote,sort_keys=True).encode()).hexdigest()
    raw=pathlib.Path(control['proof_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==control['proof_sha256']
    proof=json.loads(raw)
    validate_monitor_recovery(remote,proof,{'event_id':control['source_event_id'],'status':'CLAIMED','payload':{'bound_remote_job':True,'round_id':remote['job_id'],'terminal_status':'UNKNOWN','terminal_receipt':control['prior_unknown_path'],'terminal_receipt_sha256':control['prior_unknown_sha256']}})
    return control


def download_command(v, output, names=('TERMINAL_RECEIPT.json',)):
    command = [PYTHON, str(pathlib.Path(__file__).with_name('kaggle_receipt_fetch.py')),
               '--kernel-ref', v.get('platform_kernel_ref', v['kernel_ref']),
               '--nonce', v['nonce'], '--output', str(output)]
    for name in names:
        command.extend(['--name', name])
    return command


def cached_worker(v, output, identity):
    matches = list(output.rglob('TERMINAL_RECEIPT.json'))
    if not matches:
        return None
    assert len(matches) == 1, 'AMBIGUOUS_WORKER_TERMINAL'
    raw = matches[0].read_bytes()
    worker = json.loads(raw)
    bound_terminal(v, worker, identity)
    # Completion transport is not scientific acceptance. Missing supplementary
    # artifacts are handed to Controller, not a reason to redownload all outputs.
    evidence = {'verified': [], 'missing': [], 'hash_mismatch': []}
    for name, entry in worker.get('artifacts', {}).items():
        rel = pathlib.PurePosixPath(name)
        assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in name, 'UNSAFE_ARTIFACT_PATH'
        path = matches[0].parent.joinpath(*rel.parts).resolve()
        assert path.is_relative_to(matches[0].parent.resolve()), 'UNSAFE_ARTIFACT_PATH'
        if not path.is_file():
            evidence['missing'].append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256'] or path.stat().st_size != entry['bytes']:
            evidence['hash_mismatch'].append(name)
        else:
            evidence['verified'].append(name)
    return {'job_id': v['job_id'], 'status': worker['status'], 'terminal_status': worker['status'],
            'receipt_origin': 'LOCAL_SUPERVISOR_KAGGLE_OBSERVATION_V1',
            'execution_source_commit': v['source_commit'], 'kernel_ref': v['kernel_ref'],
            'kernel_version': v['kernel_version'], 'nonce': v['nonce'],
            'worker_terminal_sha256': hashlib.sha256(raw).hexdigest(),
            'worker_terminal_local': str(matches[0].resolve()), 'platform_identity': identity,
            'artifact_recovery': evidence,
            'scientific_acceptance': 'NOT_INFERRED_FROM_PROCESS_RECEIPT',
            'automatic_rerun_authorized': False}

def reconcile(v,cache,runner=subprocess.run,now=None):
    """Return one terminal observation or None; called only by existing Supervisor."""
    validate(v);cache=pathlib.Path(cache);cache.mkdir(parents=True,exist_ok=True);now=now or datetime.now(timezone.utc)
    receipt=None;reason=None;reported=None;identity=None
    try:
        platform_ref=v.get('platform_kernel_ref',v['kernel_ref'])
        status=runner([CLI,'kernels','status',platform_ref+'/1'],capture_output=True,text=True,timeout=45)
        if status.returncode:raise RuntimeError('KAGGLE_STATUS_UNAVAILABLE')
        match=re.search(r'\bstatus\s+[\"\']?(?:KernelWorkerStatus\.)?([A-Za-z_]+)',status.stdout,re.I)
        if not match:raise RuntimeError('KAGGLE_STATUS_UNPARSEABLE')
        reported=match.group(1).upper()
        if reported in {'COMPLETE','ERROR','CANCELLED','CANCELED','FAILED'}:
            p=runner(identity_command(v),capture_output=True,text=True,timeout=60)
            if p.returncode:raise RuntimeError('KAGGLE_IDENTITY_UNAVAILABLE')
            identity=json.loads(p.stdout.strip().splitlines()[-1]);out=cache/'kernel_output';out.mkdir(exist_ok=True)
            if (v.get('notebook_policy_sha256') and v.get('launch_visibility_deadline_utc')
                    and identity.get('kernel_id')==v['kernel_id'] and identity.get('kernel_version')==v['prior_kernel_version']
                    and identity.get('code_sha256')==v['prior_kernel_code_sha256'] and identity.get('is_private') is True
                    and now<datetime.fromisoformat(v['launch_visibility_deadline_utc'])):
                # Registration must precede push. During that bounded window
                # the previous terminal version is expected, never this result.
                return None
            receipt = cached_worker(v, out, identity)
            if receipt is None:
                # Direct Python child has bounded lifetime; no Windows CLI
                # launcher/orphan and no recursive kernels-output pagination.
                try:
                    p=runner(download_command(v,out),capture_output=True,text=True,timeout=60)
                except subprocess.TimeoutExpired:
                    # A receipt may have arrived just before download timeout.
                    receipt = cached_worker(v,out,identity)
                    if receipt is None:
                        raise RuntimeError('KAGGLE_NAMED_RECEIPT_FETCH_TIMEOUT')
                else:
                    receipt = cached_worker(v,out,identity)
                    if receipt is None:
                        raise RuntimeError('KAGGLE_OUTPUT_UNAVAILABLE' if p.returncode else 'MISSING_WORKER_TERMINAL')
            if receipt is not None:
                receipt['reported_platform_status'] = reported
        elif reported not in {'QUEUED','RUNNING','PENDING','INITIALIZING'}:
            reason='UNRECOGNIZED_PLATFORM_STATUS'
    except (Exception,) as error:
        reason=type(error).__name__+':'+('IDENTITY_OR_BOUND_TERMINAL_REJECTED' if isinstance(error,AssertionError) else 'PLATFORM_OR_OUTPUT_UNAVAILABLE')
        # Keep a small observable error without serializing exceptions/URLs or
        # treating cached old-job transports as a successful current-job check.
        observation = {'observed_at': now.isoformat(), 'job_id': v['job_id'],
                       'reported_platform_status': reported, 'reason': reason,
                       'error_type': type(error).__name__}
        temp = cache / 'OBSERVATION_ERROR.json.tmp'
        temp.write_text(json.dumps(observation,sort_keys=True)+'\n',encoding='utf8')
        temp.replace(cache / 'OBSERVATION_ERROR.json')
    # Existing-notebook latest-session status may still describe its previous
    # run while the newly pushed version queues. Without its nonce-bound worker
    # terminal, retain an observation error until the bound deadline.
    unverified_existing_notebook=(v.get('kernel_id')==133930602 and identity is not None and identity.get('kernel_version')==v['kernel_version'] and identity.get('code_sha256')==v['kernel_code_sha256'] and not (reason or '').startswith('AssertionError'))
    if receipt is None and (now>=datetime.fromisoformat(v['deadline_utc']) or ((reported in {'COMPLETE','ERROR','CANCELLED','CANCELED','FAILED'}) and not unverified_existing_notebook) or (reason and reason.startswith('AssertionError'))):
        receipt={'job_id':v['job_id'],'status':'UNKNOWN','terminal_status':'UNKNOWN','receipt_origin':'LOCAL_SUPERVISOR_KAGGLE_OBSERVATION_V1','execution_source_commit':v['source_commit'],'kernel_ref':v['kernel_ref'],'kernel_version':v['kernel_version'],'nonce':v['nonce'],'reported_platform_status':reported,'reason':reason or 'DEADLINE_WITHOUT_VERIFIED_TERMINAL','scientific_acceptance':'NOT_INFERRED','automatic_rerun_authorized':False}
    if receipt is not None:receipt['observed_at']=now.isoformat()
    return receipt
