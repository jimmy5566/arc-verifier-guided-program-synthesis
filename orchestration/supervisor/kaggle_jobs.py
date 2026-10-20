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
        assert v['owner_amendment_sha256']=='3375fe13721877fb755248d513b536047348000f580b458698afad60548042a8'
        assert type(v['prior_kernel_version']) is int and v['prior_kernel_version']==4
        assert v['kernel_version']==v['prior_kernel_version']+1
        assert v['unique_one_push_slug'] is False and v['single_version_push'] is True
        assert v['prior_kernel_code_sha256']=='95a0642d616f01f18e7b57875373cfcdce7d3317bdd89482cd30038673d51a30'
    else:
        assert re.fullmatch(r'jimmy5566/arc2-paired48-[a-z0-9-]+',v['kernel_ref']),'RESEARCH_SLUG_ONLY'
        assert v['kernel_version']==1 and v['unique_one_push_slug'] is True
    assert v['controller_role']=='selector' and v.get('completion_recipient')=='controller'
    assert v['is_private'] is True and v['kaggle_cli']==CLI
    assert re.fullmatch(r'[0-9a-f]{32}',v['nonce']) and re.fullmatch(r'[0-9a-f]{40}',v['source_commit'])
    for key in ['kernel_code_sha256','source_manifest_sha256']:assert re.fullmatch(r'[0-9a-f]{64}',v[key])
    assert v['output_root']=='/kaggle/working/arc2_paired48_'+v['nonce']
    assert v['terminal_receipt']==v['output_root']+'/TERMINAL_RECEIPT.json'
    assert datetime.fromisoformat(v['deadline_utc']).tzinfo is not None
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
 payload['raw_platform_source_sha256']=payload['code_sha256']
 # The API reformats notebook JSON. Restore the exact uploaded CLI JSON
 # serialization; all cell strings, outputs, metadata and format stay bound.
 notebook=json.loads(q.blob.source)
 payload['code_sha256']=hashlib.sha256(json.dumps(notebook,sort_keys=True).encode()).hexdigest()
print(json.dumps(payload))
""".replace('SLUG',repr(v.get('platform_kernel_ref',v['kernel_ref']).split('/')[1])).replace('BOUND_ID',repr(v.get('kernel_id')))
    return [PYTHON,'-c',code]

def bound_terminal(v,worker,identity):
    assert identity['kernel_ref']==v.get('platform_kernel_ref',v['kernel_ref']) and identity['kernel_version']==v['kernel_version'] and identity['is_private'] is True,'PLATFORM_VERSION_OR_PRIVACY_MISMATCH'
    assert identity['code_sha256']==v['kernel_code_sha256'],'PLATFORM_CODE_MISMATCH'
    if v.get('kernel_id'):assert identity['kernel_id']==v['kernel_id'],'PLATFORM_NOTEBOOK_ID_MISMATCH'
    for k in ['job_id','kernel_ref','kernel_version','nonce','source_commit','source_manifest_sha256']:
        assert worker[k]==v[k],'WORKER_IDENTITY_MISMATCH:'+k
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
    if receipt is None and (now>=datetime.fromisoformat(v['deadline_utc']) or (reported in {'COMPLETE','ERROR','CANCELLED','CANCELED','FAILED'}) or (reason and reason.startswith('AssertionError'))):
        receipt={'job_id':v['job_id'],'status':'UNKNOWN','terminal_status':'UNKNOWN','receipt_origin':'LOCAL_SUPERVISOR_KAGGLE_OBSERVATION_V1','execution_source_commit':v['source_commit'],'kernel_ref':v['kernel_ref'],'kernel_version':v['kernel_version'],'nonce':v['nonce'],'reported_platform_status':reported,'reason':reason or 'DEADLINE_WITHOUT_VERIFIED_TERMINAL','scientific_acceptance':'NOT_INFERRED','automatic_rerun_authorized':False}
    if receipt is not None:receipt['observed_at']=now.isoformat()
    return receipt
