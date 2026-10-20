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
    assert re.fullmatch(r'jimmy5566/arc2-paired48-[a-z0-9-]+',v['kernel_ref']),'RESEARCH_SLUG_ONLY'
    assert v['kernel_version']==1 and v['unique_one_push_slug'] is True
    assert v['controller_role']=='selector' and v.get('completion_recipient')=='controller'
    assert v['is_private'] is True and v['kaggle_cli']==CLI
    assert re.fullmatch(r'[0-9a-f]{32}',v['nonce']) and re.fullmatch(r'[0-9a-f]{40}',v['source_commit'])
    for key in ['kernel_code_sha256','source_manifest_sha256']:assert re.fullmatch(r'[0-9a-f]{64}',v[key])
    assert v['output_root']=='/kaggle/working/arc2_paired48_'+v['nonce']
    assert v['terminal_receipt']==v['output_root']+'/TERMINAL_RECEIPT.json'
    assert datetime.fromisoformat(v['deadline_utc']).tzinfo is not None
    return v

def identity_command(v):
    # SDK handles existing OAuth internally; no credential values are accessed or emitted.
    code="""import hashlib,json
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
a=KaggleApi();a.authenticate();r=ApiGetKernelRequest();r.user_name='jimmy5566';r.kernel_slug=SLUG
with a.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_kernel(r)
print(json.dumps({'kernel_ref':q.metadata.ref,'kernel_version':q.metadata.current_version_number,'is_private':q.metadata.is_private,'code_sha256':hashlib.sha256(q.blob.source.encode()).hexdigest()}))
""".replace('SLUG',repr(v['kernel_ref'].split('/')[1]))
    return [PYTHON,'-c',code]

def bound_terminal(v,worker,identity):
    assert identity['kernel_ref']==v['kernel_ref'] and identity['kernel_version']==v['kernel_version'] and identity['is_private'] is True,'PLATFORM_VERSION_OR_PRIVACY_MISMATCH'
    assert identity['code_sha256']==v['kernel_code_sha256'],'PLATFORM_CODE_MISMATCH'
    for k in ['job_id','kernel_ref','kernel_version','nonce','source_commit','source_manifest_sha256']:
        assert worker[k]==v[k],'WORKER_IDENTITY_MISMATCH:'+k
    assert worker['status'] in ['COMPLETE','INFRA_FAILED','CONTROLLED_STOP','FAILED_OR_PARTIAL_NO_RETRY','UNKNOWN']
    return worker

def reconcile(v,cache,runner=subprocess.run,now=None):
    """Return one terminal observation or None; called only by existing Supervisor."""
    validate(v);cache=pathlib.Path(cache);cache.mkdir(parents=True,exist_ok=True);now=now or datetime.now(timezone.utc)
    receipt=None;reason=None;reported=None;identity=None
    try:
        status=runner([CLI,'kernels','status',v['kernel_ref']+'/1'],capture_output=True,text=True,timeout=45)
        if status.returncode:raise RuntimeError('KAGGLE_STATUS_UNAVAILABLE')
        match=re.search(r'\bstatus\s+[\"\']?(?:KernelWorkerStatus\.)?([A-Za-z_]+)',status.stdout,re.I)
        if not match:raise RuntimeError('KAGGLE_STATUS_UNPARSEABLE')
        reported=match.group(1).upper()
        if reported in {'COMPLETE','ERROR','CANCELLED','CANCELED','FAILED'}:
            p=runner(identity_command(v),capture_output=True,text=True,timeout=60)
            if p.returncode:raise RuntimeError('KAGGLE_IDENTITY_UNAVAILABLE')
            identity=json.loads(p.stdout.strip().splitlines()[-1]);out=cache/'kernel_output';out.mkdir(exist_ok=True)
            p=runner([CLI,'kernels','output',v['kernel_ref']+'/1','-p',str(out),'--file-pattern',r'(TERMINAL_RECEIPT|PREFLIGHT|SOURCE_MANIFEST|DEVICE_RECEIPT|RAW_FREEZE|COST|ERROR).*\.json$','-q'],capture_output=True,text=True,timeout=90)
            if p.returncode:raise RuntimeError('KAGGLE_OUTPUT_UNAVAILABLE')
            matches=list(out.rglob('TERMINAL_RECEIPT.json'))
            if len(matches)!=1:raise RuntimeError('MISSING_OR_AMBIGUOUS_WORKER_TERMINAL')
            raw=matches[0].read_bytes();worker=json.loads(raw);bound_terminal(v,worker,identity)
            receipt={'job_id':v['job_id'],'status':worker['status'],'terminal_status':worker['status'],'receipt_origin':'LOCAL_SUPERVISOR_KAGGLE_OBSERVATION_V1','execution_source_commit':v['source_commit'],'kernel_ref':v['kernel_ref'],'kernel_version':v['kernel_version'],'nonce':v['nonce'],'worker_terminal_sha256':hashlib.sha256(raw).hexdigest(),'worker_terminal_local':str(matches[0].resolve()),'platform_identity':identity,'reported_platform_status':reported,'scientific_acceptance':'NOT_INFERRED_FROM_PROCESS_RECEIPT','automatic_rerun_authorized':False}
        elif reported not in {'QUEUED','RUNNING','PENDING','INITIALIZING'}:
            reason='UNRECOGNIZED_PLATFORM_STATUS'
    except (Exception,) as error:
        reason=type(error).__name__+':'+('IDENTITY_OR_BOUND_TERMINAL_REJECTED' if isinstance(error,AssertionError) else 'PLATFORM_OR_OUTPUT_UNAVAILABLE')
    if receipt is None and (now>=datetime.fromisoformat(v['deadline_utc']) or (reported in {'ERROR','CANCELLED','CANCELED','FAILED'}) or (reason and reason.startswith('AssertionError'))):
        receipt={'job_id':v['job_id'],'status':'UNKNOWN','terminal_status':'UNKNOWN','receipt_origin':'LOCAL_SUPERVISOR_KAGGLE_OBSERVATION_V1','execution_source_commit':v['source_commit'],'kernel_ref':v['kernel_ref'],'kernel_version':v['kernel_version'],'nonce':v['nonce'],'reported_platform_status':reported,'reason':reason or 'DEADLINE_WITHOUT_VERIFIED_TERMINAL','scientific_acceptance':'NOT_INFERRED','automatic_rerun_authorized':False}
    if receipt is not None:receipt['observed_at']=now.isoformat()
    return receipt
