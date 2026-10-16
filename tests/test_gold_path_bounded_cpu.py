import hashlib
import json
from types import SimpleNamespace
import sys

import pytest

from scripts import run_v7_eval60_gold_path_bounded as w
from scripts import run_v7_ttt24_eval60_gold_path_diagnostic as d


def test_forward_cap_counts_failed_attempts(monkeypatch,tmp_path):
    args=SimpleNamespace(output=tmp_path,max_new_view_forwards=1,context_window=10)
    def fail(*a,**kw): raise ValueError('simulated failure')
    monkeypatch.setattr(d,'score_gold_view',fail)
    with pytest.raises(ValueError): d.bounded_gold_forward(args,None,None,[],[])
    with pytest.raises(RuntimeError,match='CAP_EXHAUSTED'): d.bounded_gold_forward(args,None,None,[],[])
    assert json.loads((tmp_path/'FORWARD_BUDGET.json').read_text())['attempted_new_view_forwards']==1


@pytest.mark.parametrize('exit_code',[0,1])
def test_bounded_production_route_and_failure_receipt(monkeypatch,tmp_path,exit_code):
    auth={'status':'OWNER_AUTHORIZED','stage':'POSTFREEZE_GOLD_PATH_MISSING_ONLY',
          'authorization_id':'fixture','hard_wall_seconds':3600,'max_new_view_forwards':1428}
    path=tmp_path/'auth.json';path.write_text(json.dumps(auth))
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['wrapper','--authorization',str(path),'--authorization-sha256',
        hashlib.sha256(path.read_bytes()).hexdigest(),'--output',str(output),'--','--observed-cache','frozen'])
    monkeypatch.setattr(w.subprocess,'check_output',lambda *a,**kw:'a'*40)
    class Child:
        pid=100
        def __init__(self,command,**kw):
            assert 'scripts.run_v7_ttt24_eval60_gold_path_diagnostic' in command
            assert '--reuse-observed-search' in command and '1428' in command
            assert '--cpu-cache-preflight-only' not in command
            # CPU fixtures may deliberately mask CUDA; the launcher must not
            # silently change the caller's device policy.
            import os
            assert kw['env'].get('CUDA_VISIBLE_DEVICES')==os.environ.get('CUDA_VISIBLE_DEVICES')
            (output/'diagnostic').mkdir()
            (output/'diagnostic/SUMMARY.json').write_text(json.dumps({'status':'COMPLETE','output_count':89}))
        def wait(self,timeout):
            assert timeout==3560
            return exit_code
    monkeypatch.setattr(w.subprocess,'Popen',Child)
    assert w.main()==exit_code
    state=json.loads((output/'PIPELINE_STATE.json').read_text())
    assert state['status']==('COMPLETE' if exit_code==0 else 'FAILED')
    assert state['generation_rerun'] is False and state['training_rerun'] is False
    assert (tmp_path/'CONSUMED_fixture.json').is_file()


def test_forbidden_override():
    with pytest.raises(RuntimeError,match='INVALID_DIAGNOSTIC'):
        w.command_for(SimpleNamespace(worker_args=['--output','wrong']),{'max_new_view_forwards':1})
