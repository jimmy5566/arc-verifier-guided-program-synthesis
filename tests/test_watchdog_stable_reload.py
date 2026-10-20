"""Exercise the real PowerShell loop with process doubles; no daemon starts."""
import json,pathlib,subprocess

SCRIPT=pathlib.Path(__file__).resolve().parents[1]/'orchestration/watchdog/arc2_watchdog.ps1'

def run_fixture(tmp_path,stable,clean_after):
    state=tmp_path/'state.json'
    code=r'''
$global:arc2_fixture_starts=0
function Start-Process {
 param($FilePath,$ArgumentList,$WindowStyle,[switch]$PassThru)
 $global:arc2_fixture_starts+=1
 $rc=1
 if ($global:arc2_fixture_starts -gt CLEAN_AFTER) { $rc=0 }
 $p=[pscustomobject]@{ Id=(1000+$global:arc2_fixture_starts); ExitCode=$rc }
 $p | Add-Member -MemberType ScriptMethod -Name WaitForExit -Value {}
 return $p
}
function Start-Sleep { param($Seconds) }
& 'SCRIPT' -SupervisorFilePath 'CPU_FIXTURE_ONLY' -StatePath 'STATE' -StableSeconds STABLE -MaxRestarts 3
exit $LASTEXITCODE
'''.replace('CLEAN_AFTER',str(clean_after)).replace('SCRIPT',SCRIPT.as_posix()).replace('STATE',state.as_posix()).replace('STABLE',str(stable))
    process=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-Command',code],capture_output=True,text=True,timeout=15)
    assert state.exists(),process.stderr
    return process,json.loads(state.read_text(encoding='utf-8-sig'))

def test_five_stable_reloads_do_not_exhaust_startup_failure_guard(tmp_path):
    process,state=run_fixture(tmp_path,0,5)
    assert process.returncode==0 and state['status']=='STOPPED_CLEANLY'
    assert state['restart_count']==0 and state['total_restart_count']==5

def test_consecutive_fast_failures_still_stop_after_four_starts(tmp_path):
    process,state=run_fixture(tmp_path,60,99)
    assert process.returncode==1 and state['status']=='PERSISTENT_SUPERVISOR_STARTUP_FAILURE'
    assert state['persistent_failure'] and state['restart_count']==4 and state['total_restart_count']==4
