from __future__ import annotations
import os, shutil, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'scripts/arc2_import_runpod_pid1_secret.sh'

def bash():
    return shutil.which('bash') or r'C:/Program Files/Git/bin/bash.exe'

def test_pid1_secret_import_uses_fake_environ_without_stdout_or_persistence():
    with tempfile.TemporaryDirectory() as d:
        envfile=Path(d)/'environ'; envfile.write_bytes(b'X=1\0HF_TOKEN=synthetic-test-only\0GH_TOKEN=synthetic-gh-only\0')
        cmd=f'set -eu; source "{HELPER}"; arc2_import_runpod_hf_token; test "$HF_TOKEN" = synthetic-test-only; test -z "${{GH_TOKEN:-}}"; echo PASS'
        env={'PATH':os.environ.get('PATH',''),'ARC2_PID1_ENV_PATH':str(envfile)}
        out=subprocess.run([bash(),'-lc',cmd],env=env,text=True,capture_output=True,check=True)
        assert out.stdout=='PASS\n' and 'synthetic-test-only' not in out.stdout+out.stderr

def test_pid1_secret_import_rejects_missing_secret_and_unknown_name():
    with tempfile.TemporaryDirectory() as d:
        envfile=Path(d)/'environ'; envfile.write_bytes(b'X=1\0')
        cmd=f'set -eu; source "{HELPER}"; ! arc2_import_runpod_hf_token; ! arc2_import_runpod_pid1_secret UNSAFE; echo PASS'
        out=subprocess.run([bash(),'-lc',cmd],env={'PATH':os.environ.get('PATH',''),'ARC2_PID1_ENV_PATH':str(envfile)},text=True,capture_output=True,check=True)
        assert out.stdout=='PASS\n'
