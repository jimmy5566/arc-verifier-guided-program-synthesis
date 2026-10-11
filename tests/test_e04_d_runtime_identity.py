"""Integration checks for E04-D's immutable-source runtime contract."""
from __future__ import annotations
import importlib.util, subprocess, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('e04d_worker',ROOT/'scripts'/'run_e04_d_rotation_marker_gradient.py')
WORKER=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(WORKER)

def _git(repo,*args):
 return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()

def test_detached_descendant_of_approved_snapshot_is_accepted():
 with tempfile.TemporaryDirectory() as tmp:
  repo=Path(tmp);_git(repo,'init','-q');_git(repo,'config','user.email','test@example.invalid');_git(repo,'config','user.name','ARC2 test')
  (repo/'x').write_text('approved\n');_git(repo,'add','x');_git(repo,'commit','-qm','approved');approved=_git(repo,'rev-parse','HEAD')
  (repo/'x').write_text('advanced\n');_git(repo,'commit','-am','advance','-q');head=_git(repo,'rev-parse','HEAD');_git(repo,'checkout','--detach','-q',head)
  assert WORKER.validate_runtime_identity({'approved_source_commit':approved},repo)==head

def test_wrong_approved_snapshot_and_tracked_edits_fail_closed():
 with tempfile.TemporaryDirectory() as tmp:
  repo=Path(tmp);_git(repo,'init','-q');_git(repo,'config','user.email','test@example.invalid');_git(repo,'config','user.name','ARC2 test')
  (repo/'x').write_text('base\n');_git(repo,'add','x');_git(repo,'commit','-qm','base');approved=_git(repo,'rev-parse','HEAD')
  try:WORKER.validate_runtime_identity({'approved_source_commit':'0'*40},repo)
  except RuntimeError as exc:assert str(exc)=='E04D_APPROVED_SOURCE_NOT_ANCESTOR'
  else:raise AssertionError('wrong source accepted')
  (repo/'x').write_text('dirty\n')
  try:WORKER.validate_runtime_identity({'approved_source_commit':approved},repo)
  except RuntimeError as exc:assert str(exc)=='E04D_TRACKED_SOURCE_DIRTY'
  else:raise AssertionError('dirty source accepted')
