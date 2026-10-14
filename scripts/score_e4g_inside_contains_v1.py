"""Target-sidecar guarded E4G scorer."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.e4g_inside_contains_atomic import decision
def score(path):
 x=json.loads(Path(path).read_text())
 if not x.get('raw_freeze_complete') or not x.get('sidecar_opened_after_raw_freeze'):raise RuntimeError('E4G:RAW_FREEZE_REQUIRED')
 return decision(x['metrics'])

def require_post_freeze(raw_freeze_path, raw_paths):
 """Fail closed before a sidecar or score consumes any target material."""
 x=json.loads(Path(raw_freeze_path).read_text(encoding='utf-8'))
 if x.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or x.get('targets_accessed') is not False:
  raise RuntimeError('E4G:RAW_FREEZE_REQUIRED')
 expected=x.get('raw_sha256')
 actual={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in raw_paths.items()}
 if expected!=actual or sorted(actual)!=['TRAINED','V7']:
  raise RuntimeError('E4G:RAW_FREEZE_HASH_MISMATCH')
 return actual
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--cpu-self-test',action='store_true');ap.add_argument('--score');a=ap.parse_args()
 if a.cpu_self_test:
  try:score(Path(__file__))
  except Exception:print({'status':'PASS_E4G_SCORER_FAIL_CLOSED_TARGET_GUARD'})
 elif a.score: print(score(a.score))
 else:ap.error('choose mode')
