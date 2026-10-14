"""Target-sidecar guarded E4G scorer."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.e4g_inside_contains_atomic import decision
def score(path):
 x=json.loads(Path(path).read_text())
 if not x.get('raw_freeze_complete') or not x.get('sidecar_opened_after_raw_freeze'):raise RuntimeError('E4G:RAW_FREEZE_REQUIRED')
 return decision(x['metrics'])
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--cpu-self-test',action='store_true');ap.add_argument('--score');a=ap.parse_args()
 if a.cpu_self_test:
  try:score(Path(__file__))
  except Exception:print({'status':'PASS_E4G_SCORER_FAIL_CLOSED_TARGET_GUARD'})
 elif a.score: print(score(a.score))
 else:ap.error('choose mode')
