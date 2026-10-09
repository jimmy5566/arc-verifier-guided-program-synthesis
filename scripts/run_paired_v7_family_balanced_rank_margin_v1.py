#!/usr/bin/env python3
"""Bounded no-update worker entrypoint; runtime implementation remains gated."""
from __future__ import annotations
import argparse,json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paired_rank_margin_launch_contract import CAP_SECONDS,require_launch

PROTOCOL='PAIRED_V7_FAMILY_BALANCED_CORRECT_TOKEN_RANK_MARGIN_V1'
def main()->int:
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--receipt',type=Path,required=True);a.add_argument('--runtime-seconds',type=int,default=CAP_SECONDS);z=a.parse_args()
 if not z.binding.is_file():raise RuntimeError('LAUNCH_BINDING_MISSING')
 binding=json.loads(z.binding.read_text(encoding='utf8'))
 required={'protocol_id','runtime_cap_seconds','expected_hashes','input_paths'}
 if not required.issubset(binding) or binding['protocol_id']!=PROTOCOL:raise RuntimeError('LAUNCH_BINDING_SCHEMA_INVALID')
 actual={key:Path(value) for key,value in binding['input_paths'].items()}
 require_launch(output=z.output,receipt=z.receipt,cap_seconds=z.runtime_seconds,expected_hashes=binding['expected_hashes'],actual_paths=actual)
 if binding['runtime_cap_seconds']!=CAP_SECONDS:raise RuntimeError('BINDING_RUNTIME_CAP_DRIFT')
 # This entrypoint intentionally cannot load models until the complete
 # preflight/collation/metric implementation is frozen and Director-reviewed.
 if z.runtime_seconds!=CAP_SECONDS:raise RuntimeError('RUNTIME_CAP_DRIFT')
 if not z.binding.is_file():raise RuntimeError('LAUNCH_BINDING_MISSING')
 raise RuntimeError('COMPLETE_RUNTIME_PREFLIGHT_AND_METRIC_WORKER_REQUIRED')
if __name__=='__main__':raise SystemExit(main())
