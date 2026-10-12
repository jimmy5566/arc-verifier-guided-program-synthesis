#!/usr/bin/env python3
"""CPU fixtures for the immutable RUN_002 raw-record verifier."""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.connected_component_run002_raw_manifest_v1 import canonical_sha256, verify_run002_raw_root

def main():
    record={'record_id':'R0','prompt_sha256':'p0','effective_batch_size':32,'generated_token_ids':[1,2], 'parser_valid':False, 'parsed_grid':None}
    expected={
      'expected_rows':[{'checkpoint':'V7','cohort_record_id':'OOD:0','prompt_task_sha256':'p0','run_002_raw_file':'V7_RAW.jsonl','run_002_record_id':'R0','run_002_raw_record_sha256':canonical_sha256(record),'run_002_effective_batch_size':32}],
    }
    with tempfile.TemporaryDirectory() as raw:
        root=Path(raw); (root/'V7_RAW.jsonl').write_text(__import__('json').dumps(record)+'\n',encoding='utf-8')
        verdict=verify_run002_raw_root(expected,root)
        assert verdict['status']=='PASS_RUN_002_EXACT_RAW_RECORD_MANIFEST' and verdict['verified_rows']==1
        altered=dict(record); altered['generated_token_ids']=[1,3]
        (root/'V7_RAW.jsonl').write_text(__import__('json').dumps(altered)+'\n',encoding='utf-8')
        try: verify_run002_raw_root(expected,root)
        except RuntimeError as exc: assert str(exc)=='RUN_002_RAW_RECORD_SHA_MISMATCH:V7_RAW.jsonl'
        else: raise AssertionError('expected mutation rejection')
        altered=dict(record); altered['target_token_ids']=[2]
        altered['run_002_raw_record_sha256']=canonical_sha256(altered)
        target_expected={'expected_rows':[dict(expected['expected_rows'][0],run_002_raw_record_sha256=altered['run_002_raw_record_sha256'])]}
        (root/'V7_RAW.jsonl').write_text(__import__('json').dumps(altered)+'\n',encoding='utf-8')
        try: verify_run002_raw_root(target_expected,root)
        except RuntimeError as exc: assert str(exc)=='RUN_002_RAW_TARGET_BOUNDARY_VIOLATION:V7_RAW.jsonl'
        else: raise AssertionError('expected target boundary rejection')
    print('PASS_RUN_002_RAW_MANIFEST_V1')
if __name__=='__main__': main()
