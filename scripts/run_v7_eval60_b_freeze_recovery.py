#!/usr/bin/env python3
"""Bounded target-blind B freeze from a completed archive; no regeneration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_v7_eval60_archive_recovery import atomic, remaining_seconds


def command_for(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(ROOT / 'scripts/run_v7_ttt24_eval60_b_selector.py')]
    for name in ('archive', 'training', 'adapter_root', 'model_path', 'foundation_adapter', 'challenge', 'native_config'):
        command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    command += ['--output', str(args.output / 'b_selector'), '--batch-size', '8']
    if getattr(args, 'reuse_shards_from', None) is not None:
        command += ['--reuse-shards-from', str(args.reuse_shards_from)]
    for name in ('required_reuse_index', 'required_reuse_index_sha256'):
        if getattr(args, name, None) is not None:
            command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ('archive', 'training', 'adapter_root', 'model_path', 'foundation_adapter', 'challenge', 'native_config', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--max-wall-seconds', type=int, default=14400)
    parser.add_argument('--reuse-shards-from', type=Path)
    parser.add_argument('--required-reuse-index', type=Path)
    parser.add_argument('--required-reuse-index-sha256')
    args = parser.parse_args()
    budget = remaining_seconds(args.deadline_utc, args.max_wall_seconds)
    if budget <= 40:
        raise RuntimeError('OWNER_WINDOW_EXHAUSTED')
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'schema': 'V7_EVAL60_B_FREEZE_RECOVERY_V1', 'status': 'RUNNING', 'stage': 'b_selector',
             'execution_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'started_utc': datetime.now(timezone.utc).isoformat(), 'owner_deadline_utc': args.deadline_utc,
             'wall_limit_seconds': budget, 'archive': str(args.archive), 'generation_rerun': False,
             'training_rerun': False, 'gold_loaded': False, 'gpu_stage': True,
             'partial_output_preserved_on_failure': True, 'command': command_for(args)}
    status = args.output / 'PIPELINE_STATE.json'
    atomic(status, state)
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'src'), TOKENIZERS_PARALLELISM='false',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', HF_HOME='/workspace/arc2/hf_cache')
    try:
        with (args.output / 'b_selector.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(state['command'], cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=(os.name != 'nt'))
            state['child_pid'] = child.pid
            atomic(status, state)
            try:
                code = child.wait(timeout=budget - 40)
            except subprocess.TimeoutExpired:
                if os.name != 'nt':
                    os.killpg(child.pid, signal.SIGTERM)
                else:
                    child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name != 'nt':
                        os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                    child.wait(timeout=10)
                raise RuntimeError('BOUNDED_B_FREEZE_TIMEOUT_PARTIAL_PRESERVED')
        if code:
            raise RuntimeError(f'B_FREEZE_EXIT_{code}')
        import json
        freeze = json.loads((args.output / 'b_selector/PRE_GOLD_FREEZE.json').read_text())
        if (freeze.get('status') != 'PASS' or freeze.get('gold_loaded') is not False
                or freeze.get('B_attempts_frozen') is not True or freeze.get('output_count') != 89):
            raise RuntimeError('B_FREEZE_CONTRACT_FAILED')
        state.update(status='COMPLETE', b_freeze=str(args.output / 'b_selector'))
    except Exception as exc:
        state.update(status='FAILED', failed_stage='b_selector', reason=f'{type(exc).__name__}:{exc}')
    finally:
        state['ended_utc'] = datetime.now(timezone.utc).isoformat()
        atomic(status, state)
    return 0 if state['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
