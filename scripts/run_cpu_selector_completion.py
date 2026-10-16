"""Bounded CPU-only selector reuse wrapper with monitor-readable terminal status."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys

from scripts.score_v7_frozen_oracle_only import write, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-wall-seconds', type=int, default=1800)
    parser.add_argument('--worker-module', choices=('scripts.score_reused_rankings_cpu', 'scripts.analyze_frozen_eval60_search_cpu', 'scripts.analyze_cached_gold_sequence_nll_cpu'), default='scripts.score_reused_rankings_cpu')
    parser.add_argument('scorer_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.max_wall_seconds <= 1800:
        raise RuntimeError('CPU_BOUND_MUST_BE_AT_MOST_1800_SECONDS')
    args.output.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    state = {'status': 'RUNNING', 'stage': args.worker_module, 'source_commit': source,
             'execution_source_commit': source,
             'started_utc': datetime.now(timezone.utc).isoformat(), 'gpu_used': False,
             'model_loaded': False, 'generation_rerun': False, 'training_rerun': False,
             'owner_scope': 'Latest owner CPU-only reuse continuation; not a reset of expired GPU window',
             'wall_limit_seconds': args.max_wall_seconds, 'partial_output_preserved_on_failure': True}
    def publish():
        path = args.output / 'PIPELINE_STATE.json'
        tmp = path.with_suffix('.tmp')
        write(tmp, state)
        tmp.replace(path)
    publish()
    forwarded = args.scorer_args[1:] if args.scorer_args[:1] == ['--'] else args.scorer_args
    command = [sys.executable, '-m', args.worker_module,
               '--output', str(args.output / 'rankings'), *forwarded]
    try:
        with (args.output / 'cpu.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=(os.name != 'nt'))
            state['child_pid'] = child.pid
            publish()
            try:
                code = child.wait(timeout=args.max_wall_seconds)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    child.terminate()
                else:
                    os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=30)
                raise RuntimeError('CPU_REUSE_TIMEOUT_CHECKPOINTS_PRESERVED')
        if code:
            raise RuntimeError('CPU_SCORER_EXIT_' + str(code))
        scorecard = args.output / ('rankings/CPU_REUSED_SCORECARD.json' if args.worker_module == 'scripts.score_reused_rankings_cpu' else 'rankings/RESULT.json')
        state.update(status='COMPLETE', scorecard=str(scorecard), scorecard_sha256=digest(scorecard.read_bytes()))
    except Exception as exc:
        state.update(status='FAILED', reason=type(exc).__name__ + ':' + str(exc))
    state['ended_utc'] = datetime.now(timezone.utc).isoformat()
    publish()
    return int(state['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
