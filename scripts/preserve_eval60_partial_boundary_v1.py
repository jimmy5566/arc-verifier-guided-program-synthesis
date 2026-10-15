"""Stop the explicitly named Eval60 chain after an atomic output, then hash it."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def completed(run: Path) -> dict[str, dict]:
    rows = {}
    for path in (run / 'OUTPUT_RECEIPTS').glob('*.json'):
        row = json.loads(path.read_text())
        if row.get('status') == 'COMPLETE':
            rows[row['output_id']] = {'receipt_path': str(path.relative_to(run)), **row}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--worker-pid', type=int, required=True)
    parser.add_argument('--parent-pids', type=int, nargs='+', required=True)
    parser.add_argument('--timeout', type=int, default=2400)
    parser.add_argument('--stop-now', action='store_true')
    parser.add_argument('--identity-root', type=Path)
    args = parser.parse_args()
    if (args.run / 'GENERATION_FREEZE.json').exists():
        raise RuntimeError('already frozen; do not stop or alter completed experiment')
    args.snapshot.mkdir(parents=True, exist_ok=False)
    # Verify every PID against the exact source/run before sending any signal.
    for pid in [args.worker_pid, *args.parent_pids]:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if str(args.identity_root or args.run) not in command:
            raise RuntimeError(f'PID {pid} is not bound to the requested run')
    initial = completed(args.run)
    for pid in args.parent_pids:
        os.kill(pid, signal.SIGSTOP)
    atomic(args.snapshot / 'BOUNDARY_REQUEST.json', {
        'status': 'WAITING_OUTPUT_BOUNDARY', 'initial_completed': len(initial),
        'worker_pid': args.worker_pid, 'parent_pids': args.parent_pids,
        'run': str(args.run), 'requested_unix': time.time(),
    })
    deadline = time.monotonic() + args.timeout
    while not args.stop_now and len(completed(args.run)) <= len(initial):
        if not Path(f'/proc/{args.worker_pid}').exists():
            raise RuntimeError('worker exited before boundary; inspect existing artifacts')
        if time.monotonic() >= deadline:
            raise RuntimeError('boundary timeout; parents remain stopped for manual inspection')
        time.sleep(.25)
    os.kill(args.worker_pid, signal.SIGSTOP)
    rows = completed(args.run)
    atomic(args.snapshot / 'CONTROLLED_STOP.json', {
        'status': 'STOPPED_WITH_UNCOMMITTED_WORK_DISCARDED' if args.stop_now else 'STOPPED_AFTER_ATOMIC_OUTPUT',
        'completed_outputs': sorted(rows),
        'count': len(rows), 'run': str(args.run), 'stopped_unix': time.time(),
        'target_blind': True, 'gold_loaded': False,
    })
    # Prevent the old wrapper from advancing into scoring or launching another worker.
    for pid in [*args.parent_pids, args.worker_pid]:
        try:
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    files = {}
    for output_id, row in sorted(rows.items()):
        for key in ['receipt_path', 'raw_output', 'checkpoint_output', 'eos_events']:
            relative = row[key]
            path = args.run / relative
            if not path.is_file():
                raise RuntimeError(f'COMPLETE output missing artifact: {output_id} {key}')
            files[relative] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        atomic(args.snapshot / 'PRESERVATION_PROGRESS.json', {
            'hashed_outputs': sum(1 for p in files if p.startswith('OUTPUT_RECEIPTS/')),
            'total_outputs': len(rows), 'updated_unix': time.time(),
        })
    for name in ['PRE_GOLD_CONTRACT.json', 'CONTRACT.json', 'RUN_COHORT.json',
                 'ADAPTER_IDENTITIES.json', 'EXECUTION_POLICY_AMENDMENT_FIXED4_V1.json']:
        path = args.run / name
        if path.is_file():
            files[name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    sources = sorted({row['execution_source_commit'] for row in rows.values()
                      if row.get('execution_source_commit')})
    atomic(args.snapshot / 'PARTIAL_EVIDENCE_MANIFEST.json', {
        'status': 'PRESERVED_PARTIAL_TARGET_BLIND',
        'source_commit': sources[0] if len(sources) == 1 else 'MIXED_OR_RECEIPT_UNSPECIFIED',
        'source_commits_from_receipts': sources,
        'run': str(args.run), 'completed_outputs': sorted(rows), 'count': len(rows),
        'files': files, 'target_blind': True, 'gold_loaded': False,
        'new_generation_may_not_overwrite_these_files': True,
    })


if __name__ == '__main__':
    main()
