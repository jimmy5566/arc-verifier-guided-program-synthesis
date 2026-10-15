#!/usr/bin/env python3
"""Restore the historical READY engine, grouped by identical task adapter.

The decoder, cache pack/adopt, admission queue and fairness selector are the
existing production implementations. This file binds them to the current V7
task adapters and preserves per-output checkpoints/EOS/node evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from arc.io import load_dataset
from inference.chunked_kv_cache import ChunkedDynamicCache
from inference.hf_peft_backend import load_hf_peft_inference
from inference.nvarc_turbodfs_dynamic_ready import ready_result, start_ready_cell
from inference.rolling_resident_pool import run_rolling_resident_scheduler
from inference.root_length_memory_profile import KV_BLOCK_TOKENS
from scripts.run_clean_hf_parallel_regret_dfs_v1 import _assert_challenge_only, _config
from scripts.run_eval60_budget_eos_pilot18_v1 import (
    _ancestry_tokens, _atomic_json, _canonical, _checkpoint_values, _json_sha,
    _load_aug16, _memory, _native_prompt_record, _prefix_hash, _read, _safe_id,
)
from scripts.run_real_project_aug16_dynamic_b16_pilot_v1 import (
    _candidate_payload, _canonical_candidate_grid,
)
from scripts.run_ttt24_aug8_r1024_core_v1 import AUG8, _sha_file

MODE = 'HISTORICAL_ROOT_AWARE_READY_RESTORED_V1'


def task_groups(outputs: list[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(outputs, key=lambda x: (str(x['task_id']), int(x['output_index']))):
        groups[str(row['task_id'])].append(row)
    for group in groups.values():
        identities = {(x['adapter_identity']['adapter_sha256'],
                       x['adapter_identity']['adapter_config_sha256']) for x in group}
        if len(identities) != 1:
            raise RuntimeError('cannot batch different adapter identities')
    return list(groups.values())


def capacity(root_max: int, lane_count: int) -> tuple[int, int]:
    """Historical S/M/L/XL resident limits with B16 -> B8 soft protection."""
    if root_max <= 2048:
        resident, ceiling = 16, 16
    elif root_max <= 2653:
        resident, ceiling = 8, 8
    elif root_max <= 6493:
        resident, ceiling = 4, 4
    elif root_max <= 17245:
        resident, ceiling = 2, 2
    else:
        raise RuntimeError('root exceeds authorized Eval60 capacity envelope')
    return min(resident, lane_count), min(ceiling, lane_count)


def complete(run: Path, row: dict) -> bool:
    safe = _safe_id(row['output_id'])
    receipt = run / 'OUTPUT_RECEIPTS' / f'{safe}.json'
    if not receipt.exists():
        return False
    r = _read(receipt)
    if r.get('status') != 'COMPLETE' or r.get('output_id') != row['output_id']:
        raise RuntimeError('invalid existing output receipt')
    if not all((run / r[k]).is_file() for k in ['raw_output', 'checkpoint_output', 'eos_events']):
        raise RuntimeError('COMPLETE receipt has missing artifacts')
    return True


def canary_mechanically_valid(cells: list[dict]) -> bool:
    # The inherited 540-second wall limit is a valid censoring condition.
    # R1024 is a maximum node budget, never a requirement to erase censoring
    # by extending time or regenerating otherwise valid candidate evidence.
    return len(cells) == 8 and all(
        c['nodes_expanded'] <= 1024 and c['completed_candidates'] <= 32
        and not c['prefix_reconstruction_failures']
        and c['diagnostic_trace']['prefill']['cache_type'].endswith('ChunkedDynamicCache')
        and all(math.isfinite(lp['logprob']) for b in c['diagnostic_trace']['branch_probabilities']
                for lp in b['full_arc_logprobs']) for c in cells)


def cell_payload(cell: Any, selected: dict, candidate: dict, events: list[dict],
                 checkpoints: tuple[int, ...]) -> dict:
    """Historical Pilot18 artifact contract, using its authoritative oracles."""
    result = ready_result(cell)
    nodes = [dict(node) for node in result.nodes]
    for event in events:
        prefix = _ancestry_tokens(nodes, event.get('parent_node_id'))
        event['parent_prefix_token_ids'] = list(prefix)
        event['prefix_reconstruction_pass'] = _prefix_hash(prefix) == event['parent_prefix_hash']
        if not event['prefix_reconstruction_pass']:
            raise RuntimeError('EOS parent ancestry does not match trace identity')
        terminal = event.get('candidate_terminal_node_id')
        if terminal is not None:
            event['candidate_token_ids'] = list(_ancestry_tokens(nodes, int(terminal)))
    completed = [event for event in events if event.get('candidate_completed')]
    expanded = sum(node.get('state') == 'expanded' for node in nodes)
    rows = []
    for checkpoint in checkpoints:
        records = []
        for event in completed:
            if int(event['nodes_expanded_so_far']) > checkpoint:
                continue
            tokens = tuple(int(v) for v in event['candidate_token_ids'])
            records.append({
                'candidate_id': event.get('candidate_completion_index'), 'token_ids': list(tokens),
                'canonical_grid': _canonical_candidate_grid(tokens, candidate),
                'terminal_node_id': event.get('candidate_terminal_node_id'),
                'nodes_expanded_so_far': event.get('nodes_expanded_so_far'),
            })
        rows.append({
            'output_id': selected['output_id'], 'profile': selected['profile'],
            'cell_key': cell.cell_key, 'augmentation_id': candidate['candidate_id'],
            'checkpoint_requested': checkpoint, 'checkpoint_reached': expanded >= checkpoint,
            'nodes_expanded': expanded, 'completed_candidate_count': len(records),
            'candidate_pool_snapshot': records, 'candidate_pool_sha256': _json_sha(records),
            'termination_reason': result.termination_reason,
            'carried_forward_terminal': expanded < checkpoint and result.termination_reason != 'wall_time',
            'wall_time_censored': result.termination_reason == 'wall_time' and expanded < checkpoint,
            'actual_nodes_reached': expanded, 'elapsed_seconds': cell.active_elapsed_seconds,
            'active_elapsed_seconds': cell.active_elapsed_seconds,
            'model_forwards': result.model_forwards, 'tokens_advanced': result.tokens_advanced,
            'valid_kv_length': int(cell.cache_owner.cache.valid_lengths()[0]),
            'capacity_kv_length': int(cell.cache_owner.cache.capacity_lengths()[0]),
        })
    pool, valid, invalid = _candidate_payload(cell, candidate)
    if expanded > 1024 or result.completed_candidates > 32:
        raise RuntimeError('logical cell exceeded frozen budget')
    return {
        'cell_key': cell.cell_key, 'augmentation_id': candidate['candidate_id'], 'nodes': nodes,
        'events': events, 'checkpoints': rows, 'final_candidate_pool': pool,
        'final_candidate_pool_sha256': _json_sha(pool), 'nodes_expanded': expanded,
        'model_forwards': result.model_forwards, 'tokens_advanced': result.tokens_advanced,
        'completed_candidates': result.completed_candidates, 'valid_candidates': valid,
        'invalid_candidates': invalid, 'termination_reason': result.termination_reason,
        'budget_exhausted': result.budget_exhausted, 'model_forward_seconds': result.model_forward_seconds,
        'prefill_seconds': cell.prefill_seconds, 'active_elapsed_seconds': cell.active_elapsed_seconds,
        'max_frontier_size': result.max_frontier_size, 'prefix_reconstruction_failures': [],
        'search_order': 'legacy', 'search_order_work_items': [], 'top5_hidden_state_capture': None,
        'diagnostic_trace': {
            'prefill': cell.state.get('prefill_trace'),
            'logical_advances': cell.state.get('per_forward_trace', []),
            'branch_probabilities': cell.state.get('branch_probabilities', []),
            'frontier_samples': cell.state.get('frontier_samples', []),
            'search_trace': cell.state.get('search_trace', []),
        },
    }


class Runtime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.identity = None
        self.task = None

    def load(self, selected: dict) -> dict:
        import torch
        began = time.perf_counter()
        adapter = Path(selected['adapter_path'])
        expected = selected['adapter_identity']
        # One exact verification per adapter admission, never a full 60-adapter rehash.
        for name, key in [('adapter_model.safetensors', 'adapter_sha256'),
                          ('adapter_config.json', 'adapter_config_sha256')]:
            if _sha_file(adapter / name) != expected[key]:
                raise RuntimeError(f'adapter identity mismatch: {selected["task_id"]} {name}')
        verified = time.perf_counter() - began
        began = time.perf_counter()
        first = self.model is None
        if first:
            self.model, self.tokenizer, self.identity = load_hf_peft_inference(
                model_path=self.args.model_path, adapter_path=adapter, device=self.args.device,
                native_config_dir=self.args.native_config_dir,
                foundation_adapter_path=self.args.foundation_adapter,
                frozen_adapter_identity={k: expected[k] for k in ['adapter_sha256', 'adapter_config_sha256']},
            )
        else:
            self.model.delete_adapter('task')
            self.model.load_adapter(str(adapter), adapter_name='task', is_trainable=False,
                                    autocast_adapter_dtype=False)
            self.model.base_model.set_adapter(['foundation', 'task'])
            self.identity = {**self.identity, **{k: expected[k] for k in ['adapter_sha256', 'adapter_config_sha256']}}
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.args.device).eval()
        self.task = selected['task_id']
        gc.collect()
        torch.cuda.empty_cache()
        elapsed = time.perf_counter() - began
        return {'adapter_verify_seconds': verified, 'adapter_stage_seconds': 0.,
                'model_load_seconds': elapsed if first else 0.,
                'adapter_swap_seconds': 0. if first else elapsed, 'persistent_reuse': not first}


def run_task(args: argparse.Namespace, runtime: Runtime, group: list[dict],
             tasks: dict, candidates: list[dict], resident: int | None = None,
             ceiling: int | None = None, attempt: int = 0) -> None:
    import torch
    timings = runtime.load(group[0])
    model, tokenizer = runtime.model, runtime.tokenizer
    prepared, roots, binding = {}, {}, {}
    events: dict[str, list] = defaultdict(list)
    final: dict[str, dict] = {}
    residents = {}
    for selected in group:
        for candidate in candidates:
            key = f"{selected['task_id']}:o{selected['output_index']}:d24:aug8:{candidate['candidate_id']}"
            ids, meta = _native_prompt_record(tokenizer=tokenizer, task=tasks[selected['task_id']],
                                             output_index=int(selected['output_index']), candidate=candidate)
            prepared[key] = (ids, meta)
            roots[key] = int(ids.shape[-1])
            binding[key] = (selected, candidate)
    root_max = max(roots.values())
    default_resident, default_ceiling = capacity(root_max, len(roots))
    resident = min(resident or default_resident, len(roots))
    ceiling = min(ceiling or default_ceiling, resident)
    config = _config(1024, diagnostic_trace=True)
    schedule = (128, 256, 512, 1024)
    started = time.time()
    progress = args.output / 'READY_LIVE_STATE.json'
    soft_disabled = False
    actual_ceiling = ceiling

    def live(event: dict) -> None:
        if event['event'] == 'FORWARD' and event['physical_forward_index'] % 100:
            return
        _atomic_json(progress, {'status': 'RUNNING', 'mode': MODE,
                     'task_id': group[0]['task_id'], 'outputs': [r['output_id'] for r in group],
                     'attempt': attempt, 'updated_unix': time.time(), **event})

    def create(key: str):
        selected, candidate = binding[key]
        def transform(legacy):
            return ChunkedDynamicCache.from_legacy_cache(legacy, block_tokens=KV_BLOCK_TOKENS, owner_id=key)
        def eos(event):
            events[key].append({'output_id': selected['output_id'], 'task_id': selected['task_id'],
                                'profile': selected['profile'], 'augmentation_id': candidate['candidate_id'], **event})
        cell = start_ready_cell(model=model, input_ids=prepared[key][0].to(args.device),
                                config=config, cell_key=key, normalize_root_cache=True,
                                root_cache_transform=transform, cache_strategy='rollback',
                                release_prefill_temporaries=True, eos_event_sink=eos)
        residents[key] = cell
        if cell.cache_owner is None or cell.request is None:
            raise RuntimeError('historical READY admission produced no owned request')
        return cell

    def consume(key, cell):
        selected, candidate = binding[key]
        final[key] = cell_payload(cell, selected, candidate, events[key], schedule)

    def release(key, cell):
        if cell.cache_owner is not None:
            cell.cache_owner.cache = None
        cell.cache_owner = None
        cell.request = None
        cell.generator.close()
        residents.pop(key, None)

    def safe_ceiling(position):
        nonlocal soft_disabled, actual_ceiling
        if ceiling > 8 and (position > 2048 or torch.cuda.memory_allocated(args.device) >= 20 * 1024**3):
            soft_disabled = True
        actual_ceiling = min(ceiling, 8) if soft_disabled else ceiling
        # Current cache length, not the original profile label, bounds each call.
        if position > 6493:
            return min(actual_ceiling, 2)
        if position > 2653:
            return min(actual_ceiling, 4)
        if position > 2048:
            return min(actual_ceiling, 8)
        return actual_ceiling

    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    began = time.perf_counter()
    try:
        scheduler = run_rolling_resident_scheduler(
            model=model, pending_ids=list(prepared), resident_capacity=resident,
            physical_batch_ceiling=ceiling, create_cell=create, consume_result=consume,
            release_cell=release, admission_policy='root_aware', root_lengths=roots,
            safe_batch_ceiling=safe_ceiling, fairness_max_wait=3, event_sink=live,
        )
        torch.cuda.synchronize(args.device)
        wall = time.perf_counter() - began
        memory = {**_memory(torch, args.device), 'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(args.device)),
                  'peak_reserved_bytes': int(torch.cuda.max_memory_reserved(args.device))}
        group_id = '__'.join(_safe_id(r['output_id']) for r in group)
        scheduler_path = args.output / 'READY_TASK_SCHEDULER' / f'{group_id}_a{attempt}.json'
        _atomic_json(scheduler_path, scheduler)
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        for selected in group:
            safe = _safe_id(selected['output_id'])
            cells = {k: v for k, v in final.items() if binding[k][0]['output_id'] == selected['output_id']}
            if len(cells) != 8 or {x['augmentation_id'] for x in cells.values()} != set(AUG8):
                raise RuntimeError('output does not contain all eight frozen views')
            if complete(args.output, selected):
                raise RuntimeError('attempt to overwrite completed output')
            output_events = [e for k in cells for e in events[k]]
            eos_path = args.output / 'EOS_EVENTS' / f'{safe}.jsonl.gz'
            eos_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = eos_path.with_suffix(eos_path.suffix + '.tmp')
            with gzip.open(temporary, 'wt', encoding='utf-8', newline='\n') as handle:
                for event in output_events:
                    handle.write(_canonical(event) + '\n')
            os.replace(temporary, eos_path)
            checkpoint_path = args.output / 'OUTPUT_CHECKPOINTS' / f'{safe}.json'
            _atomic_json(checkpoint_path, {'experiment': args.experiment, 'target_blind': True, 'gold_loaded': False,
                         'output_id': selected['output_id'], 'cells': {k: v['checkpoints'] for k, v in cells.items()}})
            output_timings = {**timings, 'search_seconds': wall / len(group)}
            raw_path = args.output / 'RAW_OUTPUTS' / f'{safe}.json'
            _atomic_json(raw_path, {
                'experiment': args.experiment, 'target_blind': True, 'gold_loaded': False,
                'status': 'COMPLETE', 'output': selected, 'attempt_index': attempt,
                'execution_source_commit': source, 'execution_mode': MODE,
                'runtime_identity': runtime.identity, 'logical_search_order_policy': 'legacy',
                'profile_configuration': {'resident_capacity': resident, 'physical_batch_ceiling': ceiling},
                'cells': cells, 'scheduler': {k: v for k, v in scheduler.items() if k != 'events'},
                'scheduler_events': scheduler['events'], 'scheduler_scope': 'shared_task_group',
                'scheduler_group_id': group_id, 'scheduler_group_output_count': len(group),
                'root_admission': {'policy': 'root_aware', 'root_lengths': roots, 'fairness_max_wait': 3},
                'wall_seconds': wall / len(group), 'shared_group_wall_seconds': wall,
                'timings': output_timings, 'memory': memory, 'eos_events_path': str(eos_path.relative_to(args.output)),
                'eos_event_count': len(output_events), 'top5_hidden_state_capture': {'enabled': False},
            })
            _atomic_json(args.output / 'OUTPUT_RECEIPTS' / f'{safe}.json', {
                'experiment': args.experiment, 'status': 'COMPLETE', 'target_blind': True, 'gold_loaded': False,
                'output_id': selected['output_id'], 'execution_mode': MODE, 'execution_source_commit': source,
                'attempt_index': attempt, 'raw_output': str(raw_path.relative_to(args.output)),
                'checkpoint_output': str(checkpoint_path.relative_to(args.output)),
                'eos_events': str(eos_path.relative_to(args.output)), 'timings': output_timings,
                'wall_seconds': wall / len(group), 'started_unix': started, 'ended_unix': time.time(),
                'scheduler_group_id': group_id,
            })
        _atomic_json(progress, {'status': 'TASK_COMPLETE', 'task_id': group[0]['task_id'],
                     'updated_unix': time.time(), 'physical_batch_histogram': scheduler['physical_batch_histogram'],
                     'mean_effective_batch': scheduler['mean_effective_batch'], 'group_wall_seconds': wall})
    finally:
        for key, cell in list(residents.items()):
            release(key, cell)
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ['output', 'model_path', 'foundation_adapter', 'challenge', 'native_config_dir',
                 'candidate_pool', 'cohort', 'aug8_ids']:
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    p.add_argument('--output-ids', nargs='*')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--experiment', default='V7_TTT24_AUG8_DFS_R1024_EVAL60_V1')
    p.add_argument('--preservation-manifest', type=Path)
    p.add_argument('--canary-output', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'GENERATION_FREEZE.json').exists():
        raise RuntimeError('generation already frozen')
    _assert_challenge_only(args.challenge)
    if args.preservation_manifest:
        preserved = _read(args.preservation_manifest)
        if preserved['status'] != 'PRESERVED_PARTIAL_TARGET_BLIND':
            raise RuntimeError('partial evidence preservation not complete')
        # Preserve exact source ledger; a final ledger verifies the entire mixed run.
        for relative, identity in preserved['files'].items():
            if (args.output / relative).stat().st_size != identity['bytes']:
                raise RuntimeError('preserved artifact changed size')
    outputs = _read(args.cohort)['outputs']
    if args.output_ids:
        wanted = set(args.output_ids)
        outputs = [r for r in outputs if r['output_id'] in wanted]
        if {r['output_id'] for r in outputs} != wanted:
            raise RuntimeError('canary ID absent from frozen cohort')
    candidates = _load_aug16(args.candidate_pool, args.aug8_ids)
    if tuple(r['candidate_id'] for r in candidates) != AUG8:
        raise RuntimeError('canonical AUG8 identity/order mismatch')
    tasks = load_dataset(args.challenge)
    runtime = Runtime(args)
    import torch

    def execute(group):
        group = [r for r in group if not complete(args.output, r)]
        if not group:
            return
        # Whole-task OOM retries discard only uncommitted work. Logical budgets unchanged.
        for attempt in range(5):
            resident = ceiling = None if attempt == 0 else max(1, 8 >> (attempt - 1))
            try:
                run_task(args, runtime, group, tasks, candidates, resident, ceiling, attempt)
                break
            except torch.OutOfMemoryError as error:
                _atomic_json(args.output / 'FAILED_ATTEMPTS' / f"{group[0]['task_id']}_ready_a{attempt}.json", {
                    'status': 'OOM', 'target_blind': True, 'gold_loaded': False,
                    'attempt': attempt, 'error': str(error), 'ended_unix': time.time(),
                    'discarded_uncommitted_task_work': True,
                })
                runtime.model = None
                runtime.tokenizer = None
                gc.collect()
                torch.cuda.empty_cache()
                group = [r for r in group if not complete(args.output, r)]
                if attempt == 4:
                    raise

    if args.canary_output:
        # Prospective fixed canaries: a remaining same-adapter, equal-root pair
        # admits 16 real cells; the longest historical output checks XL rollback.
        canary_ids = {'dd6b8c4b:o0', 'dd6b8c4b:o1', '981571dc:o0'}
        canary_rows = [r for r in _read(args.cohort)['outputs'] if r['output_id'] in canary_ids]
        if {r['output_id'] for r in canary_rows} != canary_ids:
            raise RuntimeError('prospective restored READY canaries absent')
        production = args.output
        args.output = args.canary_output
        args.output.mkdir(parents=True, exist_ok=True)
        for group in sorted(task_groups(canary_rows), key=lambda g: max(int(r['root_length_max']) for r in g)):
            execute(group)
        gate_rows = []
        for selected in canary_rows:
            raw = _read(args.output / 'RAW_OUTPUTS' / f"{_safe_id(selected['output_id'])}.json")
            cells = list(raw['cells'].values())
            passed = canary_mechanically_valid(cells)
            gate_rows.append({'output_id': selected['output_id'], 'mechanical_pass': passed,
                              'histogram': raw['scheduler']['physical_batch_histogram'],
                              'mean_effective_batch': raw['scheduler']['mean_effective_batch'],
                              'wall_time_censored_cells': sum(c['termination_reason'] == 'wall_time' for c in cells),
                              'wall_seconds': raw['wall_seconds']})
        passed = all(r['mechanical_pass'] for r in gate_rows)
        _atomic_json(production / 'RESTORED_READY_CANARY_GATE.json', {
            'status': 'PASS' if passed else 'FAIL', 'target_blind': True, 'gold_loaded': False,
            'rows': gate_rows, 'criterion': 'all eight views; independent budgets; finite logits; exact EOS ancestry; owned ChunkedDynamicCache; XL completion',
            'batch16_observed': any(int(r['histogram'].get('16', 0)) > 0 for r in gate_rows),
            'wall_censoring_policy': 'preserve inherited 540 seconds; retain and report censored evidence without retry',
        })
        if not passed:
            raise RuntimeError('restored READY mechanical canary failed')
        # Preserve completed canary bytes, and reuse previously ungenerated outputs.
        for selected in canary_rows:
            if complete(production, selected):
                continue
            safe = _safe_id(selected['output_id'])
            receipt = _read(args.output / 'OUTPUT_RECEIPTS' / f'{safe}.json')
            for relative in [receipt['raw_output'], receipt['checkpoint_output'], receipt['eos_events'],
                             f'OUTPUT_RECEIPTS/{safe}.json']:
                source, dest = args.output / relative, production / relative
                if dest.exists():
                    raise RuntimeError('canary adoption would overwrite existing evidence')
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dest)
                if _sha_file(source) != _sha_file(dest):
                    raise RuntimeError('canary adoption exact-byte verification failed')
        args.output = production
    for group in task_groups(outputs):
        if (args.output / 'STOP_AFTER_TASK').exists():
            _atomic_json(args.output / 'READY_CONTROLLED_STOP.json', {'status': 'STOPPED_AT_TASK_BOUNDARY',
                         'updated_unix': time.time(), 'next_task': group[0]['task_id']})
            return
        execute(group)
    _atomic_json(args.output / 'READY_WORKER_COMPLETE.json', {'status': 'COMPLETE', 'mode': MODE,
                 'output_ids': [r['output_id'] for r in outputs], 'ended_unix': time.time(),
                 'target_blind': True, 'gold_loaded': False})


if __name__ == '__main__':
    main()
