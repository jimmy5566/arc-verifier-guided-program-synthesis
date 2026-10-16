#!/usr/bin/env python3
"""CPU-only ORC of frozen pools. No model, selector mutation or generation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

GOLD_SHA256 = '84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(grid):
    return json.dumps(grid, separators=(',', ':'), sort_keys=True)


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def checked(path, relative, hashes):
    data = (path / relative).read_bytes()
    if hashes.get(relative) != digest(data):
        raise RuntimeError('SCORING_INPUT_HASH_MISMATCH:' + str(path / relative))
    return data


def load_pool(path, depth, workers=8):
    ledger = read(path / 'manifests/ARCHIVE_HASHES.json')
    verification = read(path / 'manifests/ARCHIVE_HASH_VERIFICATION.json')
    if ledger.get('status') != 'PASS' or verification.get('status') != 'PASS':
        raise RuntimeError('ARCHIVE_NOT_VERIFIED')
    hashes = ledger['files']
    contract = json.loads(checked(path, 'CONTRACT.json', hashes))
    audit = json.loads(checked(path, 'SOURCE_COMPLETENESS_AUDIT.json', hashes))
    cohort = json.loads(checked(path, 'RUN_COHORT.json', hashes))
    if (int(contract['ttt_depth']) != depth or contract.get('gold_loaded') is not False
            or audit.get('gold_loaded') is not False or audit.get('status') != 'PASS'
            or audit.get('outputs_found') != 89 or audit.get('canonical_aug8_cells_found') != 712):
        raise RuntimeError('ARCHIVE_COHORT_CONTRACT_FAILED')
    outputs = [str(row['output_id']) for row in cohort['outputs']]
    if len(outputs) != 89 or len(set(outputs)) != 89 or len({x.split(':o')[0] for x in outputs}) != 60:
        raise RuntimeError('NOT_EXACT_EVAL60_89')
    def one(oid):
        relative = 'tables/candidate_index/' + oid.replace(':', '_') + '.jsonl.gz'
        data = checked(path, relative, hashes)
        grids = set()
        for line in gzip.decompress(data).splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('output_id') != oid or int(row['depth']) != depth:
                raise RuntimeError('CANDIDATE_IDENTITY_MISMATCH')
            if int(row['checkpoint']) == 1024 and row.get('canonical_grid') is not None:
                grids.add(canonical(row['canonical_grid']))
        return oid, grids, (relative, digest(data))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(one, outputs))
    provenance = {'archive': str(path), 'depth': depth, 'output_count': 89,
                  'archive_ledger_sha256': digest((path / 'manifests/ARCHIVE_HASHES.json').read_bytes()),
                  'scoring_input_hashes': {name: value for _, _, (name, value) in rows},
                  'verification_scope': 'All consumed candidate-index and cohort/contract bytes reverified; reuse prior full raw-archive attestation, no redundant raw/node/weight reads'}
    return outputs, {oid: grids for oid, grids, _ in rows}, provenance


def overlap(left, right):
    return {'shared': len(left & right), 'v7_only_FIX': len(left - right),
            'baseline_only_HARM': len(right - left), 'net': len(left) - len(right),
            'union': len(left | right), 'covers_all_baseline_hits': right <= left,
            'v7_only_output_ids': sorted(left - right), 'baseline_only_output_ids': sorted(right - left)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('v7', 'd24', 'd48', 'solutions', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pools, evidence, outputs = {}, {}, None
    for label, path, depth in [('V7', args.v7, 24), ('BASE_D24', args.d24, 24), ('BASE_D48', args.d48, 48)]:
        ids, pools[label], evidence[label] = load_pool(path, depth)
        if outputs is not None and outputs != ids:
            raise RuntimeError('ARCHIVE_OUTPUT_ORDER_MISMATCH')
        outputs = ids
    # Persist and freeze every consumed candidate identity BEFORE any Gold read.
    write(args.output / 'ORACLE_INPUT_FREEZE.json', {'status': 'FROZEN', 'gold_loaded': False,
          'source_commit': args.source_commit, 'archives': evidence,
          'metric': 'Any exact candidate at R1024; not Top1, Top2, Pass@2 or LB',
          'owner_scope': 'Direct owner request to report ORC now; independent of unfinished B selection, whose frozen code receives no Gold'})
    gold_bytes = args.solutions.read_bytes()
    if int(digest(gold_bytes), 16) != int(GOLD_SHA256, 16):
        raise RuntimeError('DESIGNATED_GOLD_SHA256_MISMATCH')
    gold = json.loads(gold_bytes)
    hits = {name: set() for name in pools}
    for oid in outputs:
        task, index = oid.rsplit(':o', 1)
        answers = gold[task]
        target = answers[int(index)] if isinstance(answers, list) else answers['test'][int(index)]['output']
        key = canonical(target)
        for label, pool in pools.items():
            if key in pool[oid]:
                hits[label].add(oid)
    hits['BASE_D24_D48_UNION'] = hits['BASE_D24'] | hits['BASE_D48']
    hits['V7_PLUS_BASE_UNION'] = hits['V7'] | hits['BASE_D24_D48_UNION']
    tasks = {oid.split(':o')[0] for oid in outputs}
    task_counts = {label: sum(all(oid in found for oid in outputs if oid.split(':o')[0] == task)
                              for task in tasks) for label, found in hits.items()}
    result = {'status': 'COMPLETE', 'source_commit': args.source_commit,
              'ended_utc': datetime.now(timezone.utc).isoformat(), 'cohort': {'tasks': 60, 'outputs': 89},
              'metric': 'Candidate oracle coverage, exposed Eval60 development only; NOT Pass@2',
              'output_hits': {label: len(found) for label, found in hits.items()},
              'task_all_outputs_oracle_hits': task_counts,
              'V7_vs_BASE_D24': overlap(hits['V7'], hits['BASE_D24']),
              'V7_vs_BASE_D24_D48_UNION': overlap(hits['V7'], hits['BASE_D24_D48_UNION']),
              'historical_reference_reproduced': len(hits['BASE_D24']) == 31 and len(hits['BASE_D24_D48_UNION']) == 35,
              'hit_output_ids': {label: sorted(found) for label, found in hits.items()},
              'input_freeze_sha256': digest((args.output / 'ORACLE_INPUT_FREEZE.json').read_bytes()),
              'gold_sha256': digest(gold_bytes), 'gpu_used': False, 'generation_rerun': False,
              'B_selection_mutated': False, 'B_Top2': 'NOT_RUN'}
    write(args.output / 'ORACLE_SCORECARD.json', result)
    write(args.output / 'HASHES.json', {p.name: digest(p.read_bytes()) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps({k: v for k, v in result.items() if k != 'hit_output_ids'}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
