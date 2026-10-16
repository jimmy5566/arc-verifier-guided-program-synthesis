"""CPU-only exact-shard reuse index. Never loads models, targets or candidates anew."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
from pathlib import Path

KEYS = ('output_id', 'depth', 'candidate_set_sha256', 'candidate_index_sha256',
        'adapter_model_sha256', 'model_identity', 'tokenizer_identity',
        'scoring_context_window', 'scoring_views')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect(path, include_payload=False):
    try:
        data = path.read_bytes()
        payload = json.loads(gzip.decompress(data))
        sidecar = path.with_suffix('').with_suffix('.sha256.json')
        receipt = json.loads(sidecar.read_bytes())
        unsigned = {k: v for k, v in payload.items() if k != 'payload_sha256'}
        if (payload.get('status') != 'COMPLETE' or receipt.get('status') != 'PASS'
                or receipt.get('shard_file_sha256') != sha(data)
                or payload.get('payload_sha256') != sha(canonical(unsigned).encode())
                or receipt.get('payload_sha256') != payload['payload_sha256']
                or any(receipt['provenance'][k] != payload[k] for k in KEYS)):
            raise ValueError('HASH_OR_PROVENANCE_MISMATCH')
        rows, details = payload['source_candidates'], payload['view_likelihood_rows']
        if len(details) != len(rows) * 8 or len(payload['scoring_views']) != 8:
            raise ValueError('INCOMPLETE_VIEWS')
        if any(not math.isfinite(float(row[key])) for row in rows
               for key in ('original_log_likelihood', 'mean_view_nll')):
            raise ValueError('NONFINITE_SCORE')
        result = {'path': str(path), 'shard_sha256': sha(data), 'payload_sha256': payload['payload_sha256'],
                'identity': {k: payload[k] for k in KEYS}, 'candidate_count': len(rows),
                'view_score_rows': len(details)}
        if include_payload:
            result['_payload'] = payload
        return result, None
    except (ValueError, KeyError, OSError, EOFError) as exc:
        return None, {'path': str(path), 'reason': str(exc)}


def build(roots, workers=8):
    paths = [p for root in roots for p in sorted((root / 'SCORING_SHARDS').glob('*.json.gz'))]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        checked = list(executor.map(inspect, paths))
    index, rejected, conflicts, copies = {}, [], [], 0
    for row, error in checked:
        if error:
            rejected.append(error)
            continue
        key = sha(canonical(row['identity']).encode())
        previous = index.get(key)
        if previous and previous['payload_sha256'] != row['payload_sha256']:
            conflicts.append({'identity_key': key, 'left': previous['path'], 'right': row['path']})
            continue
        copies += int(previous is not None)
        index[key] = row
    conflicted = {row['identity_key'] for row in conflicts}
    for key in conflicted:
        index.pop(key, None)
    coverage = {(r['identity']['output_id'], r['identity']['depth']) for r in index.values()}
    return {'status': 'INDEXED_NO_GPU', 'source_files_read': len(paths),
            'reusable_source_units': len(index), 'duplicate_identical_copies': copies,
            'complete_output_pairs': sum((oid, 0) in coverage and (oid, 24) in coverage for oid in {x[0] for x in coverage}),
            'conflicts_not_reused': conflicts, 'rejected_not_reused': rejected,
            'index': index, 'gold_loaded': False, 'gpu_used': False,
            'source_modified_or_deleted': False,
            'scope': 'Exact complete B shards only. Source-view NLL is separate; no missing-view imputation or bulk backfill.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = build(args.source)
    raw = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode()
    (args.output / 'REUSE_INDEX.json').write_bytes(raw)
    summary = {k: v for k, v in result.items() if k != 'index'}
    summary['index_sha256'] = sha(raw)
    summary['index_path'] = str(args.output / 'REUSE_INDEX.json')
    (args.output / 'REUSE_SUMMARY.json').write_text(json.dumps(summary, sort_keys=True, indent=2) + '\n')
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
