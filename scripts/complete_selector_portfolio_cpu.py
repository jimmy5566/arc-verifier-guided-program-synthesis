"""One fixed CPU rank-fusion check and exact cached-request coverage audit.

Uses frozen Top2 lists, not newly inferred likelihoods. This is exposed-development
analysis, never independent selector validation or a substitute for missing B views.
"""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import io
from pathlib import Path

from scripts.score_v7_frozen_oracle_only import read, write, digest, canonical, GOLD_SHA256
from scripts.score_reused_rankings_cpu import metrics
from scripts.summarize_cpu_selector_completion import paired

SOURCES = ('BASE24_SUPPORT_NLL', 'BASE48_SUPPORT_NLL', 'V7_SUPPORT_NLL')


def fuse(lists):
    candidates = {}
    for source, grids in enumerate(lists):
        seen = set()
        for rank, grid in enumerate(grids[:2], 1):
            key = canonical(grid)
            if key in seen:
                raise RuntimeError('DUPLICATE_WITHIN_SOURCE')
            seen.add(key)
            item = candidates.setdefault(key, {'grid': grid, 'score': Fraction(0), 'sources': set(),
                                               'base_rank': 3, 'key': key})
            item['score'] += Fraction(1, 60 + rank)
            item['sources'].add(source)
            if source == 0:
                item['base_rank'] = rank
    ordered = sorted(candidates.values(), key=lambda item: (
        -item['score'], -len(item['sources']), item['base_rank'], item['key']))
    return [item['grid'] for item in ordered[:2]]


def coverage(rows, units):
    counts = Counter()
    pending_units, pending_outputs = set(), set()
    potential_pending_keys = set()
    for row in rows:
        unit = (row['output_id'], int(row['depth']))
        hit = unit in units
        counts['total_requests'] += 1
        counts['cached_requests' if hit else 'missing_requests'] += 1
        if not hit:
            pending_units.add(unit)
            pending_outputs.add(unit[0])
            view = 'identity' if row['request_kind'] == 'original' else row['scoring_view']
            # Estimate only. Exact token/model/prompt hashes must match before reuse.
            potential_pending_keys.add((row['output_id'], row['grid_key'], view,
                                        row['prefix_tokens'], row['continuation_tokens'], row['eos_tokens']))
    return {**counts, 'pending_source_units': len(pending_units),
            'pending_output_count': len(pending_outputs), 'pending_output_ids': sorted(pending_outputs),
            'potential_distinct_pending_request_keys': len(potential_pending_keys),
            'dedup_estimate_is_verified_token_equivalence': False,
            'no_cached_unit_for_missing_outputs': all(not any(oid == unit[0] for unit in units)
                                                       for oid in pending_outputs),
            'limitation': 'Inventory token lengths and logical keys cannot establish exact token identity or supply unobserved cross-view model probabilities'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('rankings', 'reuse-index', 'inventory', 'historical-s2', 'historical-oracle', 'solutions', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--rankings-sha256', required=True)
    parser.add_argument('--reuse-index-sha256', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.rankings.read_bytes()
    if digest(raw) != args.rankings_sha256:
        raise RuntimeError('FROZEN_RANKINGS_HASH_MISMATCH')
    source = read(args.rankings)['rankings']
    ids = sorted(source[SOURCES[0]])
    if len(ids) != 89 or len({oid.split(':o')[0] for oid in ids}) != 60:
        raise RuntimeError('COHORT_NOT_EVAL60_89')
    if any(set(source[key]) != set(ids) for key in SOURCES):
        raise RuntimeError('SOURCE_COHORT_MISMATCH')
    if digest(args.reuse_index.read_bytes()) != args.reuse_index_sha256:
        raise RuntimeError('REUSE_INDEX_HASH_MISMATCH')
    index = read(args.reuse_index)['index']
    units = {(item['identity']['output_id'], int(item['identity']['depth'])) for item in index.values()}
    inventory = args.inventory.read_bytes()
    audit = coverage(csv.DictReader(io.StringIO(inventory.decode())), units)
    if audit['total_requests'] != 65034:
        raise RuntimeError('SCORING_INVENTORY_IDENTITY_MISMATCH')
    attempts = {oid: fuse([source[key][oid] for key in SOURCES]) for oid in ids}
    frozen = {'rule': 'Equal-weight truncated-source RRF, k=60, Top2 per source, then base-rank and grid-key ties',
              'sources': list(SOURCES), 'rankings': attempts, 'gpu_used': False,
              'source_rankings_sha256': args.rankings_sha256,
              'request_audit': audit, 'inventory_sha256': digest(inventory),
              'exposed_development_only': True, 'no_parameter_sweep': True,
              'not_historical_B_or_S2': True}
    freeze_path = args.output / 'PORTFOLIO_ATTEMPTS_FROZEN.json'
    write(freeze_path, frozen)
    # Ranking is complete and frozen before opening designated evaluation data.
    targets_raw = args.solutions.read_bytes()
    if int(digest(targets_raw), 16) != int(GOLD_SHA256, 16):
        raise RuntimeError('DESIGNATED_GOLD_MISMATCH')
    targets = __import__('json').loads(targets_raw)
    gold = {oid: targets[oid.split(':o')[0]][int(oid.rsplit(':o', 1)[1])] for oid in ids}
    current = metrics(attempts, gold)
    reference = metrics(source['BASE24_SUPPORT_NLL'], gold)
    # Verify historical S2 bytes and its actual 35-output Gold-derived scope.
    old = args.historical_s2
    for name, sha in read(old / 'PRE_GOLD_HASHES.json').items():
        if digest((old / name).read_bytes()) != sha:
            raise RuntimeError('HISTORICAL_S2_PRE_GOLD_HASH_MISMATCH:' + name)
    ledger = read(old / 'HASHES.json')
    historical_bytes = (old / 'OUTPUT_RESULTS.csv').read_bytes()
    if digest(historical_bytes) != ledger['OUTPUT_RESULTS.csv']:
        raise RuntimeError('HISTORICAL_S2_RESULT_HASH_MISMATCH')
    historical = list(csv.DictReader(io.StringIO(historical_bytes.decode())))
    old_hits = [row['output_id'] for row in historical if row['S2_TOP2_HIT'] == 'True']
    old_oracle = args.historical_oracle
    old_oracle_bytes = (old_oracle / 'OUTPUT_RESULTS.csv').read_bytes()
    if digest(old_oracle_bytes) != read(old_oracle / 'HASHES.json')['files']['OUTPUT_RESULTS.csv']:
        raise RuntimeError('HISTORICAL_ORACLE_HASH_MISMATCH')
    oracle_rows = list(csv.DictReader(io.StringIO(old_oracle_bytes.decode())))
    oracle_hits = {row['output_id'] for row in oracle_rows if row['ORC_UNION'].lower() == 'true'}
    if len(historical) != 35 or {row['output_id'] for row in historical} != oracle_hits or len(oracle_rows) != 89:
        raise RuntimeError('HISTORICAL_S2_ORACLE_COHORT_MISMATCH')
    if {row['output_id'] for row in oracle_rows} != set(ids):
        raise RuntimeError('HISTORICAL_ORACLE_COHORT_MISMATCH')
    historical_attempts = read(old / 'S2_ATTEMPTS_FROZEN.json')
    observed_old = metrics({oid: [value[key]['canonical_grid'] for key in ('attempt_1', 'attempt_2')
                                 if key in value and value[key]]
                            for oid, value in historical_attempts.items()}, gold)
    if set(observed_old['Top2_hit_output_ids']) != set(old_hits):
        raise RuntimeError('HISTORICAL_S2_REPLAY_DISAGREEMENT')
    result = {'status': 'COMPLETE_CPU_ONLY', 'rule': frozen['rule'], 'metrics': current,
              'paired_vs_BASE24_SUPPORT_NLL': paired(reference['Top2_hit_output_ids'], current['Top2_hit_output_ids'], ids),
              'historical_S2': {'Top2': len(old_hits), 'scored_outputs': 35, 'full_cohort_denominator': 89,
                                'full_cohort_score_extension_basis': 'All 54 unscored outputs have no correct Base24/Base48 candidate; not full-89 frozen S2 attempt coverage',
                                'Top2_hit_output_ids': sorted(old_hits), 'selection_cohort': 'POSTFREEZE_GOLD_DERIVED_ORACLE_HIT_DIAGNOSTIC',
                                'pregold_hashes_verified': True, 'frozen_attempts_replayed': True,
                                'official_LB': False},
              'paired_vs_historical_S2_score_extension': paired(old_hits, current['Top2_hit_output_ids'], ids),
              'request_audit': audit,
              'frozen_attempts_sha256': digest(freeze_path.read_bytes()),
              'historical_input_hashes': {'S2_results': digest(historical_bytes), 'oracle_results': digest(old_oracle_bytes)},
              'GPU_seconds': 0, 'new_generation': False, 'model_loaded': False,
              'original_B_completed': False, 'promotion': 'DO_NOT_PROMOTE',
              'limitation': 'One fixed exposed-development fusion check, no independent gain claim. Historical S2 cohort was Gold-derived. Missing B probabilities remain missing.'}
    write(args.output / 'RESULT.json', result)
    print({key: result[key] for key in ('status', 'metrics', 'request_audit')}, flush=True)


if __name__ == '__main__':
    main()
