"""CPU-only final B scorecard, reusing frozen ORC and prior comparator scores."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess

from scripts.score_v7_frozen_oracle_only import read, write, digest, checked, canonical, GOLD_SHA256
from scripts.score_reused_rankings_cpu import metrics
from scripts.summarize_cpu_selector_completion import paired

INDEX_SHA = 'b701e8e8f951196494e36064dd09a2b15ce51d162a7ad43acb3be9a1fecc8016'
RANKINGS_SHA = '161defb944b5c238319abc7810b567d6b956a2bdbf556a27221a1aaef3c222c2'
REFERENCE_SHA = '09e2c2f2ae1cee0c9eb0db4ff7817875fadc269fcec64036a22ca277812e46a5'
ORACLE_SHA = '71373776ff14aa00b375b6123cbbf8ce6e8b6350619b1303793f6f7add34243a'
HISTORICAL_SHA = 'c591343483e413a8172150086a5723614b21f2d9536188d143ca6c009eadcb84'


def bound_json(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise RuntimeError('REFERENCE_HASH_MISMATCH:' + str(path))
    return json.loads(raw)


def verify_freeze(root, required):
    freeze = read(root / 'PRE_GOLD_FREEZE.json')
    ledger = read(root / 'PRE_GOLD_HASHES.json')
    if (freeze.get('status') != 'PASS' or freeze.get('gold_loaded') is not False
            or freeze.get('B_attempts_frozen') is not True or freeze.get('output_count') != 89
            or freeze.get('task_count') != 60 or freeze.get('source_units') != 178
            or freeze.get('scoring_views_complete') != '8/8'
            or freeze.get('view_rows') != freeze.get('candidate_count', -1)*8):
        raise RuntimeError('FULL_B_FREEZE_CONTRACT_FAILED')
    verification = freeze.get('hash_verification', {})
    if (verification.get('status') != 'PASS' or verification.get('expected') != ledger
            or verification.get('observed') != ledger):
        raise RuntimeError('B_FREEZE_LEDGER_RECEIPT_FAILED')
    reuse = json.loads(checked(root, 'SHARD_REUSE_RECEIPT.json', ledger))
    required_map = {(r['identity']['output_id'], int(r['identity']['depth'])): r['payload_sha256']
                    for r in required['index'].values()}
    actual = {(r['output_id'], int(r['depth'])): r['payload_sha256'] for r in reuse['units']}
    if (len(required_map) != 102 or actual != required_map or reuse.get('reused_count') != 102
            or reuse.get('status') != 'PASS' or reuse.get('source_modified') is not False):
        raise RuntimeError('REQUIRED_102_CACHE_REUSE_NOT_ESTABLISHED')
    attempts_raw = checked(root, 'B_ATTEMPTS_FROZEN.json', ledger)
    return freeze, reuse, json.loads(attempts_raw), digest(attempts_raw)


def decode_attempts(value, output_ids):
    if set(value) != set(output_ids):
        raise RuntimeError('B_ATTEMPT_COHORT_MISMATCH')
    result = {}
    for oid, row in value.items():
        result[oid] = []
        for key in ('attempt_1', 'attempt_2'):
            candidate = row[key]
            if canonical(candidate['canonical_grid']) != candidate['grid_key']:
                raise RuntimeError('GRID_KEY_MISMATCH:' + oid)
            result[oid].append(candidate['canonical_grid'])
    return result


def same_top2(first, second):
    unique = lambda grids: list(dict.fromkeys(canonical(grid) for grid in grids[:2]))
    return unique(first) == unique(second)


def enrich_metric(value, ids):
    by_task = defaultdict(list)
    for oid in ids:
        by_task[oid.split(':o')[0]].append(oid)
    hits1, hits2 = set(value['Top1_hit_output_ids']), set(value['Top2_hit_output_ids'])
    value['output_Top1_rate'] = len(hits1)/len(ids)
    value['output_Top2_rate'] = len(hits2)/len(ids)
    for label, hits in [('Top1', hits1), ('Top2', hits2)]:
        task_scores = [sum(oid in hits for oid in group)/len(group) for group in by_task.values()]
        value['task_mean_test_exact_' + label] = sum(task_scores)/len(task_scores)
        rng = random.Random(20261016)
        draws = sorted(sum(rng.choices(task_scores, k=60))/60 for _ in range(10000))
        value['task_bootstrap_95pct_' + label] = [draws[250], draws[9750]]
    return value


def main():
    parser = argparse.ArgumentParser()
    for name in ('b-freeze', 'archive', 'required-index', 'old-rankings', 'reference-score',
                 'oracle-score', 'historical-result', 'solutions', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    required = bound_json(args.required_index, INDEX_SHA)
    freeze, reuse, frozen_attempts, attempts_sha = verify_freeze(args.b_freeze, required)
    archive_ledger = read(args.archive / 'manifests/ARCHIVE_HASHES.json')['files']
    cohort = json.loads(checked(args.archive, 'RUN_COHORT.json', archive_ledger))
    ids = sorted(row['output_id'] for row in cohort['outputs'])
    if len(ids) != 89 or len(set(ids)) != 89 or len({oid.split(':o')[0] for oid in ids}) != 60:
        raise RuntimeError('NOT_EXACT_EVAL60_89')
    if set(frozen_attempts) != {'dfs', 'greedy'}:
        raise RuntimeError('B_SOURCE_LABEL_MISMATCH')
    attempts = {label: decode_attempts(frozen_attempts[label], ids) for label in ('dfs', 'greedy')}
    old = bound_json(args.old_rankings, RANKINGS_SHA)['rankings']
    for label in ('dfs', 'greedy'):
        prior = old['B_' + label.upper() + '_PARTIAL']
        if len(prior) != 51 or any(not same_top2(grids, attempts[label][oid]) for oid, grids in prior.items()):
            raise RuntimeError('REUSED_51_OUTPUT_SELECTION_DRIFT:' + label)
    # This receipt is written before designated target access. No weights or archives are rescored.
    input_receipt = {'status': 'VERIFIED_FROZEN_INPUTS', 'attempts_sha256': attempts_sha,
                     'B_freeze_sha256': digest((args.b_freeze/'PRE_GOLD_FREEZE.json').read_bytes()),
                     'B_ledger_sha256': digest((args.b_freeze/'PRE_GOLD_HASHES.json').read_bytes()),
                     'required_cache_units_reused': 102, 'cached_51_output_pair_attempt_parity': 'PASS',
                     'complete_outputs': 89, 'complete_units': 178,
                     'archive_full_rehash': False, 'ORC_rescored': False, 'GPU_used': False}
    write(args.output/'INPUT_VERIFICATION_BEFORE_TARGETS.json', input_receipt)
    targets_raw = args.solutions.read_bytes()
    if int(digest(targets_raw), 16) != int(GOLD_SHA256, 16):
        raise RuntimeError('DESIGNATED_GOLD_MISMATCH')
    targets = json.loads(targets_raw)
    gold = {oid: targets[oid.split(':o')[0]][int(oid.rsplit(':o', 1)[1])] for oid in ids}
    scored = {label: enrich_metric(metrics(value, gold), ids) for label, value in attempts.items()}
    reference = bound_json(args.reference_score, REFERENCE_SHA)
    oracle = bound_json(args.oracle_score, ORACLE_SHA)
    historical = bound_json(args.historical_result, HISTORICAL_SHA)['historical_S2']
    dfs_hits, greedy_hits = scored['dfs']['Top2_hit_output_ids'], scored['greedy']['Top2_hit_output_ids']
    contrasts = {'DFS_B_vs_GREEDY_B': paired(greedy_hits, dfs_hits, ids),
                 'DFS_B_vs_V7_SUPPORT_NLL': paired(reference['V7_SUPPORT_NLL']['Top2_hit_output_ids'], dfs_hits, ids),
                 'DFS_B_vs_BASE24_SUPPORT_NLL': paired(reference['BASE24_SUPPORT_NLL']['Top2_hit_output_ids'], dfs_hits, ids),
                 'DFS_B_vs_historical_S2_score_extension': paired(historical['Top2_hit_output_ids'], dfs_hits, ids)}
    oracle_hits = set(oracle['hit_output_ids']['V7'])
    if len(oracle_hits) != 30 or not set(dfs_hits) <= oracle_hits:
        raise RuntimeError('FROZEN_ORACLE_SCORE_DISAGREEMENT')
    pipeline = read(args.b_freeze.parent/'PIPELINE_STATE.json')
    started = datetime.fromisoformat(pipeline['started_utc'])
    ended = datetime.fromisoformat(pipeline['ended_utc'])
    result = {'status': 'COMPLETE_FULL_89_B_POSTFREEZE_CPU_SCORE', 'metrics': scored,
              'paired_contrasts': contrasts, 'input_verification': input_receipt,
              'candidate_oracle_reused': oracle['output_hits'],
              'DFS_B_oracle_to_Top2': len(dfs_hits)/30,
              'DFS_correct_candidate_not_selected_output_ids': sorted(oracle_hits-set(dfs_hits)),
              'historical_S2_reference': historical,
              'GPU_continuation': {'source_commit': pipeline['execution_source_commit'],
                                   'started_utc': pipeline['started_utc'], 'ended_utc': pipeline['ended_utc'],
                                   'process_wall_seconds': (ended-started).total_seconds(),
                                   'GPU_useful_seconds': 'UNVERIFIED_NOT_IDENTICAL_TO_PROCESS_WALL',
                                   'oom_batch_fallbacks': freeze['oom_batch_fallback_count'],
                                   'cached_source_units': reuse['reused_count'], 'new_source_units': 76,
                                   'cached_requests': 39771, 'new_requests': 25263},
              'analysis_source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'scored_at_utc': datetime.now(timezone.utc).isoformat(),
              'decision': 'DO_NOT_PROMOTE_FROM_EXPOSED_EVAL60',
              'limitations': ['Exposed Eval60 development, not independent validation or Kaggle LB',
                              'Historical S2 uses Base24+Base48 and a Gold-derived 35-output scoring cohort; score extension is not full-89 attempt coverage',
                              'No new model inference, generation, training, ORC archive scoring, or Gold-path forward in this CPU analysis']}
    write(args.output/'FINAL_B_SCORECARD.json', result)
    print(json.dumps({'status': result['status'], 'metrics': {
        label: {k:v for k,v in values.items() if not k.endswith('_ids')}
        for label,values in scored.items()}}), flush=True)


if __name__ == '__main__':
    main()
