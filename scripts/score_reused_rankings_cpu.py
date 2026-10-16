"""Descriptive CPU scoring of frozen source-NLL and complete cached B units."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
from pathlib import Path

from scripts.score_v7_frozen_oracle_only import checked, read, write, canonical, digest, GOLD_SHA256
from scripts.score_phase3_eval60_union_v1 import aggregate_candidates, ranking, AUG8
from scripts.index_frozen_b_score_reuse import inspect


def source_rankings(root, depth):
    hashes = read(root / 'manifests/ARCHIVE_HASHES.json')['files']
    contract = json.loads(checked(root, 'CONTRACT.json', hashes))
    audit = json.loads(checked(root, 'SOURCE_COMPLETENESS_AUDIT.json', hashes))
    cohort = json.loads(checked(root, 'RUN_COHORT.json', hashes))
    if contract['ttt_depth'] != depth or contract['gold_loaded'] is not False or audit['status'] != 'PASS':
        raise RuntimeError('SOURCE_CONTRACT_INVALID')
    outputs = [x['output_id'] for x in cohort['outputs']]
    if len(outputs) != 89 or len(set(outputs)) != 89:
        raise RuntimeError('COHORT_NOT_89')
    def one(oid):
        name = oid.replace(':', '_') + '.jsonl.gz'
        node_bytes = checked(root, 'tables/node_index/' + name, hashes)
        candidate_bytes = checked(root, 'tables/candidate_index/' + name, hashes)
        scores = {}
        for line in gzip.decompress(node_bytes).splitlines():
            row = json.loads(line)
            n = row['node']
            if isinstance(n, dict) and n.get('node_id') is not None:
                scores[row['augmentation_id'], n['node_id']] = n.get('cumulative_score')
        occurrences = []
        for line in gzip.decompress(candidate_bytes).splitlines():
            row = json.loads(line)
            if int(row['checkpoint']) != 1024 or row.get('canonical_grid') is None:
                continue
            if row['output_id'] != oid or int(row['depth']) != depth:
                raise RuntimeError('CANDIDATE_IDENTITY_MISMATCH')
            nll = scores.get((row['augmentation_id'], row.get('terminal_node_id')))
            if not isinstance(nll, (int, float)) or not math.isfinite(nll):
                raise RuntimeError('MISSING_SOURCE_NLL:' + oid)
            occurrences.append({'grid_key': canonical(row['canonical_grid']), 'canonical_grid': row['canonical_grid'],
                                'depth': depth, 'augmentation_id': row['augmentation_id'], 'nll': nll,
                                'regret': None, 'nodes_expanded_so_far': row.get('nodes_expanded_so_far')})
        aggregated = aggregate_candidates(occurrences)
        return oid, {method: [r['canonical_grid'] for r in ranking(aggregated, method)[:2]]
                     for method in ('NLL', 'SUPPORT_NLL')}
    with ThreadPoolExecutor(max_workers=8) as executor:
        result = dict(executor.map(one, outputs))
    return outputs, result, {key: hashes[key] for oid in outputs for key in
                             ['tables/node_index/' + oid.replace(':', '_') + '.jsonl.gz',
                              'tables/candidate_index/' + oid.replace(':', '_') + '.jsonl.gz']}


def b_top2(payload):
    def tie(row):
        rep = row['stable_representative']
        return (AUG8.index(rep['augmentation_id']), int(rep['candidate_completion_index']),
                int(rep['nodes_expanded_so_far']), str(rep['candidate_id']), row['grid_key'])
    rows = sorted(payload['source_candidates'],
                  key=lambda r: (-(float(r['support_count']) - float(r['mean_view_nll'])), tie(r)))
    return [r['canonical_grid'] for r in rows[:2]]


def metrics(attempts, gold):
    hits1 = {oid for oid, values in attempts.items() if values and canonical(gold[oid]) == canonical(values[0])}
    hits2 = {oid for oid, values in attempts.items() if canonical(gold[oid]) in {canonical(v) for v in values[:2]}}
    tasks = {oid.split(':o')[0] for oid in attempts}
    fully_observed = {task for task in tasks if all(oid in attempts for oid in gold if oid.split(':o')[0] == task)}
    return {'outputs_evaluated': len(attempts), 'Top1': len(hits1), 'Top2': len(hits2),
            'Top2_hit_output_ids': sorted(hits2), 'Top1_hit_output_ids': sorted(hits1),
            'fully_observed_tasks_in_panel': len(fully_observed),
            'task_all_outputs_Top2': sum(all(oid in hits2 for oid in gold if oid.split(':o')[0] == task)
                                        for task in fully_observed),
            'scope': 'Frozen-pool exact selection on exposed Eval60, not competition LB'}


def main():
    parser = argparse.ArgumentParser()
    for name in ('v7', 'd24', 'd48', 'reuse-index', 'solutions', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reuse-index-sha256', required=True)
    parser.add_argument('--cached-only', action='store_true',
                        help='Score already-complete B units without rereading full source node archives')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if digest(args.reuse_index.read_bytes()) != args.reuse_index_sha256:
        raise RuntimeError('REUSE_INDEX_HASH_MISMATCH')
    all_rankings, inputs, outputs = {}, {}, None
    if args.cached_only:
        hashes = read(args.v7 / 'manifests/ARCHIVE_HASHES.json')['files']
        cohort_bytes = checked(args.v7, 'RUN_COHORT.json', hashes)
        outputs = [row['output_id'] for row in json.loads(cohort_bytes)['outputs']]
        if len(outputs) != 89 or len(set(outputs)) != 89:
            raise RuntimeError('COHORT_NOT_89')
        inputs['cohort_sha256'] = digest(cohort_bytes)
    for label, root, depth in ([] if args.cached_only else [('V7', args.v7, 24), ('BASE24', args.d24, 24), ('BASE48', args.d48, 48)]):
        ids, found, inputs[label] = source_rankings(root, depth)
        if outputs is not None and ids != outputs:
            raise RuntimeError('COHORT_MISMATCH')
        outputs = ids
        for method in ('NLL', 'SUPPORT_NLL'):
            all_rankings[label + '_' + method] = {oid: row[method] for oid, row in found.items()}
    cached = {0: {}, 24: {}}
    for unit in read(args.reuse_index)['index'].values():
        path = Path(unit['path'])
        verified, error = inspect(path)
        if error or verified['payload_sha256'] != unit['payload_sha256']:
            raise RuntimeError('B_SHARD_RECHECK_FAILED')
        payload = json.loads(gzip.decompress(path.read_bytes()))
        cached[payload['depth']][payload['output_id']] = b_top2(payload)
    matched = sorted(set(cached[0]) & set(cached[24]))
    for label, depth in [('B_GREEDY_PARTIAL', 0), ('B_DFS_PARTIAL', 24)]:
        all_rankings[label] = {oid: cached[depth][oid] for oid in matched}
    # Fixed portfolio, not fitted using targets: Base24 first, first different
    # V7 NLL candidate second; if unavailable, Base24 second. Never Oracle-rank.
    portfolio = {}
    for oid in ([] if args.cached_only else outputs):
        base, v7 = all_rankings['BASE24_NLL'][oid], all_rankings['V7_NLL'][oid]
        if not base:
            portfolio[oid] = v7[:2]
            continue
        second = next((grid for grid in v7 if canonical(grid) != canonical(base[0])),
                      base[1] if len(base) > 1 else base[0])
        portfolio[oid] = [base[0], second]
    if not args.cached_only:
        all_rankings['FIXED_BASE24_V7_PORTFOLIO'] = portfolio
    write(args.output / 'CPU_RANKINGS_FROZEN.json', {'status': 'FROZEN', 'gpu_used': False,
          'source_input_hashes': inputs, 'reuse_index_sha256': args.reuse_index_sha256,
          'rankings': all_rankings, 'B_full_89_completed': False,
          'policy': 'Historical NLL/support-NLL and historical B formula unchanged. Portfolio prespecified in code before this scoring. Exposed Eval60, descriptive only.'})
    # Only now read designated targets. No GPU/model/inference imports exist.
    data = args.solutions.read_bytes()
    if int(digest(data), 16) != int(GOLD_SHA256, 16):
        raise RuntimeError('DESIGNATED_GOLD_MISMATCH')
    targets, gold = json.loads(data), {}
    for oid in outputs:
        task, index = oid.rsplit(':o', 1)
        gold[oid] = targets[task][int(index)]
    result = {name: metrics(values, gold) for name, values in all_rankings.items()}
    result['MATCHED_PANEL_COMPARISON'] = {name: metrics({oid: all_rankings[name][oid] for oid in matched}, gold)
                                         for name in ['BASE24_NLL', 'BASE48_NLL', 'V7_NLL'] if name in all_rankings}
    result.update(status='COMPLETE_CPU_ONLY', gpu_used=False, generation_rerun=False,
                  B_missing_output_count=89-len(matched), independent_validation=False,
                  full_source_NLL_status='NOT_RUN' if args.cached_only else 'COMPLETE',
                  frozen_rankings_sha256=digest((args.output / 'CPU_RANKINGS_FROZEN.json').read_bytes()))
    write(args.output / 'CPU_REUSED_SCORECARD.json', result)
    print(json.dumps({k: ({a:b for a,b in v.items() if not a.endswith('_ids')} if isinstance(v,dict) and k!='MATCHED_PANEL_COMPARISON' else v)
                      for k,v in result.items() if k!='MATCHED_PANEL_COMPARISON'}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
