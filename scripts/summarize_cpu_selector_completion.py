"""Compare frozen CPU rankings without reading targets or fitting selector parameters."""
import argparse
import random
from pathlib import Path

from scripts.score_v7_frozen_oracle_only import read, write, digest


def paired(reference, treatment, output_ids, draws=10000):
    before, after = set(reference), set(treatment)
    units = sorted(set(output_ids))
    if len(units) != 89 or len({oid.split(':o')[0] for oid in units}) != 60:
        raise RuntimeError('NOT_EVAL60_89')
    if (before | after) - set(units):
        raise RuntimeError('HIT_OUTSIDE_COHORT')
    tasks = sorted({oid.split(':o')[0] for oid in units})
    by_task = {task: [oid for oid in units if oid.split(':o')[0] == task] for task in tasks}
    counts = {task: sum(int(oid in after)-int(oid in before) for oid in ids)
              for task, ids in by_task.items()}
    generator = random.Random(20261016)
    boot = []
    for _ in range(draws):
        sample = generator.choices(tasks, k=60)
        boot.append(sum(counts[task] for task in sample) / sum(len(by_task[task]) for task in sample))
    boot.sort()
    task_score = lambda hits: sum(sum(oid in hits for oid in ids)/len(ids) for ids in by_task.values()) / 60
    return {'FIX_output_ids': sorted(after-before), 'HARM_output_ids': sorted(before-after),
            'FIX': len(after-before), 'HARM': len(before-after), 'net_outputs': len(after)-len(before),
            'reference_Top2': len(before), 'treatment_Top2': len(after),
            'paired_output_rate_delta': (len(after)-len(before))/89,
            'task_cluster_bootstrap_95pct_delta': [boot[int(.025*draws)], boot[min(draws-1, int(.975*draws))]],
            'bootstrap_tasks': 60, 'bootstrap_draws': draws, 'seed': 20261016,
            'reference_task_mean_test_exact_Pass2': task_score(before),
            'treatment_task_mean_test_exact_Pass2': task_score(after),
            'scope': 'Exposed Eval60 frozen-pool descriptive comparison; not official competition score or independent validation'}


def summarize(scorecard, frozen):
    if scorecard['frozen_rankings_sha256'] != digest(frozen.read_bytes()):
        raise RuntimeError('FROZEN_RANKINGS_HASH_MISMATCH')
    rankings = read(frozen)['rankings']
    ids = sorted(rankings['BASE24_NLL'])
    baseline = scorecard['BASE24_NLL']['Top2_hit_output_ids']
    full = {key: value for key, value in scorecard.items()
            if isinstance(value, dict) and value.get('outputs_evaluated') == 89}
    return {'status': 'COMPLETE_CPU_SELECTOR_ALTERNATIVES', 'outputs': 89, 'tasks': 60,
            'metrics': {key: {field: value[field] for field in ('Top1', 'Top2', 'task_all_outputs_Top2')}
                        for key, value in full.items()},
            'paired_vs_BASE24_SOURCE_NLL': {key: paired(baseline, value['Top2_hit_output_ids'], ids)
                                           for key, value in full.items() if key != 'BASE24_NLL'},
            'full_historical_B_status': 'INCOMPLETE_51_OF_89_OUTPUT_PAIRS',
            'hybrid_definition': '51 cached unchanged V7 DFS B outputs; 38 historical source SUPPORT_NLL fallbacks',
            'gold_tuned_selector': False, 'GPU_seconds_this_completion': 0, 'promotion': 'DO_NOT_PROMOTE',
            'limitations': ['Eval60 already exposed', 'Methods are not all historical B',
                            'No missing cross-view probability has been invented',
                            'No official submission or independent validation',
                            'No maximum-over-methods score is a validated selector gain']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scorecard', required=True, type=Path)
    parser.add_argument('--frozen', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('OUTPUT_ALREADY_EXISTS')
    write(args.output, summarize(read(args.scorecard), args.frozen))


if __name__ == '__main__':
    main()
