"""Post-freeze cached sequence-NLL diagnostic. No new model calls or ranking.

Correct-candidate availability selects this panel; it is NOT an all-output or
causal V7 evaluation. Missing token probabilities/ranks/regret remain unknown.
"""
import argparse
import gzip
import json
import math
from pathlib import Path

from scripts.score_v7_frozen_oracle_only import read, write, digest, checked, canonical, GOLD_SHA256
from scripts.analyze_frozen_eval60_search_cpu import task_interval

VIEWS = {'identity', 'flip_ud', 'transpose', 'anti_transpose', 'rot90', 'rot180', 'rot270', 'flip_lr'}


def collect_rows(rows, depth=24):
    result = {}
    for row in rows:
        if int(row['depth']) != depth:
            continue
        count = int(row['token_count'])
        nll = float(row['sequence_negative_log_likelihood'])
        if (count <= 0 or not math.isfinite(nll) or nll < 0
                or not math.isfinite(float(row['mean_token_nll']))
                or not math.isclose(nll, float(row['mean_token_nll'])*count, rel_tol=1e-8, abs_tol=1e-6)
                or not math.isclose(nll, -float(row['sequence_log_likelihood']), rel_tol=1e-8, abs_tol=1e-6)):
            raise RuntimeError('INVALID_CACHED_SEQUENCE_SCORE')
        key = (row['output_id'], row['grid_key'])
        view = row['scoring_view']
        if view not in VIEWS or view in result.setdefault(key, {}):
            raise RuntimeError('INVALID_OR_DUPLICATE_VIEW')
        result[key][view] = row
    return result


def pair_gold(left, right, anatomy):
    if set(left) != VIEWS or set(right) != VIEWS:
        return [], 'MISSING_CACHED_VIEW'
    paired = []
    for view in sorted(VIEWS):
        a, b = left[view], right[view]
        aug = 'geom='+view+'__color=id__order=canonical'
        source_a, source_b = anatomy['modes']['base'], anatomy['modes']['v7']
        prompt = source_a['prompt_hashes'].get(aug)
        if not prompt or prompt != source_b['prompt_hashes'].get(aug):
            return [], 'ARCHIVED_PROMPT_MISMATCH'
        for field in ('prefix_tokens', 'continuation_tokens', 'eos_tokens', 'token_count', 'tokenizer_identity', 'scoring_context_window'):
            if a.get(field) is None or a[field] != b.get(field):
                return [], 'CACHED_CONTEXT_OR_TOKENIZER_MISMATCH:'+field
        if any(int(row['prefix_tokens']) != source['cells'][aug]['root_length']
               or int(row['eos_tokens']) != 1
               or int(row['continuation_tokens'])+1 != int(row['token_count'])
               or int(row['prefix_tokens'])+int(row['token_count']) > int(row['scoring_context_window'])
               for row, source in ((a, source_a), (b, source_b))):
            return [], 'CONTEXT_OR_EOS_CONTRACT_MISMATCH'
        paired.append({'view': view, 'token_count_including_eos': a['token_count'],
                       'base_sequence_nll': a['sequence_negative_log_likelihood'],
                       'v7_sequence_nll': b['sequence_negative_log_likelihood'],
                       'V7_minus_Base_mean_token_nll': b['mean_token_nll']-a['mean_token_nll'],
                       'V7_minus_Base_sequence_nll': b['sequence_negative_log_likelihood']-a['sequence_negative_log_likelihood']})
    return paired, 'MATCHED_RECORDED_CONTEXT_AND_ARCHIVED_PROMPT'


def frozen_rows(root):
    ledger = read(root/'PRE_GOLD_HASHES.json')
    freeze = read(root/'PRE_GOLD_FREEZE.json')
    if freeze.get('status') != 'PASS' or freeze.get('gold_loaded') is not False:
        raise RuntimeError('CACHE_NOT_PRE_GOLD_FROZEN')
    name = 'B_VIEW_LIKELIHOOD.jsonl.gz'
    raw = checked(root, name, ledger)
    rows = collect_rows(json.loads(line) for line in gzip.decompress(raw).splitlines() if line)
    return rows, {'root': str(root), 'view_rows_file_sha256': digest(raw),
                  'freeze_sha256': digest((root/'PRE_GOLD_FREEZE.json').read_bytes()),
                  'ledger_sha256': digest((root/'PRE_GOLD_HASHES.json').read_bytes())}


def main():
    parser = argparse.ArgumentParser()
    for name in ('base-cache', 'v7-cache', 'anatomy', 'solutions', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    left, li = frozen_rows(args.base_cache); right, ri = frozen_rows(args.v7_cache)
    anatomy_raw = (args.anatomy/'RESULT.json').read_bytes(); anatomy = json.loads(anatomy_raw)
    ids = sorted(anatomy['checkpoint_manifest'])
    if len(ids) != 89 or len({x.split(':o')[0] for x in ids}) != 60:
        raise RuntimeError('NOT_EVAL60_89')
    write(args.output/'CACHE_INPUTS_FROZEN.json', {'base': li, 'v7': ri, 'anatomy_sha256': digest(anatomy_raw),
          'GPU_used': False, 'estimand': 'All-view cached Gold sequence NLL conditional on Gold candidate availability in both systems',
          'no_new_inference_or_selector_change': True})
    target_raw = args.solutions.read_bytes()
    if int(digest(target_raw),16) != int(GOLD_SHA256,16): raise RuntimeError('DESIGNATED_GOLD_MISMATCH')
    targets = json.loads(target_raw); outputs = []; mean_rows = []; seq_rows = []; available = {'base': 0, 'v7': 0}
    for oid in ids:
        task, index = oid.rsplit(':o',1); key = (oid, canonical(targets[task][int(index)]))
        a, b = left.get(key,{}), right.get(key,{})
        available['base'] += int(set(a)==VIEWS); available['v7'] += int(set(b)==VIEWS)
        record = {'output_id': oid, 'base_views': len(a), 'v7_views': len(b)}
        if set(a)==VIEWS and set(b)==VIEWS:
            item = anatomy['checkpoint_manifest'][oid]; raw = (args.anatomy/item['file']).read_bytes()
            if digest(raw) != item['sha256']: raise RuntimeError('ANATOMY_CHECKPOINT_HASH_MISMATCH')
            pairs, status = pair_gold(a, b, json.loads(gzip.decompress(raw)))
            record.update(status=status, paired_views=pairs)
            if pairs:
                mean_rows.append((oid, sum(x['V7_minus_Base_mean_token_nll'] for x in pairs)/8))
                seq_rows.append((oid, sum(x['V7_minus_Base_sequence_nll'] for x in pairs)/8))
        else: record['status'] = 'GOLD_CANDIDATE_NOT_CACHED_IN_BOTH_SYSTEMS'
        outputs.append(record)
    scope = 'Candidate-availability-conditioned full-sequence cached NLL, output means then task-cluster bootstrap; exposed NONBLIND diagnostic'
    intervals = {}
    for name, rows in [('mean_token_nll',mean_rows),('sequence_nll',seq_rows)]:
        intervals[name] = {**task_interval(rows), 'scope': scope, 'lower_delta_is_better_for_V7': True}
    write(args.output/'RESULT.json', {'status':'COMPLETE_CPU_CACHED_GOLD_SEQUENCE_NLL', 'outputs_audited':89,
          'fully_cached_gold_outputs':available, 'matched_output_count':len(mean_rows), 'matched_view_pairs':8*len(mean_rows),
          'paired_V7_minus_Base':intervals, 'outputs':outputs, 'inputs':{'base':li,'v7':ri},
          'GPU_used':False, 'model_loaded':False, 'new_generation':False, 'selector_changed':False,
          'all_89_output_Gold_path':'NOT_RUN', 'token_rank_and_regret':'NOT_RECOVERABLE_FROM_SCALAR_NLL',
          'limitations':['Gold-candidate survivorship, not an all-output preference estimate',
              'TTT parameterization differs; not a causal V7-initialization comparison',
              'B cache records context lengths/tokenizer identity, not per-request token SHA; archived generation prompt hashes checked',
              'Inherited source/runtime provenance limitations unchanged; no independent or LB claim']})
    print('CACHED_GOLD_SEQUENCE_DIAGNOSTIC_COMPLETE',flush=True)


if __name__ == '__main__': main()
