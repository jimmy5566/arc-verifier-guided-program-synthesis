"""Nonblind CPU anatomy of existing Base24/V7 trees. No model calls or tuning.

Only observed Gold prefixes have token probabilities. Missing prefixes and
historical frontier telemetry are explicitly unknown, never reconstructed logits.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from inference.nvarc_native_augmentation import NativeAugmentation
from inference.arc_native_io import ARCNativeInputAdapter
from scripts.score_v7_frozen_oracle_only import read, write, digest, checked, canonical, GOLD_SHA256

LEGAL = set(range(11)) | {15}
sys.setrecursionlimit(10000)


def gold_tokens(grid, geometry):
    transformed = NativeAugmentation(geometry=geometry).transform_grid(grid).astype(int).tolist()
    return tuple(10 if char == '\n' else int(char) for char in ARCNativeInputAdapter.serialize_grid(transformed)) + (15,)


def gold_node_lengths(nodes, tokens):
    by_id = {int(node['node_id']): node for node in nodes}
    if len(by_id) != len(nodes):
        raise RuntimeError('DUPLICATE_NODE_ID')
    visiting = set()
    @lru_cache(None)
    def length(node_id):
        if node_id in visiting or node_id not in by_id:
            raise RuntimeError('BROKEN_OR_CYCLIC_NODE_ANCESTRY')
        visiting.add(node_id)
        node = by_id[node_id]; parent = node.get('parent_node_id'); token = node.get('selected_token')
        if parent is None:
            value = 0 if token is None else None
        else:
            previous = length(int(parent))
            value = previous+1 if previous is not None and previous < len(tokens) and token == tokens[previous] else None
        visiting.remove(node_id)
        return value
    lengths = {node_id: length(node_id) for node_id in by_id}
    return by_id, lengths


def prefix_hashes(tokens):
    h = hashlib.sha256(); result = {}
    for pos, token in enumerate(tokens, 1):
        if pos > 1:
            h.update(b',')
        h.update(str(token).encode('ascii')); result[h.hexdigest()] = pos
    return result


def analyze_cell(cell, tokens):
    nodes = cell.get('nodes', []); by_id, lengths = gold_node_lengths(nodes, tokens)
    grouped = defaultdict(list)
    for node_id, length in lengths.items():
        if length is not None:
            grouped[length].append(by_id[node_id])
    trace = cell.get('diagnostic_trace') or {}; hash_positions = prefix_hashes(tokens)
    events = defaultdict(list)
    for event in trace.get('search_trace', []):
        pos = hash_positions.get(event.get('prefix_hash'))
        if pos is not None:
            events[pos].append(event)
    # Fallback to complete sibling sets when the older archive lacks probability telemetry.
    branches = defaultdict(dict)
    for node in nodes:
        parent = node.get('parent_node_id'); token = node.get('selected_token'); lp = node.get('token_logprob')
        if parent is not None and lengths.get(int(parent)) is not None and token in LEGAL and isinstance(lp, (int,float)) and math.isfinite(lp):
            branches[int(parent)].setdefault(int(token), float(lp))
    for row in trace.get('branch_probabilities', []):
        parent = row.get('parent_node_id')
        if parent not in lengths or lengths[parent] is None:
            continue
        if row.get('prefix_length') != lengths[parent]:
            raise RuntimeError('BRANCH_PREFIX_LENGTH_MISMATCH')
        values = {int(v['token_id']): float(v['logprob']) for v in row.get('full_arc_logprobs', [])}
        if set(values) == LEGAL and all(math.isfinite(v) for v in values.values()):
            branches[parent] = values
    observations = {}
    for parent in sorted(branches):
        pos = lengths.get(parent); values = branches[parent]
        if pos is None or pos >= len(tokens) or tokens[pos] not in values or pos in observations:
            continue
        lp = values[tokens[pos]]
        ranked = sorted(values, key=lambda token: (-values[token],token)) if set(values) == LEGAL else []
        observations[pos] = {'position': pos, 'parent_node_id': parent, 'gold_logprob': lp,
                             'rank_among_12_legal_tokens': ranked.index(tokens[pos])+1 if ranked else None,
                             'gap_to_best_legal': max(values.values())-lp if ranked else None,
                             'complete_legal_probabilities': bool(ranked)}
    states = []; first_halt = None
    for pos in range(1,len(tokens)+1):
        ns, es = grouped.get(pos,[]), events.get(pos,[])
        if any(n.get('state')=='completed' for n in ns) or any(e.get('candidate_completed') for e in es):
            state = 'COMPLETED'
        elif any(n.get('state')=='expanded' or n['node_id'] in branches for n in ns) or any(e.get('frontier_pop_order') is not None for e in es):
            state = 'EXPANDED'
        elif any(e.get('frontier_insert_order') is not None for e in es) or any(n.get('state') not in {'pruned','completed'} for n in ns):
            state = 'RETAINED_NOT_EXPANDED'
        elif any(n.get('state')=='pruned' for n in ns) or any(e.get('prune_reason') for e in es):
            state = 'PRUNED'
        else:
            state = 'UNOBSERVED_OR_CENSORED'
        if first_halt is None and state in {'PRUNED','RETAINED_NOT_EXPANDED','UNOBSERVED_OR_CENSORED'}:
            first_halt = {'position': pos, 'state': state,
                          'reasons': sorted({str(n['prune_reason']) for n in ns if n.get('prune_reason')} |
                                            {str(e['prune_reason']) for e in es if e.get('prune_reason')})}
        # Production mutates the insertion event on pop, overwriting elapsed_seconds.
        # A single row carrying both counters has no recoverable insertion clock.
        inserted = [e for e in es if e.get('frontier_insert_order') is not None
                    and e.get('frontier_pop_order') is None]
        popped = [e for e in es if e.get('frontier_pop_order') is not None]
        ins_times = [e['elapsed_seconds'] for e in inserted if isinstance(e.get('elapsed_seconds'),(int,float))]
        pop_times = [e['elapsed_seconds'] for e in popped if isinstance(e.get('elapsed_seconds'),(int,float))]
        wait = min(pop_times)-min(ins_times) if ins_times and pop_times and min(pop_times)>=min(ins_times) else None
        # Insert and pop ranks are different counters: never subtract them as a wait.
        states.append({'position': pos, 'state': state, 'wait_seconds': wait,
                       'node_ids': [n['node_id'] for n in ns]})
    checkpoints = {}
    for budget in (128,256,512,1024):
        eligible = [(pos,e) for pos, es in events.items() for e in es
                    if isinstance(e.get('nodes_expanded_so_far'),int) and e['nodes_expanded_so_far']<=budget]
        checkpoints[str(budget)] = {'telemetry_available': bool(trace.get('search_trace')),
            'max_observed_gold_successor_length': max((pos for pos,e in eligible),default=None),
            'max_popped_gold_prefix_length': max((pos for pos,e in eligible if e.get('frontier_pop_order') is not None),default=None)}
    roots = [n for n in nodes if n.get('parent_node_id') is None and n.get('selected_token') is None]
    return {'nodes_expanded': cell.get('nodes_expanded'), 'termination_reason': cell.get('termination_reason'),
            'budget_exhausted': cell.get('budget_exhausted'), 'wall_censored': cell.get('wall_time_censored'),
            'completed_candidates': cell.get('completed_candidates'), 'first_halt': first_halt,
            'gold_length_including_eos': len(tokens), 'observed_probability_prefixes': len(observations),
            'root_length': roots[0].get('token_position') if len(roots)==1 else None,
            'frontier_trace_available': bool(trace.get('search_trace')),
            'frontier_floor_flagged_nodes': sum(bool(n.get('frontier_floor_activated')) for n in nodes),
            'states': states, 'probabilities': list(observations.values()), 'checkpoints': checkpoints}


def archive_info(root):
    ledger = read(root/'manifests/ARCHIVE_HASHES.json')
    if ledger.get('status')!='PASS' or read(root/'manifests/ARCHIVE_HASH_VERIFICATION.json').get('status')!='PASS':
        raise RuntimeError('ARCHIVE_NOT_VERIFIED')
    files = ledger['files']; contract=json.loads(checked(root,'CONTRACT.json',files))
    if contract['ttt_depth']!=24 or contract.get('gold_loaded') is not False:
        raise RuntimeError('FROZEN_ARCHIVE_CONTRACT_FAILED')
    audit=json.loads(checked(root,'SOURCE_COMPLETENESS_AUDIT.json',files))
    if audit.get('status')!='PASS' or audit.get('gold_loaded') is not False or audit.get('outputs_found')!=89 or audit.get('canonical_aug8_cells_found')!=712:
        raise RuntimeError('ARCHIVE_COMPLETENESS_FAILED')
    cohort=json.loads(checked(root,'RUN_COHORT.json',files))
    return files, [row['output_id'] for row in cohort['outputs']]


def task_interval(rows):
    if not rows:
        return {'status':'UNOBSERVED'}
    groups=defaultdict(list)
    for oid,value in rows:
        groups[oid.split(':o')[0]].append(value)
    values=[sum(xs)/len(xs) for xs in groups.values()]
    rng=random.Random(20261016); draws=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(10000))
    return {'tasks':len(values),'mean_task_delta':sum(values)/len(values),'task_bootstrap_95pct':[draws[250],draws[9750]],
            'scope':'Common observed-prefix survivor-conditioned NONBLIND diagnostic, not whole-Gold-path model preference'}


def main():
    parser=argparse.ArgumentParser()
    for name in ('base','v7','solutions','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    roots={'base':args.base,'v7':args.v7};infos={key:archive_info(root) for key,root in roots.items()}
    ids=infos['base'][1]
    if len(ids)!=89 or len(set(ids))!=89 or len({oid.split(':o')[0] for oid in ids})!=60 or infos['v7'][1]!=ids:
        raise RuntimeError('NOT_SAME_EVAL60_89')
    write(args.output/'ANALYSIS_PLAN_FROZEN.json',{'cohort':ids,'GPU_used':False,'new_inference':False,
          'question':'Observed Gold-prefix preferences, search censoring and frontier evidence availability',
          'archive_ledger_sha256':{mode:digest((root/'manifests/ARCHIVE_HASHES.json').read_bytes()) for mode,root in roots.items()},
          'limits':'No missing-logit imputation; no selector or traversal changes; no independent generalization claim'})
    target_bytes=args.solutions.read_bytes()
    if int(digest(target_bytes),16)!=int(GOLD_SHA256,16):raise RuntimeError('DESIGNATED_TARGET_HASH_MISMATCH')
    targets=json.loads(target_bytes);checkpoints=args.output/'checkpoints';checkpoints.mkdir()
    def one(oid):
        task,index=oid.rsplit(':o',1);answers=targets[task]
        gold=answers[int(index)] if isinstance(answers,list) else answers['test'][int(index)]['output']
        result={'output_id':oid,'modes':{},'input_hashes':{}}
        for mode,root in roots.items():
            files=infos[mode][0];name='tables/candidate_index/'+oid.replace(':','_')+'.jsonl.gz'
            rows=[json.loads(line) for line in gzip.decompress(checked(root,name,files)).splitlines() if line]
            sources={row['source_archive_file'] for row in rows}
            if len(sources)!=1:raise RuntimeError('RAW_SOURCE_AMBIGUOUS:'+oid)
            source=sources.pop()
            if Path(source).is_absolute() or '..' in Path(source).parts:raise RuntimeError('UNSAFE_RAW_SOURCE')
            raw=checked(root,source,files);payload=json.loads(gzip.decompress(raw))
            cells=payload['cells'];cells=list(cells.values()) if isinstance(cells,dict) else cells
            if len(cells)!=8 or len({cell['augmentation_id'] for cell in cells})!=8:raise RuntimeError('MISSING_OR_DUPLICATE_AUG8_CELL:'+oid)
            output=payload.get('output') or {};runtime=payload.get('runtime_identity') or {}
            result['input_hashes'][mode]={'raw_source':source,'sha256':digest(raw),'candidate_index_sha256':files[name]}
            result['modes'][mode]={'cells':{},'prompt_hashes':output.get('canonical_aug8_prompt_sha256') or {},
                'adapter_config':runtime.get('adapter_config'),'tokenizer_identity':runtime.get('tokenizer_identity')}
            for cell in cells:
                aug=cell['augmentation_id'];geometry=next(part[5:] for part in aug.split('__') if part.startswith('geom='))
                result['modes'][mode]['cells'][aug]=analyze_cell(cell,gold_tokens(gold,geometry))
            del payload,cells,rows,raw
        if set(result['modes']['base']['cells'])!=set(result['modes']['v7']['cells']):raise RuntimeError('VIEW_IDENTITY_MISMATCH')
        raw=(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n').encode()
        path=checkpoints/(oid.replace(':','_')+'.json.gz');path.write_bytes(gzip.compress(raw,mtime=0))
        print('CPU_ANATOMY_COMPLETE '+oid,flush=True)
        return result, {'file':str(path.relative_to(args.output)),'sha256':digest(path.read_bytes())}
    summary={key:{'termination_reasons':Counter(),'first_halt_states':Counter(),'frontier_trace_cells':0,'cells':0,
                  'observed_probability_prefixes':0,'configuration_variants':Counter()}
             for key in roots}
    paired_lp=[];paired_rank=[];prompt_match=0;tokenizer_match=0;pair_prefixes=0;manifest={};config_diff_outputs=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
        for result,identity in executor.map(one,ids):
            oid=result['output_id'];manifest[oid]=identity
            base,v7=result['modes']['base'],result['modes']['v7']
            if base['adapter_config']!=v7['adapter_config']:config_diff_outputs.append(oid)
            local_lp=[];local_rank=[]
            for mode,data in result['modes'].items():
                summary[mode]['configuration_variants'][canonical(data['adapter_config'])]+=1
                for cell in data['cells'].values():
                    s=summary[mode];s['cells']+=1;s['termination_reasons'][str(cell['termination_reason'])]+=1
                    s['first_halt_states'][cell['first_halt']['state'] if cell['first_halt'] else 'NO_HALT_OBSERVED']+=1
                    s['frontier_trace_cells']+=int(cell['frontier_trace_available']);s['observed_probability_prefixes']+=cell['observed_probability_prefixes']
            for aug,bcell in base['cells'].items():
                vcell=v7['cells'][aug]
                same_prompt=bool(base['prompt_hashes'].get(aug)) and base['prompt_hashes'].get(aug)==v7['prompt_hashes'].get(aug) and bcell['root_length']==vcell['root_length']
                prompt_match+=int(same_prompt)
                same_tokenizer=base['tokenizer_identity'] is not None and base['tokenizer_identity']==v7['tokenizer_identity']
                tokenizer_match+=int(same_tokenizer)
                if not(same_prompt and same_tokenizer):continue
                left={row['position']:row for row in bcell['probabilities']};right={row['position']:row for row in vcell['probabilities']}
                for pos in set(left)&set(right):
                    pair_prefixes+=1;local_lp.append(right[pos]['gold_logprob']-left[pos]['gold_logprob'])
                    if left[pos]['rank_among_12_legal_tokens'] is not None and right[pos]['rank_among_12_legal_tokens'] is not None:
                        local_rank.append(right[pos]['rank_among_12_legal_tokens']-left[pos]['rank_among_12_legal_tokens'])
            if local_lp:paired_lp.append((oid,sum(local_lp)/len(local_lp)))
            if local_rank:paired_rank.append((oid,sum(local_rank)/len(local_rank)))
    result={'status':'COMPLETE_CPU_OBSERVED_SEARCH_ANATOMY','outputs':89,'cells_per_model':712,'models':summary,
            'prompt_hash_and_root_length_matching_cells':prompt_match,'tokenizer_identity_matching_cells':tokenizer_match,
            'common_observed_prefix_pairs':pair_prefixes,'paired_V7_minus_Base_logprob':task_interval(paired_lp),
            'paired_V7_minus_Base_legal_token_rank':task_interval(paired_rank),
            'runtime_adapter_configuration_difference_outputs':config_diff_outputs,
            'GPU_used':False,'new_generation':False,'model_loaded':False,'Gold_path_teacher_forcing':'NOT_RUN',
            'limitations':['Survivor-conditioned observed paths, not full-path probabilities',
                           'Different TTT parameterization/runtime configurations confound a pure V7 effect when present',
                           'Missing historical frontier telemetry is UNAVAILABLE, not zero waiting',
                           'Final trace rows mutate at pop; checkpoint summaries are observation bounds, not complete historical frontier snapshots',
                           'Canonical native Gold serialization plus EOS only; parsed-equivalent token serializations may differ',
                           'Exposed Eval60; no independent or competition improvement claim'],
            'checkpoint_manifest':manifest}
    write(args.output/'RESULT.json',result)
    print('CPU_ANATOMY_ALL_89_COMPLETE',flush=True)


if __name__=='__main__':main()
