"""Pure runtime contract for the approved no-update component diagnostic."""
from __future__ import annotations
import hashlib, json, math
from collections import Counter

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def sha(value): return hashlib.sha256(canonical(value)).hexdigest()

def teacher_forced_layout(prompt_ids,target_ids,pad=13):
    if not prompt_ids or not target_ids: raise RuntimeError('EMPTY_CAUSAL_SEQUENCE')
    sequence=[int(x) for x in prompt_ids]+[int(x) for x in target_ids]
    return {'input_ids':sequence,'attention_mask':[1]*len(sequence),'labels':[-100]*len(prompt_ids)+[int(x) for x in target_ids], 'target_logit_indices':list(range(len(prompt_ids)-1,len(sequence)-1))}

def token_metric(logits,target):
    if not logits or not 0<=int(target)<len(logits): raise RuntimeError('TARGET_LOGIT_INVALID')
    values=[float(x) for x in logits]; correct=values[int(target)]
    maximum=max(values); logz=maximum+math.log(sum(math.exp(v-maximum) for v in values))
    rank=1+sum(v>correct for v in values)
    other=max(v for i,v in enumerate(values) if i!=int(target)) if len(values)>1 else float('-inf')
    return {'rank':rank,'top1':rank==1,'top2':rank<=2,'margin':correct-other,'nll':logz-correct}

def aggregate_token_metrics(metrics):
    if not metrics: raise RuntimeError('METRICS_EMPTY')
    n=len(metrics); return {'tokens':n,'top1_count':sum(x['top1'] for x in metrics),'top2_count':sum(x['top2'] for x in metrics),'mean_margin':sum(x['margin'] for x in metrics)/n,'token_weighted_nll':sum(x['nll'] for x in metrics)/n}

def first_divergence(generated,target):
    for i,(left,right) in enumerate(zip(generated,target)):
        if int(left)!=int(right): return i
    return None if len(generated)==len(target) else min(len(generated),len(target))

def semantic_taxonomy(predicted, expected, parser_valid):
    if not parser_valid: return {'primary':'PARSER_FAILURE','axes':['PARSER_FAILURE']}
    if predicted==expected: return {'primary':'EXACT','axes':[]}
    if not isinstance(predicted,list) or not predicted or len(predicted)!=len(expected) or any(len(a)!=len(b) for a,b in zip(predicted,expected)):
        return {'primary':'DIMENSIONS','axes':['DIMENSIONS']}
    axes=[]
    # The frozen diagnostic's count targets differ only by component multiplicity;
    # use the observable non-zero count as the independent count signature.
    p=sum(v!=0 for row in predicted for v in row); e=sum(v!=0 for row in expected for v in row)
    if p!=e: axes.append('CONNECTED_COMPONENT_COUNT')
    if sorted(v for row in predicted for v in row if v) != sorted(v for row in expected for v in row if v): axes.append('COLOR')
    if not axes: axes.append('OUTPUT_CONSTRUCTION')
    order=['CONNECTED_COMPONENT_COUNT','SELECTED_OBJECT_IDENTITY','GEOMETRY','COLOR','OUTPUT_CONSTRUCTION']
    return {'primary':next(x for x in order if x in axes),'axes':axes}

def validate_sidecar(sidecar, inventory):
    records=sidecar.get('records') if isinstance(sidecar,dict) else None
    if not isinstance(records,list) or len(records)!=72 or len({x.get('cohort_record_id') for x in records})!=72: raise RuntimeError('SIDECAR_LAYOUT_INVALID')
    for row in records:
        forbidden={'target_grid','target_serialization','prompt_text'} & set(row)
        if forbidden: raise RuntimeError('SIDECAR_BOUNDARY_VIOLATION')
        ids=row.get('target_token_ids'); mask=row.get('supervised_token_mask')
        if not isinstance(ids,list) or not ids or ids[-1]!=15 or not isinstance(mask,list) or len(mask)!=len(ids) or any(x!=1 for x in mask): raise RuntimeError('SIDECAR_TOKEN_MASK_INVALID')
    expected={(x['checkpoint'],x['cohort_record_id']) for x in inventory.get('inventory',[])}
    if len(expected)!=216: raise RuntimeError('INVENTORY_LAYOUT_INVALID')
    return {x['cohort_record_id']:x for x in records}

def validate_reuse(raw_by_checkpoint, inventory):
    reused=[x for x in inventory['inventory'] if x['disposition']=='REUSE_IMMUTABLE_RUN_002_RAW']
    missing=[x for x in inventory['inventory'] if x['disposition']=='GENERATE_NEW_ONCE_AFTER_FRESH_AUTHORIZATION']
    if len(reused)!=168 or len(missing)!=48 or any(x['split']!='TRAIN_FIT' for x in missing): raise RuntimeError('REUSE_SET_DIFFERENCE_INVALID')
    for row in reused:
        got=raw_by_checkpoint.get(row['checkpoint'],{}).get(row['prompt_task_sha256'])
        if got is None or sha(got)!=row['run_002_raw_record_sha256']: raise RuntimeError('REUSED_RAW_IDENTITY_MISMATCH')
    return reused,missing

def batch_gate(primary,b1,tolerance):
    if len(primary)!=len(b1) or not primary: raise RuntimeError('BATCH1_PAIRING_INVALID')
    keys=sorted(primary)
    if keys!=sorted(b1): raise RuntimeError('BATCH1_PAIRING_INVALID')
    rank=max(abs(primary[k]['rank']-b1[k]['rank']) for k in keys)
    top1=abs(sum(primary[k]['top1'] for k in keys)-sum(b1[k]['top1'] for k in keys))
    top2=abs(sum(primary[k]['top2'] for k in keys)-sum(b1[k]['top2'] for k in keys))
    margin=max(abs(primary[k]['margin']-b1[k]['margin']) for k in keys)
    nll=max(abs(primary[k]['nll']-b1[k]['nll']) for k in keys)
    passed=rank<=tolerance['correct_token_rank_abs_delta_max'] and top1<=tolerance['Top1_and_Top2_aggregate_count_delta_max'] and top2<=tolerance['Top1_and_Top2_aggregate_count_delta_max'] and margin<=tolerance['per_row_mean_margin_abs_delta_max'] and nll<=tolerance['token_weighted_NLL_abs_delta_max']
    return {'status':'PASS' if passed else 'INVALID_NOT_INTERPRETABLE','rank_abs_delta_max':rank,'top1_count_delta':top1,'top2_count_delta':top2,'margin_abs_delta_max':margin,'nll_abs_delta_max':nll}

def aggregate_record_metrics(record_metrics):
    """Aggregate token metrics while retaining sequence-normalized NLL."""
    if not isinstance(record_metrics, dict) or not record_metrics:
        raise RuntimeError("RECORD_METRICS_EMPTY")
    flattened=[]
    sequences=[]
    for record_id, metrics in sorted(record_metrics.items()):
        if not isinstance(record_id, str) or not isinstance(metrics, list) or not metrics:
            raise RuntimeError("RECORD_METRICS_LAYOUT_INVALID")
        flattened.extend(metrics)
        sequences.append(sum(float(item["nll"]) for item in metrics) / len(metrics))
    pooled=aggregate_token_metrics(flattened)
    pooled["records"]=len(sequences)
    pooled["sequence_normalized_nll"]=sum(sequences)/len(sequences)
    return pooled

def require_inventory_partition(inventory):
    rows=inventory.get("inventory") if isinstance(inventory,dict) else None
    if not isinstance(rows,list) or len(rows)!=216:
        raise RuntimeError("INVENTORY_ROW_COUNT_INVALID")
    reused=[row for row in rows if row.get("disposition")=="REUSE_IMMUTABLE_RUN_002_RAW"]
    fresh=[row for row in rows if row.get("disposition")=="GENERATE_NEW_ONCE_AFTER_FRESH_AUTHORIZATION"]
    if len(reused)!=168 or len(fresh)!=48 or any(row.get("split")!="TRAIN_FIT" for row in fresh):
        raise RuntimeError("INVENTORY_REUSE_FRESH_PARTITION_INVALID")
    keys={(row.get("checkpoint"),row.get("cohort_record_id")) for row in rows}
    if len(keys)!=216 or None in {key[0] for key in keys} or None in {key[1] for key in keys}:
        raise RuntimeError("INVENTORY_IDENTITY_LAYOUT_INVALID")
    return reused,fresh
