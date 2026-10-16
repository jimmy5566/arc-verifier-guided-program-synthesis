import copy
import gzip
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_v7_ttt24_eval60_gold_path_diagnostic as d


class Tokenizer:
    def apply_chat_template(self,messages,**kw):
        return [1,2,3] if kw['tokenize'] else 'rendered'


def cache_data():
    result={'cells':{},'prompt_hashes':{},'tokenizer_identity':{'native':True}}
    for view in d.SCORING_VIEWS:
        aug='geom='+view+'__color=id__order=canonical'
        result['prompt_hashes'][aug]=hashlib.sha256(b'rendered').hexdigest()
        result['cells'][aug]={'root_length':3,'gold_length_including_eos':2,
            'probabilities':[{'position':pos,'gold_logprob':-.1,'gap_to_best_legal':.05,
                'rank_among_12_legal_tokens':2,'complete_legal_probabilities':True} for pos in range(2)]}
    return result


def args_for(tmp_path):
    return SimpleNamespace(output=tmp_path/'result',context_window=100,observed_cache_sha256='cache',
        _cache_tokenizer=Tokenizer(),_cache_tokenizer_identity={'native':True})


def route_fixture(monkeypatch,data):
    class View:
        def __init__(self,geometry):pass
        def transform_task(self,task):return task
        def transform_grid(self,grid):return np.asarray(grid)
    monkeypatch.setattr(d,'NativeAugmentation',View)
    monkeypatch.setattr(d,'native_messages',lambda *a:[])
    monkeypatch.setattr(d,'observed_output',lambda *a:copy.deepcopy(data))


def test_complete_cache_never_loads_model_and_existing_shard_reuses(monkeypatch,tmp_path):
    data=cache_data();route_fixture(monkeypatch,data)
    def forbidden(*a,**kw):raise AssertionError('MODEL_LOADING_FOR_CACHED_CELL')
    monkeypatch.setattr(d,'load_mode',forbidden)
    monkeypatch.setattr(d,'score_gold_view',forbidden)
    args=args_for(tmp_path);adapters={'tasks':{'a':{'path':'unused','adapter_model_sha256':'adapter'}}}
    for _ in range(2):
        tokens,views=d.score_mode(args,'base',['a:o0'],{'a':object()},{'a:o0':[[1]]},adapters)
        assert len(tokens)==16 and len(views)==8
        assert all(row['measurement_source']=='FROZEN_OBSERVED_DISTRIBUTIONS' for row in tokens)
        assert views[0]['cumulative_gold_nll']==pytest.approx(.2)
        assert views[0]['cumulative_gold_regret']==pytest.approx(.1)


def test_only_one_incomplete_view_calls_forward(monkeypatch,tmp_path):
    data=cache_data();full=d.cached_gold_view(data,'identity',[[1]])
    data['cells']['geom=identity__color=id__order=canonical']['probabilities'].pop()
    route_fixture(monkeypatch,data);calls=[]
    monkeypatch.setattr(d,'load_mode',lambda *a:(calls.append('load') or (object(),SimpleNamespace(tokenizer=object()),{})))
    monkeypatch.setattr(d,'activate',lambda *a: calls.append('activate'))
    monkeypatch.setattr(d,'score_gold_view',lambda *a,**kw:(calls.append('forward') or copy.deepcopy(full)))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
    tokens,views=d.score_mode(args_for(tmp_path),'v7',['a:o0'],{'a':object()},{'a:o0':[[1]]},
                             {'tasks':{'a':{'path':'unused','adapter_model_sha256':'adapter'}}})
    assert calls==['load','forward']
    assert sum(v['measurement_source']=='NEW_TEACHER_FORCED_FORWARD' for v in views)==1
    assert sum(v['measurement_source']=='FROZEN_OBSERVED_DISTRIBUTIONS' for v in views)==7


def test_partial_or_wrong_gold_length_never_fabricates_distribution():
    data=cache_data();cell=data['cells']['geom=identity__color=id__order=canonical']
    cell['probabilities'][0]['complete_legal_probabilities']=False
    assert d.cached_gold_view(data,'identity',[[1]]) is None
    with pytest.raises(RuntimeError,match='GOLD_LENGTH'):
        d.cached_gold_view(data,'identity',[[1,2]])


@pytest.mark.parametrize('field,value',[('gold_logprob',float('nan')),('rank_among_12_legal_tokens',0),('gap_to_best_legal',-1)])
def test_bad_cached_values_rejected(field,value):
    data=cache_data();data['cells']['geom=identity__color=id__order=canonical']['probabilities'][0][field]=value
    with pytest.raises(RuntimeError,match='DISTRIBUTION'):d.cached_gold_view(data,'identity',[[1]])


def test_prompt_identity_is_checked_before_model_load(monkeypatch,tmp_path):
    data=cache_data();data['prompt_hashes']['geom=identity__color=id__order=canonical']='wrong'
    route_fixture(monkeypatch,data)
    monkeypatch.setattr(d,'load_mode',lambda *a:pytest.fail('must reject before loading'))
    with pytest.raises(RuntimeError,match='PROMPT_OR_TOKENIZER'):
        d.score_mode(args_for(tmp_path),'v7',['a:o0'],{'a':object()},{'a:o0':[[1]]},
                     {'tasks':{'a':{'path':'unused','adapter_model_sha256':'adapter'}}})


def test_sha_bound_cache_reader_and_archive_link(tmp_path):
    cache=tmp_path/'cache';archive=tmp_path/'archive';cache.mkdir();(archive/'manifests').mkdir(parents=True)
    payload={'output_id':'a:o0','modes':{'base':cache_data()},'input_hashes':{
        'base':{'raw_source':'raw/a.json.gz','sha256':'raw_hash','candidate_index_sha256':'index_hash'}}}
    raw=gzip.compress(json.dumps(payload).encode());(cache/'a.gz').write_bytes(raw)
    result=json.dumps({'checkpoint_manifest':{'a:o0':{'file':'a.gz','sha256':hashlib.sha256(raw).hexdigest()}}}).encode()
    (cache/'RESULT.json').write_bytes(result)
    (archive/'manifests/ARCHIVE_HASHES.json').write_text(json.dumps({'files':{
        'raw/a.json.gz':'raw_hash','tables/candidate_index/a_o0.jsonl.gz':'index_hash'}}))
    args=SimpleNamespace(observed_cache=cache,observed_cache_sha256=hashlib.sha256(result).hexdigest(),base_archive=archive)
    assert d.observed_output(args,'base','a:o0')['tokenizer_identity']=={'native':True}
    (cache/'a.gz').write_bytes(b'bad')
    with pytest.raises(RuntimeError,match='CHECKPOINT_HASH'):d.observed_output(args,'base','a:o0')


def test_cpu_preflight_uses_production_reader_no_models_or_targets(monkeypatch,tmp_path):
    data=cache_data();route_fixture(monkeypatch,data)
    monkeypatch.setattr(d,'checkpoint_native_tokenizer',lambda *a:(Tokenizer(),{'native':True}))
    args=args_for(tmp_path);args.observed_cache=tmp_path;args.model_path=tmp_path;args.native_config=tmp_path
    report=d.preflight_observed_cache(args,['a:o0'],{'a':object()})
    assert report['status']=='PASS' and report['view_cells']==16
    assert report['counts']['base']=={'reusable_complete_cells':8,'cells_needing_forward':0}
    assert report['model_loaded'] is False and report['GPU_used'] is False and report['targets_opened'] is False


def test_full_paired_summary_requires_all_views_and_clusters_task():
    rows=[]
    for oid,delta in [('a:o0',0),('a:o1',2),('b:o0',3)]:
        for mode in ('base_ttt24','v7_ttt24'):
            for view in d.SCORING_VIEWS:
                value=10+(delta if mode=='v7_ttt24' else 0)
                rows.append({'comparison_mode':mode,'output_id':oid,'view':view,'prefix_tokens':3,
                             'gold_tokens_including_eos':2,'cumulative_gold_nll':value,
                             'cumulative_gold_regret':value,'mean_gold_token_rank':value})
    summary=d.paired_model_preference(rows,['a:o0','a:o1','b:o0'])
    assert summary['cumulative_gold_nll']['tasks']==2
    assert summary['cumulative_gold_nll']['mean_task_delta']==2
    with pytest.raises(RuntimeError,match='INCOMPLETE'):
        d.paired_model_preference(rows[:-1],['a:o0','a:o1','b:o0'])
    with pytest.raises(RuntimeError,match='DUPLICATE'):
        d.paired_model_preference(rows+[rows[0]],['a:o0','a:o1','b:o0'])
