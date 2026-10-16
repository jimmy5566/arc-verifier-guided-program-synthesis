import copy
import math

import pytest

from scripts.analyze_cached_gold_sequence_nll_cpu import VIEWS, collect_rows, pair_gold


def fixture():
    rows = [{'output_id':'a:o0','grid_key':'[[1]]','depth':24,'scoring_view':view,
             'token_count':2,'continuation_tokens':1,'eos_tokens':1,'prefix_tokens':5,
             'mean_token_nll':1.,'sequence_negative_log_likelihood':2.,'sequence_log_likelihood':-2.,
             'tokenizer_identity':'tokenizer','scoring_context_window':20} for view in sorted(VIEWS)]
    modes = {m:{'prompt_hashes':{},'cells':{}} for m in ('base','v7')}
    for view in VIEWS:
        aug='geom='+view+'__color=id__order=canonical'
        for m in modes.values():m['prompt_hashes'][aug]='same';m['cells'][aug]={'root_length':5}
    return rows, {'modes':modes}


def test_eight_view_pair_and_no_score_substitution():
    rows, anatomy=fixture(); left=collect_rows(rows)[('a:o0','[[1]]')]
    right=copy.deepcopy(left)
    for row in right.values():row['sequence_negative_log_likelihood']=4.;row['mean_token_nll']=2.
    pairs,status=pair_gold(left,right,anatomy)
    assert len(pairs)==8 and status.startswith('MATCHED')
    assert all(row['V7_minus_Base_mean_token_nll']==1. for row in pairs)
    del right['rot90']
    assert pair_gold(left,right,anatomy)==([], 'MISSING_CACHED_VIEW')


@pytest.mark.parametrize('field,value', [('token_count',0),('sequence_negative_log_likelihood',math.nan),('sequence_log_likelihood',-5)])
def test_corrupt_score_rejected(field,value):
    rows,_=fixture();rows[0][field]=value
    with pytest.raises(RuntimeError):collect_rows(rows)


def test_duplicate_and_depth_isolation():
    rows,_=fixture()
    with pytest.raises(RuntimeError):collect_rows(rows+[rows[0]])
    wrong=copy.deepcopy(rows);wrong[0]['depth']=48
    assert len(collect_rows(wrong)[('a:o0','[[1]]')])==7


@pytest.mark.parametrize('field,value', [('prefix_tokens',6),('tokenizer_identity','other'),('eos_tokens',0)])
def test_mismatch_never_treated_as_matched(field,value):
    rows,anatomy=fixture();left=collect_rows(rows)[('a:o0','[[1]]')];right=copy.deepcopy(left)
    right['identity'][field]=value
    assert pair_gold(left,right,anatomy)[0]==[]


def test_prompt_mismatch():
    rows,anatomy=fixture();left=collect_rows(rows)[('a:o0','[[1]]')]
    anatomy['modes']['v7']['prompt_hashes']['geom=identity__color=id__order=canonical']='other'
    assert pair_gold(left,left,anatomy)==([], 'ARCHIVED_PROMPT_MISMATCH')
