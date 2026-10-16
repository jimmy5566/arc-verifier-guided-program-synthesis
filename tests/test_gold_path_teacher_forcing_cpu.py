"""Actual Torch CPU causal-shift fixture, no trained weights or GPU.

Skipped only when local Torch is absent; the actual Pod CPU environment runs it.
"""
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from scripts.run_v7_ttt24_eval60_gold_path_diagnostic import score_gold_view


class Tokenizer:
    eos_token_id=15
    def apply_chat_template(self,*a,**kw):return {'input_ids':torch.tensor([[14,11,12]])}
    def __call__(self,text,**kw):
        assert text=='12\n34'
        return {'input_ids':torch.tensor([[1,2,10,3,4]])}


class DeterministicLogitFixture:
    device=torch.device('cpu')
    calls=0
    def __call__(self,**kw):
        self.calls+=1
        assert kw['use_cache'] is False
        assert kw['input_ids'].tolist()==[[14,11,12,1,2,10,3,4,15]]
        logits=torch.zeros((1,9,16))
        for pos,token in enumerate((1,2,10,3,4,15)):
            logits[0,2+pos,token]=pos+1
        logits[0,4,5]=3.5  # Gold newline rank 2 at its causal predictor.
        logits[0,6,13]=40  # Invalid ARC token affects normalization, not legal rank.
        return SimpleNamespace(logits=logits)


def test_actual_torch_shift_eos_full_vocab_normalizer_and_legal_rank():
    fixture=DeterministicLogitFixture()
    rows,summary=score_gold_view(fixture,Tokenizer(),[],[[1,2],[3,4]],context_window=9)
    assert fixture.calls==1
    assert [r['gold_token_id'] for r in rows]==[1,2,10,3,4,15]
    assert [r['gold_token_local_rank'] for r in rows]==[1,1,2,1,1,1]
    assert summary['mean_gold_token_rank']==pytest.approx(7/6)
    assert summary['cumulative_gold_regret']==pytest.approx(.5)
    expected=[]
    for pos in range(6):
        score=pos+1
        norm=(math.exp(score)+math.exp(3.5)+14) if pos==2 else (math.exp(score)+math.exp(40)+14) if pos==4 else math.exp(score)+15
        expected.append(score-math.log(norm))
    assert [r['gold_token_logprob'] for r in rows]==pytest.approx(expected,abs=1e-5)
    assert summary['cumulative_gold_nll']==pytest.approx(-sum(expected),abs=1e-5)


def test_context_limit_rejects_before_fixture_forward():
    fixture=DeterministicLogitFixture()
    with pytest.raises(RuntimeError,match='CONTEXT_EXCEEDED'):
        score_gold_view(fixture,Tokenizer(),[],[[1,2],[3,4]],context_window=8)
    assert fixture.calls==0
