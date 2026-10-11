from __future__ import annotations
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.e04_e_equal_slot_loss import EqualSlotLossError,causal_slot_mean_ce,equal_slot_objective

def slot(length:int,valid:int):
 logits=torch.randn(1,length,7,dtype=torch.float64,requires_grad=True)
 labels=torch.full((1,length),-100,dtype=torch.long)
 labels[0,1:1+valid]=torch.arange(valid)%7
 return logits,labels

def test_equal_slot_scalar_and_gradient_match_explicit_reference():
 slots=[slot(5,1),slot(7,3),slot(9,5),slot(11,7)]
 objective,counts=equal_slot_objective([x[0] for x in slots],[x[1] for x in slots])
 assert counts==[1,3,5,7]
 reference=sum(F.cross_entropy(x[0][:,:-1,:].float().reshape(-1,7),x[1][:,1:].reshape(-1),ignore_index=-100,reduction='mean') for x in slots)*.25
 assert torch.allclose(objective,reference,atol=1e-6,rtol=1e-6)
 objective.backward();got=[x[0].grad.clone() for x in slots]
 for x in slots:x[0].grad=None
 reference.backward()
 for observed,(logits,_) in zip(got,slots,strict=True):assert torch.allclose(observed,logits.grad,atol=1e-6,rtol=1e-6)

def test_token_weighted_aggregation_is_not_equal_slot_objective():
 slots=[slot(5,1),slot(7,3),slot(9,5),slot(11,7)]
 objective,_=equal_slot_objective([x[0] for x in slots],[x[1] for x in slots])
 token_sums=[];counts=[]
 for logits,labels in slots:
  mean,count=causal_slot_mean_ce(logits,labels);token_sums.append(mean*count);counts.append(count)
 weighted=sum(token_sums)/sum(counts)
 assert not torch.allclose(objective,weighted,atol=1e-10,rtol=1e-10)

def test_rejects_zero_and_missing_or_duplicate_slots():
 logits,labels=slot(5,1); zero=torch.full((1,5),-100,dtype=torch.long)
 with pytest.raises(EqualSlotLossError,match='ZERO_SUPERVISED'):causal_slot_mean_ce(logits,zero)
 with pytest.raises(EqualSlotLossError,match='MISSING_OR_DUPLICATE'):equal_slot_objective([logits]*3,[labels]*3)
