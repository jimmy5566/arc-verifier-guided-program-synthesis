"""Equal-slot supervised cross-entropy used by the E04-E V2 worker.

Every slot is independently normalized over its valid causal supervised labels.
The step objective is the arithmetic mean of exactly four slot means.
"""
from __future__ import annotations
from collections.abc import Sequence

class EqualSlotLossError(ValueError):
    pass

def causal_slot_mean_ce(logits, labels, *, ignore_index: int = -100):
    """FP32 mean CE for one causal-LM slot, rejecting an empty label mask."""
    import torch.nn.functional as F
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise EqualSlotLossError('E04E_LOSS_SLOT_SHAPE')
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(ignore_index)
    count = int(valid.sum().item())
    if count == 0:
        raise EqualSlotLossError('E04E_LOSS_ZERO_SUPERVISED_SLOT')
    token_losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=ignore_index, reduction='none')
    return token_losses.masked_select(valid.view(-1)).sum() / count, count

def equal_slot_objective(slot_logits: Sequence, slot_labels: Sequence, *, expected_slots: int = 4, ignore_index: int = -100):
    if len(slot_logits) != expected_slots or len(slot_labels) != expected_slots:
        raise EqualSlotLossError('E04E_LOSS_MISSING_OR_DUPLICATE_SLOT')
    means, counts = zip(*(causal_slot_mean_ce(logits, labels, ignore_index=ignore_index) for logits, labels in zip(slot_logits, slot_labels, strict=True)), strict=True)
    return sum(means) * (1.0 / expected_slots), list(counts)

def backward_equal_slot_term(slot_logits, slot_labels, *, expected_slots: int = 4, ignore_index: int = -100):
    """Backpropagate one independently normalized term of an equal-slot step.

    The caller invokes this once for each slot after ``optimizer.zero_grad``.
    Accumulating these four backwards is mathematically the same gradient as
    backpropagating ``equal_slot_objective`` once, while releasing each slot's
    activation graph before the next forward.  This is required for the
    physical-B1 E04-E worker on the 24 GiB runtime.
    """
    mean, count = causal_slot_mean_ce(slot_logits, slot_labels, ignore_index=ignore_index)
    term = mean * (1.0 / expected_slots)
    term.backward()
    return term.detach(), count
