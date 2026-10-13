"""CPU-only repair for the deprecated Code-Pair final-answer locator.
Not imported by E4-A.
"""
from __future__ import annotations
IGNORE=-100
def locate_final_query_assistant(labels,input_ids,train_pairs):
 runs=[]; start=None
 for i,v in enumerate(list(labels)+[IGNORE]):
  if v!=IGNORE and start is None:start=i
  elif v==IGNORE and start is not None:runs.append((start,i));start=None
 if len(runs)!=train_pairs+1: raise ValueError("ASSISTANT_TURN_GRID_MISMATCH")
 a,b=runs[-1]
 if a==b or list(labels[a:b])!=list(input_ids[a:b]): raise ValueError("FINAL_QUERY_TOKEN_IDENTITY")
 return {"start":a,"end":b,"token_ids":list(input_ids[a:b]),"assistant_turns":runs}
