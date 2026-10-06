#!/usr/bin/env python3
"""Reproduce C1 from frozen S2 final ranks; Gold scoring must remain post-freeze."""
from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path

def same_shape(a,b):
    return len(a)==len(b) and len(a[0])==len(b[0])

def consensus(grids):
    h,w=len(grids[0]),len(grids[0][0]);out=[[0]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            cnt=Counter(g[r][c] for g in grids);m=max(cnt.values());tied={v for v,n in cnt.items() if n==m}
            out[r][c]=next(g[r][c] for g in grids if g[r][c] in tied)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--s2-final-ranks",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    rows=list(csv.DictReader(a.s2_final_ranks.open()));by={}
    for row in rows:by.setdefault(row["output_id"],[]).append(row)
    frozen={}
    for oid,group in by.items():
        group.sort(key=lambda x:int(x["final_s2_rank"]));rank1=json.loads(group[0]["canonical_grid"])
        same=[json.loads(x["canonical_grid"]) for x in group if same_shape(json.loads(x["canonical_grid"]),rank1)][:5]
        frozen[oid]=None if len(same)<3 else consensus(same)
    a.output.write_text(json.dumps(frozen,sort_keys=True,separators=(",",":"))+"\n")
if __name__=="__main__":main()
