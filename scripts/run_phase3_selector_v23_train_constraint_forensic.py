#!/usr/bin/env python3
"""CPU-only V2.3 train-derived deterministic constraint forensic."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from collections import Counter
from fractions import Fraction
from pathlib import Path

CHALLENGE_SHA="e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8"
SOLUTIONS_SHA="84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1"
MISSES={"142ca369:o0","20270e3b:o0","332f06d7:o0","58490d8a:o0","7491f3cf:o0","97d7923e:o0","b5ca7ac4:o0","b6f77b65:o0"}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dims(g): return len(g),len(g[0])
def colors(g): return sorted({x for r in g for x in r})
def dom(g):
    c=Counter(x for r in g for x in r).most_common()
    return None if len(c)>1 and c[0][1]==c[1][1] else (c[0][0] if c else None)
def hsym(g): return g==list(reversed(g))
def vsym(g): return all(r==list(reversed(r)) for r in g)
def r180(g): return g==[list(reversed(r)) for r in reversed(g)]
def bbox(g):
    p=[(r,c) for r,row in enumerate(g) for c,x in enumerate(row) if x!=0]
    if not p:return None
    rs=[x[0] for x in p];cs=[x[1] for x in p]
    return max(rs)-min(rs)+1,max(cs)-min(cs)+1
def comps(g):
    h,w=dims(g);seen=set();n=0
    for r in range(h):
      for c in range(w):
        if g[r][c]==0 or (r,c) in seen:continue
        n+=1;st=[(r,c)];seen.add((r,c))
        while st:
          x,y=st.pop()
          for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            a,b=x+dx,y+dy
            if 0<=a<h and 0<=b<w and g[a][b]!=0 and (a,b) not in seen:
              seen.add((a,b));st.append((a,b))
    return n
def ratio(a,b): return str(Fraction(a,b))

def infer(task):
    tr=task["train"];n=len(tr);I=[x["input"] for x in tr];O=[x["output"] for x in tr]
    ID=[dims(x) for x in I];OD=[dims(x) for x in O];A={k:[] for k in ("shape","palette","symmetry","bbox","components")}
    if all(ID[i]==OD[i] for i in range(n)):A["shape"].append(("same",True))
    if all(OD[i]==(ID[i][1],ID[i][0]) for i in range(n)):A["shape"].append(("transpose",True))
    if n>=2 and all(x==OD[0] for x in OD):A["shape"].append(("const_shape",OD[0]))
    d=[(OD[i][0]-ID[i][0],OD[i][1]-ID[i][1]) for i in range(n)]
    if n>=2 and all(x==d[0] for x in d):A["shape"].append(("delta",d[0]))
    s=[(ratio(OD[i][0],ID[i][0]),ratio(OD[i][1],ID[i][1])) for i in range(n)]
    if n>=2 and all(x==s[0] for x in s):A["shape"].append(("scale",s[0]))
    IC=[colors(x) for x in I];OC=[colors(x) for x in O]
    if all(set(OC[i])<=set(IC[i]) for i in range(n)):A["palette"].append(("subset",True))
    if all(OC[i]==IC[i] for i in range(n)):A["palette"].append(("equal",True))
    q=[len(x) for x in OC]
    if n>=2 and all(x==q[0] for x in q):A["palette"].append(("count",q[0]))
    q=[len(set(OC[i])-set(IC[i])) for i in range(n)]
    if n>=2 and all(x==q[0] for x in q):A["palette"].append(("new",q[0]))
    q=[len(set(IC[i])-set(OC[i])) for i in range(n)]
    if n>=2 and all(x==q[0] for x in q):A["palette"].append(("removed",q[0]))
    DI=[dom(x) for x in I];DO=[dom(x) for x in O]
    if all(x is not None for x in DI+DO) and all(DI[i]==DO[i] for i in range(n)):A["palette"].append(("dom_preserved",True))
    if n>=2 and all(x is not None for x in DO) and all(x==DO[0] for x in DO):A["palette"].append(("const_dom",DO[0]))
    if all(hsym(x) for x in O):A["symmetry"].append(("h",True))
    if all(vsym(x) for x in O):A["symmetry"].append(("v",True))
    if all(r180(x) for x in O):A["symmetry"].append(("r180",True))
    IB=[bbox(x) for x in I];OB=[bbox(x) for x in O]
    if all(x is not None for x in IB+OB):
      if all(IB[i]==OB[i] for i in range(n)):A["bbox"].append(("equal",True))
      if all(OB[i]==(IB[i][1],IB[i][0]) for i in range(n)):A["bbox"].append(("swap",True))
      d=[(OB[i][0]-IB[i][0],OB[i][1]-IB[i][1]) for i in range(n)]
      if n>=2 and all(x==d[0] for x in d):A["bbox"].append(("delta",d[0]))
      s=[(ratio(OB[i][0],IB[i][0]),ratio(OB[i][1],IB[i][1])) for i in range(n)]
      if n>=2 and all(x==s[0] for x in s):A["bbox"].append(("scale",s[0]))
    CI=[comps(x) for x in I];CO=[comps(x) for x in O]
    if all(CI[i]==CO[i] for i in range(n)):A["components"].append(("equal",True))
    d=[CO[i]-CI[i] for i in range(n)]
    if n>=2 and all(x==d[0] for x in d):A["components"].append(("delta",d[0]))
    if n>=2 and all(x==CO[0] for x in CO):A["components"].append(("const",CO[0]))
    return A

def score(A,inp,out):
    D,O=dims(inp),dims(out);IC,OC=colors(inp),colors(out);DI,DO=dom(inp),dom(out);IB,OB=bbox(inp),bbox(out);CI,CO=comps(inp),comps(out)
    app=vio=0
    for fam,atoms in A.items():
      if not atoms:continue
      app+=1;bad=False
      for n,v in atoms:
        ok=True
        if fam=="shape":
          ok={"same":D==O,"transpose":O==(D[1],D[0]),"const_shape":O==tuple(v) if n=="const_shape" else True}.get(n,ok)
          if n=="delta":ok=(O[0]-D[0],O[1]-D[1])==tuple(v)
          if n=="scale":ok=(ratio(O[0],D[0]),ratio(O[1],D[1]))==tuple(v)
        elif fam=="palette":
          if n=="subset":ok=set(OC)<=set(IC)
          elif n=="equal":ok=OC==IC
          elif n=="count":ok=len(OC)==v
          elif n=="new":ok=len(set(OC)-set(IC))==v
          elif n=="removed":ok=len(set(IC)-set(OC))==v
          elif n=="dom_preserved":ok=DI is not None and DO is not None and DI==DO
          elif n=="const_dom":ok=DO==v
        elif fam=="symmetry":
          ok={"h":hsym(out),"v":vsym(out),"r180":r180(out)}[n]
        elif fam=="bbox":
          if n=="equal":ok=IB is not None and OB==IB
          elif n=="swap":ok=IB is not None and OB==(IB[1],IB[0])
          elif n=="delta":ok=IB is not None and OB is not None and (OB[0]-IB[0],OB[1]-IB[1])==tuple(v)
          elif n=="scale":ok=IB is not None and OB is not None and (ratio(OB[0],IB[0]),ratio(OB[1],IB[1]))==tuple(v)
        else:
          if n=="equal":ok=CI==CO
          elif n=="delta":ok=CO-CI==v
          elif n=="const":ok=CO==v
        bad|=not ok
      vio+=int(bad)
    return app,vio,(vio/app if app else None)

def main():
    p=argparse.ArgumentParser();p.add_argument("--challenge",type=Path,required=True);p.add_argument("--solutions",type=Path,required=True);p.add_argument("--s2-final-ranks",type=Path,required=True);p.add_argument("--s2-output-results",type=Path,required=True);a=p.parse_args()
    if sha(a.challenge)!=CHALLENGE_SHA or sha(a.solutions)!=SOLUTIONS_SHA:raise RuntimeError("PROVENANCE_MISMATCH")
    ch=json.loads(a.challenge.read_text());sol=json.loads(a.solutions.read_text());rows=list(csv.DictReader(a.s2_final_ranks.open()));outs=list(csv.DictReader(a.s2_output_results.open()));OM={x["output_id"]:x for x in outs};BY={}
    for r in rows:BY.setdefault(r["output_id"],[]).append(r)
    for v in BY.values():v.sort(key=lambda x:int(x["final_s2_rank"]))
    per=[];PB=PW=PT=0
    for oid in sorted(MISSES):
      tid,idxs=oid.split(":o");idx=int(idxs);A=infer(ch[tid]);inp=ch[tid]["test"][idx]["input"];gold=sol[tid][idx];R=BY[oid];m=next(r for r in R if json.loads(r["canonical_grid"])==gold);gr=int(m["final_s2_rank"]);assert gr==int(OM[oid]["gold_s2_rank"]);gs=score(A,inp,gold);b=w=t=0
      for r in R:
        if int(r["final_s2_rank"])>=gr:continue
        cs=score(A,inp,json.loads(r["canonical_grid"]))
        if gs[2]<cs[2]:b+=1
        elif gs[2]>cs[2]:w+=1
        else:t+=1
      PB+=b;PW+=w;PT+=t;disc=b+w;signed=(b-w)/disc if disc else None;cl="UNINFORMATIVE" if not disc else ("WIN" if signed>0 else "LOSS" if signed<0 else "TIE");per.append((oid,gr,len(R),gr-1,b,w,t,disc,signed,cl))
    N=sum(x[9]!="UNINFORMATIVE" for x in per);wins=sum(x[9]=="WIN" for x in per);loss=sum(x[9]=="LOSS" for x in per);q=PB/(PB+PW) if PB+PW else None;c="TRAIN_CONSTRAINT_SIGNAL_SUPPORTED" if N>=5 and wins>=math.ceil(2*N/3) and q>=.60 else ("TRAIN_CONSTRAINT_SIGNAL_WEAK" if N>=4 and wins>loss and q>=.55 else "TRAIN_CONSTRAINT_SIGNAL_NOT_SUPPORTED");print(json.dumps({"classification":c,"informative_outputs":N,"wins":wins,"losses":loss,"pair_ties":PT,"per_output":per},indent=2))
if __name__=="__main__":main()
