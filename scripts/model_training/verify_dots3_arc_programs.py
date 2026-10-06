#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, gzip, hashlib, json, multiprocessing as mp, os
from pathlib import Path
import numpy as np

BANNED_NAMES={"eval","exec","open","compile","input","globals","locals","vars","getattr","setattr","delattr",
              "__import__","breakpoint","help","dir","type","object","super"}
BANNED_NODES=(ast.Import,ast.ImportFrom,ast.ClassDef,ast.AsyncFunctionDef,ast.With,ast.AsyncWith,
              ast.Try,ast.Raise,ast.Global,ast.Nonlocal,ast.Lambda,ast.Delete)
SAFE_BUILTINS={"range":range,"len":len,"min":min,"max":max,"sum":sum,"abs":abs,"int":int,"float":float,
               "bool":bool,"list":list,"tuple":tuple,"set":set,"dict":dict,"enumerate":enumerate,
               "zip":zip,"sorted":sorted,"any":any,"all":all,"round":round}

def canonical(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def sha(x): return hashlib.sha256(x.encode()).hexdigest()

def lint(code:str):
    tree=ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node,BANNED_NODES): raise ValueError(f"banned AST node {type(node).__name__}")
        if isinstance(node,ast.Name) and node.id in BANNED_NAMES: raise ValueError(f"banned name {node.id}")
        if isinstance(node,ast.Attribute) and node.attr.startswith("__"): raise ValueError("dunder attribute")
    return tree

def valid_grid(x):
    if not isinstance(x,np.ndarray) or x.ndim!=2: return False
    if not (1<=x.shape[0]<=30 and 1<=x.shape[1]<=30): return False
    if not np.issubdtype(x.dtype,np.integer): return False
    return bool(np.all((x>=0)&(x<=9)))

def worker(payload,q):
    try:
        input_code,output_code,seeds=payload
        g={"np":np,"__builtins__":SAFE_BUILTINS}
        exec(compile(lint(input_code),"<input>","exec"),g,g)
        exec(compile(lint(output_code),"<output>","exec"),g,g)
        gi=g.get("generate_input"); tf=g.get("transform")
        if not callable(gi) or not callable(tf): raise ValueError("required functions missing")
        pairs=[]
        for s in seeds:
            x=gi(int(s))
            if not valid_grid(x): raise ValueError(f"invalid input seed={s}")
            y1=tf(x.copy()); y2=tf(x.copy())
            if not valid_grid(y1): raise ValueError(f"invalid output seed={s}")
            if not np.array_equal(y1,y2): raise ValueError(f"nondeterministic transform seed={s}")
            pairs.append({"seed":int(s),"input":x.astype(int).tolist(),"output":y1.astype(int).tolist()})
        q.put({"ok":True,"pairs":pairs})
    except BaseException as e:
        q.put({"ok":False,"error":repr(e)})

def run_program(inp,out,seeds,timeout):
    q=mp.Queue(1); p=mp.Process(target=worker,args=((inp,out,seeds),q)); p.start(); p.join(timeout)
    if p.is_alive():
        p.kill(); p.join(); return {"ok":False,"error":"timeout"}
    return q.get() if not q.empty() else {"ok":False,"error":f"worker_exit_{p.exitcode}"}

def behavior_signature(pairs):
    return sha(canonical([{"input":p["input"],"output":p["output"]} for p in pairs]))

def split_family(fid):
    v=int(hashlib.sha256(fid.encode()).hexdigest()[:8],16)%100
    return "train" if v<70 else ("val" if v<85 else "test")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--drafts",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,default=64)
    ap.add_argument("--timeout",type=float,default=20)
    args=ap.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    seen_code=set(); seen_behavior=set(); accepted=0; rejected=0
    with args.output.open("wt",encoding="utf-8") if args.output.suffix!=".gz" else gzip.open(args.output,"wt",encoding="utf-8") as out:
        for line in args.drafts.read_text().splitlines():
            if not line.strip(): continue
            rec=json.loads(line)
            if "error" in rec: rejected+=1; continue
            t=rec["teacher_output"]; inp=t["input_code"]; outc=t["output_code"]
            try:
                lint(inp); lint(outc)
                ph=sha(inp+"\n---\n"+outc)
                if ph in seen_code: raise ValueError("duplicate_code")
                result=run_program(inp,outc,list(range(args.seeds)),args.timeout)
                if not result["ok"]: raise ValueError(result["error"])
                pairs=result["pairs"]
                ins={sha(canonical(p["input"])) for p in pairs}; outs={sha(canonical(p["output"])) for p in pairs}
                identity=sum(p["input"]==p["output"] for p in pairs)/len(pairs)
                if len(ins)<int(0.75*len(pairs)): raise ValueError("low_input_diversity")
                if len(outs)<int(0.35*len(pairs)): raise ValueError("low_output_diversity")
                if identity>0.25: raise ValueError("too_many_identity_examples")
                bs=behavior_signature(pairs)
                if bs in seen_behavior: raise ValueError("duplicate_behavior")
                seen_code.add(ph); seen_behavior.add(bs)
                accepted+=1
                out.write(json.dumps({
                    "status":"PASS","family_id":rec["family_id"],"split":split_family(rec["family_id"]),
                    "draft_index":rec["draft_index"],"teacher_model":rec["teacher_model"],
                    "program_hash":ph,"behavior_hash":bs,"rules_summary":t["rules_summary"],
                    "key_insight":t["key_insight"],"concepts":t["concepts"],
                    "difficulty_axes":t.get("difficulty_axes",[]),"input_code":inp,"output_code":outc,
                    "pairs":pairs
                },sort_keys=True)+"\n")
            except Exception as e:
                rejected+=1
    print(json.dumps({"accepted":accepted,"rejected":rejected,"output":str(args.output)},indent=2))
if __name__=="__main__": main()
