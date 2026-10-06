#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip, hashlib, json, random
from pathlib import Path
import numpy as np

def read_rows(path):
    op=gzip.open if path.suffix==".gz" else open
    with op(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def grid_text(g): return "\n".join("".join(str(int(x)) for x in row) for row in g)
def d4(a,k):
    if k<4: return np.rot90(a,k)
    return np.fliplr(np.rot90(a,k-4))
def augment_pair(p,rng):
    x=np.asarray(p["input"],dtype=np.int8); y=np.asarray(p["output"],dtype=np.int8)
    k=rng.randrange(8); x=d4(x,k); y=d4(y,k)
    perm=list(range(10)); rng.shuffle(perm); lut=np.asarray(perm,dtype=np.int8)
    return {"input":lut[x].tolist(),"output":lut[y].tolist()}
def messages(pairs):
    out=[]
    for p in pairs:
        out.append({"role":"user","content":grid_text(p["input"])})
        out.append({"role":"assistant","content":grid_text(p["output"])})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--verified",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--train-augmentations",type=int,default=8)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    train=(args.out_dir/"train.jsonl").open("w",encoding="utf-8")
    evfiles={s:(args.out_dir/f"{s}_episodes.jsonl").open("w",encoding="utf-8") for s in ("val","test")}
    counts={"train":0,"val":0,"test":0}
    try:
        for rec in read_rows(args.verified):
            pairs=rec["pairs"]; split=rec["split"]; ph=rec["program_hash"]; fid=rec["family_id"]
            if split=="train":
                for a in range(args.train_augmentations):
                    rng=random.Random(int(hashlib.sha256(f"{args.seed}:{ph}:{a}".encode()).hexdigest()[:16],16))
                    chosen=rng.sample(pairs,k=min(6,len(pairs)))
                    aug=[augment_pair(p,rng) for p in chosen]
                    train.write(json.dumps({"program_hash":ph,"family_id":fid,"messages":messages(aug)},sort_keys=True)+"\n")
                    counts["train"]+=1
            else:
                rng=random.Random(int(hashlib.sha256(f"EVAL:{args.seed}:{ph}".encode()).hexdigest()[:16],16))
                # Four fixed episodes/program, complete family held out.
                for e in range(min(4,max(1,len(pairs)-5))):
                    chosen=rng.sample(pairs,k=min(6,len(pairs)))
                    ctx=chosen[:-1]; query=chosen[-1]
                    evfiles[split].write(json.dumps({
                        "program_hash":ph,"family_id":fid,"context_messages":messages(ctx),
                        "query_input":grid_text(query["input"]),"target_output":grid_text(query["output"])
                    },sort_keys=True)+"\n")
                    counts[split]+=1
    finally:
        train.close()
        for f in evfiles.values(): f.close()
    (args.out_dir/"MANIFEST.json").write_text(json.dumps(counts,indent=2,sort_keys=True)+"\n")
    print(json.dumps(counts,indent=2))
if __name__=="__main__": main()
