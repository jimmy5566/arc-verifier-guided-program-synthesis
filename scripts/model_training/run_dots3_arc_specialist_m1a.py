#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PY=sys.executable
def run(cmd):
    print("+"," ".join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),cwd=ROOT,check=True)

def count_verified(path):
    import gzip
    op=gzip.open if path.suffix==".gz" else open
    n=0
    with op(path,"rt",encoding="utf-8") as f:
        for line in f:
            if line.strip(): n+=1
    return n

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--work-dir",type=Path,required=True)
    p.add_argument("--model-path",required=True)
    p.add_argument("--teacher-base-url",required=True)
    p.add_argument("--teacher-model",default="dots3-note-prev")
    p.add_argument("--api-key-env",default="DOTS_API_KEY")
    p.add_argument("--drafts-per-family",type=int,default=16)
    p.add_argument("--min-accepted-programs",type=int,default=256)
    p.add_argument("--max-episodes",type=int,default=0)
    args=p.parse_args()
    w=args.work_dir; w.mkdir(parents=True,exist_ok=True)
    exp=ROOT/"experiments/model_training/dots3_arc_specialist_v1"
    drafts=w/"teacher_drafts.jsonl"; verified=w/"verified_programs.jsonl.gz"; data=w/"dataset"
    m0=w/"M0_TEST.json"; m1=w/"M1_TEST.json"; train=w/"M1A_TRAIN"; decision=w/"M1A_SCALE_GATE.json"

    if not drafts.exists():
        run([PY,ROOT/"scripts/model_training/generate_dots3_teacher_curriculum.py",
             "--families",exp/"CURRICULUM_FAMILIES.json","--prompt",exp/"TEACHER_PROMPT.md",
             "--output",drafts,"--base-url",args.teacher_base_url,"--model",args.teacher_model,
             "--api-key-env",args.api_key_env,"--drafts-per-family",args.drafts_per_family])
    if not verified.exists():
        run([PY,ROOT/"scripts/model_training/verify_dots3_arc_programs.py",
             "--drafts",drafts,"--output",verified,"--seeds","64","--timeout","20"])
    accepted=count_verified(verified)
    (w/"TEACHER_VERIFICATION_RECEIPT.json").write_text(json.dumps(
        {"status":"PASS" if accepted>=args.min_accepted_programs else "FAIL","accepted_programs":accepted,
         "minimum_required":args.min_accepted_programs,"teacher_model":args.teacher_model},indent=2,sort_keys=True)+"\n")
    if accepted<args.min_accepted_programs:
        raise SystemExit(f"teacher verification gate failed: accepted {accepted} < {args.min_accepted_programs}")

    if not (data/"MANIFEST.json").exists():
        run([PY,ROOT/"scripts/model_training/build_dots3_arc_dataset.py",
             "--verified",verified,"--out-dir",data,"--train-augmentations","8","--seed","42"])
    if not m0.exists():
        run([PY,ROOT/"scripts/model_training/eval_arc_specialist_gold_likelihood.py",
             "--base-model",args.model_path,"--episodes",data/"test_episodes.jsonl","--output",m0,
             *(["--max-episodes",str(args.max_episodes)] if args.max_episodes else [])])
    if not (train/"TRAINING_COMPLETE.json").exists():
        run([PY,ROOT/"scripts/model_training/train_arc_specialist_lora.py",
             "--model-path",args.model_path,"--train-jsonl",data/"train.jsonl","--output-dir",train,
             "--max-seq-length","16384","--epochs","1","--lr","0.0001",
             "--micro-batch-size","1","--gradient-accumulation","8","--seed","42"])
    if not m1.exists():
        run([PY,ROOT/"scripts/model_training/eval_arc_specialist_gold_likelihood.py",
             "--base-model",args.model_path,"--adapter",train/"adapter",
             "--episodes",data/"test_episodes.jsonl","--output",m1,
             *(["--max-episodes",str(args.max_episodes)] if args.max_episodes else [])])
    run([PY,ROOT/"scripts/model_training/compare_arc_specialist_m0_m1.py",
         "--m0",m0,"--m1",m1,"--output",decision])
    print(decision.read_text())
if __name__=="__main__": main()
