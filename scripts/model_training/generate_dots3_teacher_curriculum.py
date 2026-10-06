#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from openai import OpenAI

def load_json(path: Path):
    return json.loads(path.read_text())

def extract_json(text: str):
    text=text.strip()
    if text.startswith("```"):
        lines=text.splitlines()
        if lines and lines[0].startswith("```"): lines=lines[1:]
        if lines and lines[-1].strip()=="```": lines=lines[:-1]
        text="\n".join(lines)
    return json.loads(text)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--families",type=Path,required=True)
    p.add_argument("--prompt",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--base-url",required=True)
    p.add_argument("--model",default="dots3-note-prev")
    p.add_argument("--api-key-env",default="DOTS_API_KEY")
    p.add_argument("--drafts-per-family",type=int,default=16)
    p.add_argument("--temperature",type=float,default=0.9)
    p.add_argument("--top-p",type=float,default=0.95)
    p.add_argument("--max-tokens",type=int,default=6000)
    p.add_argument("--sleep",type=float,default=0.0)
    args=p.parse_args()

    key=os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f"missing API key env: {args.api_key_env}")
    client=OpenAI(base_url=args.base_url,api_key=key)
    families=load_json(args.families)["families"]
    template=args.prompt.read_text()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    existing=set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                r=json.loads(line); existing.add((r["family_id"],int(r["draft_index"])))

    with args.output.open("a",encoding="utf-8") as out:
        for idx, concepts in enumerate(families):
            family_id=f"F{idx:02d}__{'__'.join(concepts)}"
            spec=json.dumps({"family_id":family_id,"concepts":concepts},sort_keys=True)
            user=template.replace("{FAMILY_SPEC}",spec)
            for draft in range(args.drafts_per_family):
                if (family_id,draft) in existing: continue
                last_error=None
                for attempt in range(3):
                    try:
                        resp=client.chat.completions.create(
                            model=args.model,
                            messages=[{"role":"user","content":user}],
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_tokens=args.max_tokens,
                            extra_body={"chat_template_kwargs":{"enable_thinking":True}},
                        )
                        raw=resp.choices[0].message.content
                        obj=extract_json(raw)
                        if obj.get("family_id")!=family_id:
                            raise ValueError("family_id mismatch")
                        for k in ("rules_summary","key_insight","concepts","input_code","output_code"):
                            if not obj.get(k): raise ValueError(f"missing {k}")
                        rec={"teacher_model":args.model,"family_id":family_id,"draft_index":draft,
                             "concepts_requested":concepts,"teacher_output":obj}
                        out.write(json.dumps(rec,sort_keys=True)+"\n"); out.flush()
                        last_error=None; break
                    except Exception as e:
                        last_error=repr(e); time.sleep(2*(attempt+1))
                if last_error:
                    out.write(json.dumps({"teacher_model":args.model,"family_id":family_id,
                                          "draft_index":draft,"error":last_error},sort_keys=True)+"\n"); out.flush()
                if args.sleep: time.sleep(args.sleep)
if __name__=="__main__": main()
