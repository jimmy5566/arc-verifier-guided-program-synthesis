"""CPU-only serialization feasibility audit for the R5 Cap192 successor."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p: Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def atomic_json(path:Path,value:dict)->None:
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
 tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n",encoding="utf-8"); os.replace(tmp,path)

def audit(model_path:Path,native_config_dir:Path,output:Path)->dict:
 # The tokenizer loader validates the checkpoint-native 16-token vocabulary; it
 # constructs neither a model nor tensors on CUDA.  No task, prompt or target is read.
 from inference.nvarc_native import checkpoint_native_tokenizer, parse_native_grid, serialize_grid
 tokenizer, tokenizer_identity=checkpoint_native_tokenizer(model_path,native_config_dir)
 records=[]
 for height in range(1,14):
  for width in range(1,14):
   grid=[[(r*width+c)%10 for c in range(width)] for r in range(height)]
   text=serialize_grid(grid)
   if parse_native_grid(text)!=grid: raise RuntimeError(f"SERIALIZATION_PARSE_FAILURE:{height}x{width}")
   ids=tokenizer(text,add_special_tokens=False)["input_ids"]
   records.append({"height":height,"width":width,"completion_tokens":len(ids)})
 max_record=max(records,key=lambda r:(r["completion_tokens"],r["height"],r["width"]))
 if max_record["completion_tokens"]!=181: raise RuntimeError(f"UNEXPECTED_NATIVE_TOKEN_CAP:{max_record}")
 result={"protocol_id":"ATOMIC_OOD_IDENTIFIABILITY_PILOT_V4_THREE_CHECKPOINT_NATIVE_GREEDY_NO_UPDATE_R5_CAP192","status":"PASS_CPU_TOKENIZER_AND_PARSER","model_loaded":False,"gpu_used":False,"target_sidecar_accessed":False,"dimensions_tested":len(records),"max_dimension":[13,13],"max_serialized_tokens":max_record["completion_tokens"],"old_cap":128,"old_cap_feasible":False,"eos_allowance":1,"conservative_completion_cap":192,"tokenizer_identity":tokenizer_identity,"records":records}
 atomic_json(output,result); return result

if __name__=="__main__":
 ap=argparse.ArgumentParser(); ap.add_argument("--model-path",type=Path,required=True); ap.add_argument("--native-config-dir",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args(); audit(a.model_path,a.native_config_dir,a.output)
