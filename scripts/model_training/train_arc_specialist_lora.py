#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class ArcSFTDataset(Dataset):
    def __init__(self,path,tokenizer,max_len):
        self.rows=[]; self.tok=tokenizer; self.max_len=max_len
        for line in Path(path).read_text().splitlines():
            if not line.strip(): continue
            rec=json.loads(line)
            enc=self.encode(rec["messages"])
            if enc is not None: self.rows.append(enc)
        if not self.rows: raise RuntimeError("no training rows survived length checks")

    def one(self,text):
        return self.tok.encode(text,add_special_tokens=False)

    def encode(self,messages):
        ims=self.tok.convert_tokens_to_ids("<|im_start|>")
        ime=self.tok.convert_tokens_to_ids("<|im_end|>")
        nl=self.one("\n")
        if not isinstance(ims,int) or not isinstance(ime,int) or len(nl)!=1:
            raise RuntimeError("unexpected ARC tokenizer structure")
        ids=[]; labels=[]
        for m in messages:
            role=self.one(m["role"])
            if len(role)!=1: raise RuntimeError(f"role not one token: {m['role']}")
            content=self.one(m["content"])
            seg=[ims,role[0],nl[0],*content,ime]
            ids.extend(seg)
            if m["role"]=="assistant":
                labels.extend([-100,-100,-100,*content,ime])
            else:
                labels.extend([-100]*len(seg))
        if len(ids)>self.max_len: return None
        return {"input_ids":ids,"labels":labels,"attention_mask":[1]*len(ids)}

    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

class Collator:
    def __init__(self,pad): self.pad=pad
    def __call__(self,batch):
        n=max(len(x["input_ids"]) for x in batch)
        out={"input_ids":[],"labels":[],"attention_mask":[]}
        for x in batch:
            p=n-len(x["input_ids"])
            out["input_ids"].append(x["input_ids"]+[self.pad]*p)
            out["labels"].append(x["labels"]+[-100]*p)
            out["attention_mask"].append(x["attention_mask"]+[0]*p)
        return {k:torch.tensor(v,dtype=torch.long) for k,v in out.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path",required=True)
    ap.add_argument("--train-jsonl",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--max-seq-length",type=int,default=16384)
    ap.add_argument("--epochs",type=float,default=1.0)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--micro-batch-size",type=int,default=1)
    ap.add_argument("--gradient-accumulation",type=int,default=8)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--no-4bit",action="store_true")
    args=ap.parse_args()

    tok=AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
    # Bind to the ARC-specific vocabulary before spending GPU.
    if len(tok)!=16:
        raise RuntimeError(f"expected 16-token ARC tokenizer, got {len(tok)}")
    for d in "0123456789":
        if tok.encode(d,add_special_tokens=False)!=[int(d)]:
            raise RuntimeError("digit-token identity mismatch")

    qcfg=None
    if not args.no_4bit:
        qcfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(
        args.model_path,trust_remote_code=True,torch_dtype=torch.bfloat16,
        quantization_config=qcfg,device_map="auto",attn_implementation="sdpa")
    model.config.use_cache=False
    if qcfg is not None: model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    else: model.gradient_checkpointing_enable()

    lcfg=LoraConfig(
        r=64,lora_alpha=128,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model=get_peft_model(model,lcfg)
    model.print_trainable_parameters()

    ds=ArcSFTDataset(args.train_jsonl,tok,args.max_seq_length)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    meta={"base_model":str(args.model_path),"train_rows":len(ds),"max_seq_length":args.max_seq_length,
          "epochs":args.epochs,"lr":args.lr,"lora_r":64,"lora_alpha":128,"seed":args.seed}
    (out/"TRAINING_CONTRACT.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")

    ta=TrainingArguments(
        output_dir=str(out/"checkpoints"),num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,weight_decay=0.01,warmup_ratio=0.03,
        bf16=True,logging_steps=10,save_strategy="epoch",report_to=[],
        remove_unused_columns=False,gradient_checkpointing=True,
        optim="paged_adamw_8bit" if qcfg is not None else "adamw_torch",
        seed=args.seed,data_seed=args.seed)
    tr=Trainer(model=model,args=ta,train_dataset=ds,data_collator=Collator(tok.pad_token_id))
    tr.train()
    model.save_pretrained(out/"adapter")
    tok.save_pretrained(out/"adapter")
    (out/"TRAINING_COMPLETE.json").write_text(json.dumps({"status":"PASS",**meta},indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
