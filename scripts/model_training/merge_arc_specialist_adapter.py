#!/usr/bin/env python3
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
p=argparse.ArgumentParser(); p.add_argument("--base-model",required=True); p.add_argument("--adapter",required=True); p.add_argument("--output",required=True)
a=p.parse_args()
m=AutoModelForCausalLM.from_pretrained(a.base_model,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map="cpu")
m=PeftModel.from_pretrained(m,a.adapter).merge_and_unload(); m.save_pretrained(a.output,safe_serialization=True,max_shard_size="4GB")
AutoTokenizer.from_pretrained(a.base_model,trust_remote_code=True).save_pretrained(a.output)
