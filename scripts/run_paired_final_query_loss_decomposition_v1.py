"""One-forward, no-update worker for PAIRED_FINAL_QUERY_LOSS_DECOMPOSITION_V1."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from paired_final_query_loss_common import atomic_json, component_sums, masks_for_task, sha
IGNORE, PAD = -100, 13

def load(config):
    configured = ROOT / config["inputs"]["target_dev_path"]
    candidates = [configured,
        Path("/workspace/arc2/mounted_inputs/base_only_targeted_repair_available_data_v1/TARGET_DEV.jsonl"),
        Path("/workspace/arc2/mounted_inputs/base_only_targeted_repair_remote_first_v2/TARGET_DEV.jsonl")]
    matching = [path for path in candidates if path.is_file() and sha(path) == config["inputs"]["target_dev_sha256"]]
    if len(matching) != 1:
        raise RuntimeError("TARGET_DEV_EXACT_IDENTITY_RESOLUTION_FAIL")
    dev, manifest_path = matching[0], ROOT / config["inputs"]["manifest_path"]
    if sha(manifest_path) != config["inputs"]["manifest_sha256"]:
        raise RuntimeError("INPUT_IDENTITY_FAIL")
    source = {row["episode_id"]: row for row in (json.loads(x) for x in dev.read_text().splitlines() if x.strip())}
    manifest, groups = json.loads(manifest_path.read_text()), []
    for batch in manifest["batches"]:
        group = []
        for member in batch["members"]:
            row = source.get(member["episode_id"])
            if not row or row["family"] != member["family"]:
                raise RuntimeError("ROW_BINDING_FAIL")
            item = masks_for_task({"source_id": row["episode_id"], **row["task"]})
            if len(item["ids"]) != member["sequence_length"] or item["component_token_counts"] != member["component_token_counts"]:
                raise RuntimeError("MASK_IDENTITY_FAIL")
            group.append({"episode_id": row["episode_id"], "family": row["family"], **item})
        if len(group) != batch["effective_batch_size"]:
            raise RuntimeError("BATCH_BINDING_FAIL")
        groups.append(group)
    if sum(map(len, groups)) != 60:
        raise RuntimeError("FROZEN_60_FAIL")
    return groups, manifest

def self_test(config):
    groups, manifest = load(config); flat = [x for group in groups for x in group]
    counts = {family: sum(x["family"] == family for x in flat) for family in manifest["families"]}
    if counts != {family: 12 for family in manifest["families"]}:
        raise RuntimeError("FAMILY_COUNT_FAIL")
    return {"status": "PASS_CPU_ONLY_NO_MODEL_IMPORT", "rows": 60, "batches": [len(g) for g in groups], "family_counts": counts, "crosscheck_rows": len(manifest["batch1_sensitivity_episode_ids"])}

def collate(group, torch):
    width = max(len(x["ids"]) for x in group); ids=[]; mask=[]; labels=[]
    for row in group:
        pad = width-len(row["ids"]); ids.append(row["ids"]+[PAD]*pad); mask.append([1]*len(row["ids"])+[0]*pad); labels.append(row["labels"]+[IGNORE]*pad)
    return torch.tensor(ids, device="cuda:0"), torch.tensor(mask, device="cuda:0"), torch.tensor(labels, device="cuda:0")

def score(model, groups, torch, F, checkpoint_id, mode_id, deadline):
    raw=[]
    with torch.no_grad():
        for batch_index, group in enumerate(groups):
            if time.monotonic() > deadline: raise RuntimeError("RUNTIME_CAP_REACHED")
            ids, attention, labels = collate(group, torch)
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits.float()
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=IGNORE, reduction="none").view_as(labels[:, 1:])
            for i, row in enumerate(group):
                vector = [float(v) for v in loss[i, :len(row["ids"])-1].tolist()]
                components = component_sums(vector, row["loss_masks"])
                raw.append({"checkpoint_id": checkpoint_id, "mode_id": mode_id, "batch_index": batch_index, "effective_batch_size": len(group), "episode_id": row["episode_id"], "family": row["family"], "sequence_length": len(row["ids"]), "components": components, "legacy_all_assistant_nll_sum": components["ALL_ASSISTANT_LEGACY"]["nll_sum"], "legacy_all_assistant_token_count": components["ALL_ASSISTANT_LEGACY"]["token_count"], "reconstruction_exact": True, "per_token_nll": vector})
    return raw

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--out", type=Path); args=parser.parse_args(); config=json.loads(args.config.read_text())
    if args.self_test: print(json.dumps(self_test(config), sort_keys=True)); return
    if args.out is None or args.out.exists(): raise RuntimeError("FRESH_OUTPUT_REQUIRED")
    groups, manifest = load(config); by_id={x["episode_id"]:x for group in groups for x in group}; cross=[[by_id[x]] for x in manifest["batch1_sensitivity_episode_ids"]]
    import torch, torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("CUDA_BF16_UNAVAILABLE")
    deadline=time.monotonic()+int(config["cap_seconds"]); raw=[]
    for checkpoint_id, condition in config["inputs"]["conditions"].items():
        checkpoint=json.loads((ROOT / condition["manifest"]).read_text())
        model=AutoModelForCausalLM.from_pretrained(checkpoint["base_path"], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
        model=PeftModel.from_pretrained(model, checkpoint["adapter_path"], is_trainable=False); model.eval()
        try: raw += score(model, groups, torch, F, checkpoint_id, "PRIMARY_B32", deadline) + score(model, cross, torch, F, checkpoint_id, "SENSITIVITY_B1", deadline)
        finally: del model; torch.cuda.empty_cache()
    args.out.mkdir(parents=True); raw_path=args.out / "RAW.jsonl"; raw_path.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in raw))
    atomic_json(args.out / "TERMINAL_RECEIPT.json", {"schema_version": 1, "protocol_id": config["protocol_id"], "status": "COMPLETE_NO_UPDATE", "raw_sha256": sha(raw_path), "raw_rows": len(raw), "optimizer_steps": 0, "backward_performed": False, "generation_performed": False, "final_audit_opened": False})
if __name__ == "__main__": main()
