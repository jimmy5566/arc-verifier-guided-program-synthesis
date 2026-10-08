#!/usr/bin/env python3
"""Target-blind 12-item decoder characterization worker.

This worker has no correctness scorer: held-out test outputs are not accessed,
and its output schema intentionally has no exact_grid_match field.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.arc2_token_grid_parser import parse_generated_token_ids, tokenizer_token_contract
from scripts.collect_capability_repair_baseline_v1 import observation, prompt, read_rows, sha, sha_bytes, dig
from scripts.decoder_characterization_launch_v2 import atomic_json, terminal_failure, validate_contract, validate_preflight

def character_row(surface: str, record: dict[str, Any], rendered_prompt: str, token_ids: list[int], contract: Any, model_identity: dict[str, Any]) -> dict[str, Any]:
    """Return parser evidence only; never inspect ``task.test[*].output``."""
    extracted = parse_generated_token_ids(token_ids, contract)
    grid = extracted.grid
    return {"surface": surface, "episode_id": record["episode_id"], "role": record["role"],
            "observation_sha256": dig(observation(record)), "prompt_sha256": sha_bytes(rendered_prompt.encode("utf-8")),
            "generated_token_ids": extracted.generated_token_ids, "generated_length": extracted.generated_length,
            "termination_status": extracted.termination_status, "eos_observed": extracted.eos_observed,
            "trailing_pad_count": extracted.trailing_pad_count, "content_token_ids": extracted.content_token_ids,
            "parse_reason": extracted.parse_reason, "parse_valid": grid is not None,
            "canonical_prediction_sha256": dig({"grid": grid} if grid is not None else {"invalid_output": True}),
            "model_identity": model_identity, "parser_contract_identity": contract.as_dict()}

def selected_rows(target: Path, retention: Path, cohort_path: Path) -> list[tuple[str, dict[str, Any]]]:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8")); ids = cohort["episode_ids"]
    all_rows = read_rows(target, {"TARGETED_EVALUATION", "TARGETED_COMPOSITION"}) + read_rows(retention, {"RETENTION_SENTINEL"})
    indexed = {item["episode_id"]: item for item in all_rows}
    if set(ids) != set(indexed).intersection(ids): raise RuntimeError("COHORT_EPISODE_MISSING")
    result = [("RETENTION_SENTINEL" if indexed[item]["role"] == "RETENTION_SENTINEL" else "TARGET_DEV", indexed[item]) for item in ids]
    if len(result) != 12: raise RuntimeError("COHORT_SELECTION_MISMATCH")
    return result

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-contract", type=Path, required=True); parser.add_argument("--governor-review", type=Path)
    parser.add_argument("--preflight", action="store_true"); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(); launch: dict[str, Any] | None = None
    try:
        # Bind the invoked interpreter path itself.  Resolving it would erase
        # the frozen venv identity when Python is a symlink to a system binary.
        actual = [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
        environment = {key: os.environ.get(key, "") for key in ("PYTHONHASHSEED", "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES")}
        # Preflight proves binding without consuming the launch nonce or loading a model.
        launch = validate_contract(args.launch_contract, argv=actual, environment=environment, require_review=None if args.preflight else args.governor_review, consume_nonce=not args.preflight)
        if args.preflight:
            import subprocess
            source_commit = subprocess.check_output(["git", "-C", launch["source_root"], "rev-parse", "HEAD"], text=True).strip()
            payload = {"schema_version": 1, "status": "PASS_NO_MODEL_IMPORT", "source_commit": source_commit, "worker_source_commit": launch["worker_source_commit"], "no_target_access": True, "model_loaded": False,
                       "tokenizer_ids": {"digits": list(range(10)), "newline": 10, "eos": 15, "pad": 13}, "cohort_sha256": launch["cohort"]["sha256"],
                       "parser_sha256": launch["immutable_files"]["token_parser"]["sha256"], "worker_sha256": launch["immutable_files"]["worker"]["sha256"]}
            atomic_json(args.receipt, payload); return 0
        validate_preflight(launch, source_commit=launch["worker_source_commit"])
        # All model imports occur only after immutable admission succeeds.
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch
        runtime = launch["runtime"]
        tokenizer = AutoTokenizer.from_pretrained(launch["checkpoint"]["base_path"], local_files_only=True); tokenizer.padding_side = "left"; tokenizer.pad_token_id = 13
        contract = tokenizer_token_contract(tokenizer, eos_token_id=15, pad_token_id=13)
        c = json.loads(Path(launch["inference_contract_path"]).read_text(encoding="utf-8"))
        model = AutoModelForCausalLM.from_pretrained(launch["checkpoint"]["base_path"], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=runtime["attention_backend"]).to("cuda:0")
        model = PeftModel.from_pretrained(model, launch["checkpoint"]["adapter_path"], is_trainable=False).eval()
        rows = []
        for surface, record in selected_rows(Path(launch["datasets"]["target_dev_path"]), Path(launch["datasets"]["retention_path"]), Path(launch["cohort"]["path"])):
            rendered = prompt(record, c); encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda:0"); width = int(encoded["input_ids"].shape[-1])
            with torch.inference_mode(): generated = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=runtime["max_new_tokens"], eos_token_id=15, pad_token_id=13)
            rows.append(character_row(surface, record, rendered, [int(x) for x in generated[0][width:].detach().cpu().tolist()], contract, launch["checkpoint"]))
        if any("exact_grid_match" in row for row in rows): raise RuntimeError("FORBIDDEN_TARGET_DERIVED_FIELD")
        atomic_json(args.output, {"schema_version": 1, "status": "COMPLETE_TARGET_BLIND", "rows": rows, "completed_episode_count": len(rows), "row_sha256": dig(rows)})
        atomic_json(args.receipt, {"schema_version": 1, "status": "SUCCESS", "output_sha256": sha(args.output), "items": len(rows), "scientific_training_started": False})
        return 0
    except Exception as exc:
        terminal_failure(launch, str(exc)); raise

if __name__ == "__main__": raise SystemExit(main())
