"""Local, no-model preflight for the V3 train-only recognition interface."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from v3.evidence.cross_pair import derive_cross_pair_evidence
from v3.evidence.extractor import extract_task_evidence
from v3.recognition.recognizer_interface import recognition_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=11000)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite frozen preflight")
    from transformers import AutoTokenizer
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    task_ids = tuple(cohort["task_ids"])
    task_hash = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    if len(task_ids) != 30: raise ValueError("requires frozen30 cohort")
    tasks, tokenizer = load_dataset(args.challenge_path), AutoTokenizer.from_pretrained(str(args.tokenizer_path), local_files_only=True)
    records: dict[str, dict[str, object]] = {}
    for task_id in task_ids:
        task = tasks[task_id]
        try:
            evidence = extract_task_evidence(task)
            prompt = recognition_prompt(task, evidence, derive_cross_pair_evidence(evidence), top_k=3)
            # Serialisation is intentionally checked before tokenisation.
            json.loads(prompt)
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True,
                enable_thinking=False, tokenize=True,
            )
            records[task_id] = {"serializable": True, "prompt_tokens": len(tokens), "within_limit": len(tokens) < args.max_prompt_tokens}
        except Exception as exc:  # status is retained in private artifact only
            records[task_id] = {"serializable": False, "error": f"{type(exc).__name__}:{exc}"}
    serializable = sum(bool(item["serializable"]) for item in records.values())
    tokenized = [int(item["prompt_tokens"]) for item in records.values() if "prompt_tokens" in item]
    artifact = {
        "experiment_id": "ARC2_V3_FINAL_ARCHITECTURE_HARDENING",
        "status": "UPSTREAM_LOCAL_READY" if serializable == 30 and len(tokenized) == 30 and max(tokenized) < args.max_prompt_tokens else "UPSTREAM_LOCAL_NOT_READY",
        "task_ids_hash": task_hash,
        "protocol": "Train grids plus deterministic train-derived evidence only. This gate loads a tokenizer only; it does not load model weights, generate text, access a backend, RuleSpec, executor, test grid/output or solution file.",
        "summary": {"serializable": serializable, "tokenizer_preflight": len(tokenized), "max_prompt_tokens": max(tokenized) if tokenized else None, "limit_exclusive": args.max_prompt_tokens},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], **artifact["summary"]}, sort_keys=True))


if __name__ == "__main__": main()
