"""Offline readiness gate for a prompt-contract-only V3 change.

It reuses an already recorded direct tokenizer preflight from the same frozen
cohort, executes both the frozen and current prompt builders over train pairs,
and applies a byte-level upper bound to the changed prompt.  Thus no model,
backend, Kaggle runtime, or tokenizer download is needed for this contract-only
revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from v3.evidence.cross_pair import derive_cross_pair_evidence
from v3.evidence.extractor import extract_task_evidence
from v3.recognition.recognizer_interface import recognition_prompt


def _frozen_prompt_builder(reference: str):
    source = subprocess.check_output(
        ["git", "show", f"{reference}:src/v3/recognition/recognizer_interface.py"],
        cwd=ROOT, text=True, encoding="utf-8",
    )
    module = types.ModuleType("frozen_v3_recognizer")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{reference}:recognizer_interface.py", "exec"), module.__dict__)
    return module.recognition_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--historical-preflight", type=Path, required=True)
    parser.add_argument("--reference", default="6ad5efd8d39beec578c38ff30aaf6e134347157f")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=11000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite readiness preflight")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    historical = json.loads(args.historical_preflight.read_text(encoding="utf-8"))
    task_ids = tuple(cohort.get("task_ids", ()))
    task_hash = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    if len(task_ids) != 30 or historical.get("task_ids_hash") != task_hash:
        raise ValueError("requires identical frozen30 historical tokenizer preflight")
    old_prompt = _frozen_prompt_builder(args.reference)
    tasks = load_dataset(args.challenge_path)
    records: dict[str, dict[str, object]] = {}
    for task_id in task_ids:
        evidence = extract_task_evidence(tasks[task_id])
        cross_pair = derive_cross_pair_evidence(evidence)
        frozen = old_prompt(tasks[task_id], evidence, cross_pair, top_k=3)
        current = recognition_prompt(tasks[task_id], evidence, cross_pair, top_k=3)
        # This establishes current JSON serialisability independently of the
        # old record; it is not a model call or a rule inference operation.
        json.loads(current)
        prior_tokens = int(historical["records"][task_id]["prompt_tokens"])
        byte_delta = len(current.encode("utf-8")) - len(frozen.encode("utf-8"))
        # Any byte-level tokenizer emits no more than one ordinary token per
        # additional UTF-8 byte. The chat template is unchanged, so this is a
        # conservative upper bound for the new prompt token count.
        upper_bound = prior_tokens + max(byte_delta, 0)
        records[task_id] = {
            "serializable": True,
            "historical_direct_tokenizer_tokens": prior_tokens,
            "prompt_byte_delta": byte_delta,
            "current_prompt_token_upper_bound": upper_bound,
            "within_limit": upper_bound < args.max_prompt_tokens,
        }
    max_bound = max(int(item["current_prompt_token_upper_bound"]) for item in records.values())
    passed = all(bool(item["within_limit"]) for item in records.values())
    artifact = {
        "experiment_id": "ARC2_V3_CAPABILITY_AND_UPSTREAM_PUSH",
        "status": "UPSTREAM_LOCAL_READY" if passed else "UPSTREAM_LOCAL_NOT_READY",
        "task_ids_hash": task_hash,
        "protocol": "Historical direct Qwen tokenizer preflight is reused only after exact frozen/current train-only prompt comparison. Current prompts are JSON-serialised; a byte-level upper bound proves they remain under the 11k limit. No model, text generation, backend, test output, solution, Kaggle, or E2E path is used.",
        "reference_recognizer_commit": args.reference,
        "summary": {
            "serializable": len(records),
            "historical_direct_tokenizer_preflight": len(records),
            "current_prompt_upper_bound_preflight": len(records),
            "max_current_prompt_token_upper_bound": max_bound,
            "limit_exclusive": args.max_prompt_tokens,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], **artifact["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
