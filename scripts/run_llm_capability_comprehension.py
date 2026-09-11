"""Run the frozen 40-question registry comprehension benchmark via Ollama."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen

from llm.comprehension import build_comprehension_benchmark, score_comprehension


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"], "properties": {"answer": {"type": "string"}}}


def ask(question: dict[str, object]) -> tuple[str | None, dict[str, object]]:
    prompt = "Choose exactly one canonical primitive ID from the options. Return JSON only.\nQuestion: " + str(question["question"]) + "\nOptions: " + json.dumps(question["options"])
    payload = {"model": "qwen3:14b", "prompt": prompt, "stream": False, "think": False, "format": SCHEMA, "options": {"temperature": 0, "seed": 0, "num_predict": 32, "num_ctx": 4096}}
    request = Request("http://127.0.0.1:11434/api/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=300) as response:
        raw = json.loads(response.read().decode())
    try:
        answer = json.loads(raw["response"])["answer"]
    except (KeyError, TypeError, json.JSONDecodeError):
        answer = None
    return answer if isinstance(answer, str) else None, raw


def main() -> None:
    benchmark = build_comprehension_benchmark()
    answers, details, timings, prompts, outputs = {}, [], [], [], []
    started = perf_counter()
    for question in benchmark["questions"]:
        turn = perf_counter()
        answer, raw = ask(question)
        timings.append(perf_counter() - turn); prompts.append(raw.get("prompt_eval_count", 0)); outputs.append(raw.get("eval_count", 0))
        answers[question["question_id"]] = answer
        details.append({"question_id": question["question_id"], "topic": question["topic"], "answer": answer, "correct": answer == question["answer"]})
    score = score_comprehension(benchmark, answers)
    misunderstood = Counter(item["answer"] for item, question in zip(details, benchmark["questions"]) if not item["correct"] and item["answer"])
    payload = {"model": "qwen3:14b", "temperature": 0, "question_count": benchmark["question_count"], "score": score, "details": details, "most_misunderstood_selected_ids": dict(misunderstood), "mean_call_seconds": sum(timings) / len(timings), "runtime_seconds": perf_counter() - started, "prompt_tokens": sum(prompts), "output_tokens": sum(outputs)}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/llm_capability_comprehension_qwen3_14b.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
