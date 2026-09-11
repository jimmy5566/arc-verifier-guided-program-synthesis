"""One local, structured-output readiness measurement for Qwen3."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen


def main() -> None:
    payload = {
        "model": "qwen3:14b", "prompt": "Return an object with boolean field ok set to true.",
        "stream": False, "think": False,
        "format": {"type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "boolean"}}},
        "options": {"temperature": 0, "seed": 0, "num_predict": 32},
    }
    started = perf_counter()
    request = Request("http://127.0.0.1:11434/api/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode())
    result["wall_seconds"] = perf_counter() - started
    eval_duration = result.get("eval_duration", 0)
    result["tokens_per_second"] = (result.get("eval_count", 0) / (eval_duration / 1e9)) if eval_duration else None
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/ollama_qwen3_sanity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
