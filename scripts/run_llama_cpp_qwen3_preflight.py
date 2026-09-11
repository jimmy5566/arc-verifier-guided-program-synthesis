"""One-request GPU0 compatibility gate for llama.cpp and the existing Qwen3 GGUF."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import RunnerStatus, atomic_write_json, gpu_observation, inspect_hardware
from inference.llama_cpp_backend import LlamaCppServer, locate_llama_cpp_artifact, materialize_llama_server, server_log_has_backend_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True, help="Read-only directory containing the separately obtained model/runtime artifact.")
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--context-window", type=int, default=12288)
    args = parser.parse_args()
    result_path = args.output_root / "results" / "LLAMA_CPP_QWEN3_PREFLIGHT.json"
    hardware = inspect_hardware()
    result: dict[str, object] = {"experiment_id": "LLAMA_CPP_QWEN3_PREFLIGHT", "hardware": hardware.to_dict(), "worker_to_gpu_mapping": {"0": 0}, "prompt": "Return JSON {\\\"ok\\\": true}", "max_tokens": 32}
    if hardware.status != RunnerStatus.SUCCESS:
        result |= {"status": "INVALID_GPU_RUNTIME", "reason": hardware.reason}
        atomic_write_json(result_path, result)
        print(json.dumps(result)); return
    server = None
    try:
        artifact = locate_llama_cpp_artifact(args.input_root)
        binary = materialize_llama_server(artifact, args.output_root / "checkpoints" / "llama_cpp_runtime")
        log_path = args.output_root / "checkpoints" / "llama_cpp_runtime" / "gpu0_llama_server.log"
        before = gpu_observation(0)
        server = LlamaCppServer(binary=binary, model=Path(artifact.model_path), endpoint="http://127.0.0.1:18080", gpu_id=0, context_window=args.context_window, log_path=log_path)
        server.start()
        response = server.chat(prompt='Return JSON {"ok": true}', max_tokens=32, temperature=0, top_p=1, seed=0)
        after = gpu_observation(0)
        content = response.get("choices", [{}])[0].get("message", {}).get("content")
        usage = response.get("usage", {})
        parsed = json.loads(content) if isinstance(content, str) else None
        completion_tokens = int(usage.get("completion_tokens", 0))
        vram_delta = (after.get("vram_used_mib") or 0) - (before.get("vram_used_mib") or 0)
        result |= {"artifact": artifact.to_dict(), "gpu_before": before, "gpu_after": after, "vram_delta_mib": vram_delta, "http_status": 200, "completion_tokens": completion_tokens, "response_text": content, "response_json": parsed, "server_log": str(log_path), "server_log_backend_error": False, "status": "SUCCESS"}
        if completion_tokens <= 0 or vram_delta < 4096 or parsed is None:
            raise RuntimeError("preflight requires >0 completion tokens, parseable JSON, and >=4096 MiB GPU0 VRAM increase")
    except Exception as exc:
        result |= {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        if server is not None:
            server.stop()
            log_path = Path(result.get("server_log", args.output_root / "checkpoints" / "llama_cpp_runtime" / "gpu0_llama_server.log"))
            result["server_log"] = str(log_path)
            result["server_log_backend_error"] = server_log_has_backend_error(log_path)
            if result.get("status") == "SUCCESS" and result["server_log_backend_error"]:
                result["status"] = "FAILED"; result["reason"] = "llama-server log contains a backend error"
        atomic_write_json(result_path, result)
    print(json.dumps(result))
    if result["status"] != "SUCCESS":
        raise SystemExit("LLAMA_CPP_PREFLIGHT_FAILED")


if __name__ == "__main__":
    main()
