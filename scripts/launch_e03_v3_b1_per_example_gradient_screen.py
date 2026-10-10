#!/usr/bin/env python3
"""One-job external cap for the separately reviewed E03 V3 diagnostic."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap-seconds", type=int, default=1800)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("worker", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.worker or args.worker[0] != "--" or not 0 < args.cap_seconds <= 1800:
        raise SystemExit("E03_V3_EXTERNAL_CAP_OR_COMMAND_INVALID")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    process = subprocess.Popen(args.worker[1:], start_new_session=True, env={**os.environ, "E03_EXTERNAL_CAP_ENFORCED": "1"})
    while process.poll() is None and time.monotonic() - started < args.cap_seconds:
        time.sleep(0.25)
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        atomic(args.output_root / "TERMINAL_RECEIPT.json", {"protocol_id": "E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN", "status": "FAILED_OR_PARTIAL_NO_UPDATE", "error_class": "E03_V3_EXTERNAL_RUNTIME_CAP_EXCEEDED", "elapsed_seconds": time.monotonic() - started, "optimizer_steps": 0, "parameter_updates": 0, "generation_calls": 0, "final_audit_opened": False, "retry": False})
        raise SystemExit(124)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
