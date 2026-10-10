#!/usr/bin/env python3
"""One-shot, externally capped launcher for the reviewed E03 V3 worker."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

PROTOCOL_ID = "E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN"


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temp, path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_lock_path(output_root: Path, authorization_id: str) -> Path:
    identity = hashlib.sha256(authorization_id.encode("utf-8")).hexdigest()[:24]
    return output_root.parent / ".e03_consumed_authorizations" / f"{identity}.lock"


def default_run_lock_path(output_root: Path) -> Path:
    identity = hashlib.sha256(str(output_root).encode("utf-8")).hexdigest()[:24]
    return output_root.parent / ".e03_active_runs" / f"{identity}.lock"


def default_log_path(output_root: Path) -> Path:
    return output_root.parent / f"{output_root.name}.launcher.log"


def dependency_preflight(modules: list[str]) -> None:
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("E03_V3_DEPENDENCY_MISSING:" + ",".join(missing))


def acquire_lock(lock_path: Path, authorization_id: str, command: list[str], kind: str) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "authorization_id": authorization_id,
        "kind": kind,
        "launcher_pid": os.getpid(),
        "command": command,
        "created_at_unix": time.time(),
        "consumed": True,
    }
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")


def failure_receipt(output_root: Path, *, error_class: str, started: float, log_path: Path, detail: str | None = None) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    receipt = output_root / "TERMINAL_RECEIPT.json"
    if receipt.exists():
        return
    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "FAILED_OR_PARTIAL_NO_UPDATE",
        "failure_scope": "INFRASTRUCTURE_OR_WORKER_FAILURE",
        "error_class": error_class,
        "elapsed_seconds": time.monotonic() - started,
        "launcher_log_path": str(log_path),
        "launcher_log_sha256": sha(log_path) if log_path.is_file() else None,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "generation_calls": 0,
        "final_audit_opened": False,
        "retry": False,
    }
    if detail:
        payload["detail"] = detail
    atomic(receipt, payload)


def terminate_process_group(process: subprocess.Popen, grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=grace_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap-seconds", type=int, default=1800)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument("--run-lock-path", type=Path)
    parser.add_argument("--required-module", action="append", default=[])
    parser.add_argument("worker", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    repo_root = args.repo_root.resolve()
    log_path = (args.log_path or default_log_path(output_root)).resolve()
    lock_path = (args.lock_path or default_lock_path(output_root, args.authorization_id)).resolve()
    run_lock_path = (args.run_lock_path or default_run_lock_path(output_root)).resolve()
    command = args.worker[1:] if args.worker and args.worker[0] == "--" else []
    if not command or not 0 < args.cap_seconds <= 1800 or not args.authorization_id.strip():
        raise SystemExit("E03_V3_EXTERNAL_CAP_OR_COMMAND_INVALID")
    if lock_path == run_lock_path:
        raise SystemExit("E03_V3_LOCK_IDENTITY_COLLISION")
    if not repo_root.is_dir():
        raise SystemExit("E03_V3_REPO_ROOT_INVALID")
    # Create every parent before opening the log; external shell redirection is unnecessary.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        raise SystemExit("E03_V3_FROZEN_OUTPUT_ALREADY_EXISTS")
    started = time.monotonic()
    try:
        acquire_lock(run_lock_path, args.authorization_id, command, "OUTPUT_RUN_IDENTITY")
    except FileExistsError:
        raise SystemExit("E03_V3_DUPLICATE_LAUNCH_BLOCKED")
    try:
        acquire_lock(lock_path, args.authorization_id, command, "ONE_SHOT_AUTHORIZATION")
    except FileExistsError:
        # This run never started, so release only the run lock created above;
        # the existing authorization lock remains immutable evidence.
        run_lock_path.unlink(missing_ok=True)
        raise SystemExit("E03_V3_DUPLICATE_LAUNCH_BLOCKED")
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    environment["E03_EXTERNAL_CAP_ENFORCED"] = "1"
    try:
        dependency_preflight(args.required_module)
        with log_path.open("xb") as log_handle:
            log_handle.write(json.dumps({
                "protocol_id": PROTOCOL_ID,
                "authorization_id": args.authorization_id,
                "repo_root": str(repo_root),
                "command": command,
            }, sort_keys=True).encode("utf-8") + b"\n")
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                start_new_session=True,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            try:
                return_code = process.wait(timeout=args.cap_seconds)
            except subprocess.TimeoutExpired:
                terminate_process_group(process)
                failure_receipt(output_root, error_class="E03_V3_EXTERNAL_RUNTIME_CAP_EXCEEDED", started=started, log_path=log_path)
                return 124
        receipt = output_root / "TERMINAL_RECEIPT.json"
        if return_code != 0:
            failure_receipt(output_root, error_class=f"E03_V3_WORKER_EXIT_{return_code}", started=started, log_path=log_path)
            return return_code
        if not receipt.is_file():
            failure_receipt(output_root, error_class="E03_V3_WORKER_EXITED_WITHOUT_TERMINAL_RECEIPT", started=started, log_path=log_path)
            return 2
        atomic(output_root / "LAUNCHER_RECEIPT.json", {
            "protocol_id": PROTOCOL_ID,
            "status": "WORKER_TERMINAL_RECEIPT_PRESERVED",
            "authorization_id": args.authorization_id,
            "worker_return_code": return_code,
            "terminal_receipt_sha256": sha(receipt),
            "launcher_log_path": str(log_path),
            "launcher_log_sha256": sha(log_path),
            "elapsed_seconds": time.monotonic() - started,
            "retry": False,
        })
        return 0
    except Exception as exc:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab") as handle:
                handle.write(traceback.format_exc().encode("utf-8", "replace"))
        finally:
            failure_receipt(output_root, error_class=f"{type(exc).__name__}:{exc}", started=started, log_path=log_path)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
