"""Atomic, deterministic D1 cell claims independent of worker ownership.

This is scheduler plumbing only.  A checkpoint is the sole completion signal;
claims merely prevent two workers from computing the same unfinished cell.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any


def cell_key(*, output_id: str, depth: int, view: str) -> str:
    return f"{output_id}:d{depth}:{view}"


def claim_key(*, policy: str, output_id: str, depth: int, view: str) -> str:
    return hashlib.sha256(f"{policy}|{cell_key(output_id=output_id, depth=depth, view=view)}".encode("utf-8")).hexdigest()


def deterministic_order(items: list[tuple[str, str, int, str]]) -> list[tuple[str, str, int, str]]:
    return sorted(items, key=lambda item: (hashlib.sha256((item[0] + "|" + cell_key(output_id=item[1], depth=item[2], view=item[3])).encode("utf-8")).hexdigest(), item))


def _read_claim(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def claim_cell(*, claims_root: Path, policy: str, output_id: str, depth: int, view: str,
               worker_id: str, stale_seconds: float) -> Path | None:
    """Atomically claim a cell; safely reclaim only expired claims.

    An unfinished process retains its claim until the fixed expiry.  A completed
    checkpoint is checked by the caller before and after claiming, so an expired
    claim can never overwrite a valid checkpoint.
    """
    if stale_seconds <= 0:
        raise ValueError("stale_seconds must be positive")
    claims_root.mkdir(parents=True, exist_ok=True)
    path = claims_root / f"{claim_key(policy=policy, output_id=output_id, depth=depth, view=view)}.json"
    payload = {
        "policy": policy, "output_id": output_id, "depth": depth, "view": view,
        "worker_id": worker_id, "host": socket.gethostname(), "claimed_unix": time.time(),
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        old = _read_claim(path)
        claimed = float(old.get("claimed_unix", 0.0)) if old else 0.0
        if time.time() - claimed < stale_seconds:
            return None
        tombstone = path.with_suffix(path.suffix + f".stale.{os.getpid()}")
        try:
            os.replace(path, tombstone)
        except FileNotFoundError:
            return None
        try:
            tombstone.unlink(missing_ok=True)
        except OSError:
            return None
        return claim_cell(claims_root=claims_root, policy=policy, output_id=output_id, depth=depth, view=view,
                          worker_id=worker_id, stale_seconds=stale_seconds)
    try:
        os.write(fd, encoded)
    finally:
        os.close(fd)
    return path


def release_claim(path: Path) -> None:
    path.unlink(missing_ok=True)
