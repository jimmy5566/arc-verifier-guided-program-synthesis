#!/usr/bin/env python3
"""ASCII-safe, resumable byte transport for RunPod's forced-PTY gateway.

This is transport code only.  It never opens JSONL, imports ML libraries, or
inspects ARC content.  A deterministic gzip envelope is split into ordered
base64 chunks; the remote endpoint independently verifies every chunk before
acknowledging it and atomically publishes only a fully verified reconstruction.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

CHUNK_SIZE = 262_144
SCHEMA = 1


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_manifest(encoded: str) -> dict[str, Any]:
    try:
        value = json.loads(base64.b64decode(encoded, validate=True).decode("utf-8", errors="strict"))
    except Exception as exc:
        raise RuntimeError(f"TRANSPORT_MANIFEST_DECODE_ERROR:{type(exc).__name__}") from exc
    required = {"schema_version", "filename", "uncompressed_bytes", "uncompressed_sha256", "envelope_sha256", "chunk_size", "chunk_count", "chunks"}
    if set(value) != required or value["schema_version"] != SCHEMA:
        raise RuntimeError("TRANSPORT_MANIFEST_FIELDS_INVALID")
    if Path(str(value["filename"])).name != value["filename"] or not str(value["filename"]).endswith(".jsonl"):
        raise RuntimeError("TRANSPORT_FILENAME_INVALID")
    chunks = value["chunks"]
    if not isinstance(chunks, list) or len(chunks) != value["chunk_count"] or value["chunk_count"] < 1:
        raise RuntimeError("TRANSPORT_CHUNK_COUNT_INVALID")
    for index, item in enumerate(chunks):
        if set(item) != {"index", "bytes", "sha256"} or item["index"] != index or item["bytes"] < 1 or len(item["sha256"]) != 64:
            raise RuntimeError("TRANSPORT_CHUNK_MANIFEST_INVALID")
    return value


def build(source: Path, manifest_path: Path, envelope_path: Path, chunk_size: int = CHUNK_SIZE) -> dict[str, Any]:
    if chunk_size < 1024 or chunk_size > 262_144:
        raise RuntimeError("TRANSPORT_CHUNK_SIZE_OUT_OF_BOUNDS")
    raw = source.read_bytes()
    envelope = gzip.compress(raw, compresslevel=9, mtime=0)
    chunks = [envelope[offset:offset + chunk_size] for offset in range(0, len(envelope), chunk_size)]
    value = {"schema_version": SCHEMA, "filename": source.name, "uncompressed_bytes": len(raw), "uncompressed_sha256": sha_bytes(raw),
             "envelope_sha256": sha_bytes(envelope), "chunk_size": chunk_size, "chunk_count": len(chunks),
             "chunks": [{"index": index, "bytes": len(chunk), "sha256": sha_bytes(chunk)} for index, chunk in enumerate(chunks)]}
    for path, content in ((manifest_path, json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"), (envelope_path, envelope)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return value


def stage_dir(root: Path, manifest: dict[str, Any]) -> Path:
    return root / (manifest["filename"] + "." + manifest["envelope_sha256"][:16])


def receive(staging_root: Path, manifest: dict[str, Any], index: int, encoded_chunk: str) -> dict[str, Any]:
    if not 0 <= index < int(manifest["chunk_count"]):
        raise RuntimeError("TRANSPORT_CHUNK_INDEX_INVALID")
    try:
        chunk = base64.b64decode(encoded_chunk, validate=True)
    except Exception as exc:
        raise RuntimeError("TRANSPORT_CHUNK_BASE64_INVALID") from exc
    expected = manifest["chunks"][index]
    if len(chunk) != expected["bytes"] or sha_bytes(chunk) != expected["sha256"]:
        raise RuntimeError("TRANSPORT_CHUNK_HASH_OR_SIZE_MISMATCH")
    stage = stage_dir(staging_root, manifest); chunks = stage / "chunks"; chunks.mkdir(parents=True, exist_ok=True)
    frozen_manifest = stage / "manifest.json"; encoded_manifest = canonical(manifest)
    if frozen_manifest.exists():
        if frozen_manifest.read_bytes() != encoded_manifest: raise RuntimeError("TRANSPORT_MANIFEST_CONFLICT")
    else:
        fd = os.open(frozen_manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle: handle.write(encoded_manifest); handle.flush(); os.fsync(handle.fileno())
    target = chunks / f"{index:08d}.chunk"
    existing = sorted(int(path.stem) for path in chunks.glob("*.chunk") if path.stem.isdigit())
    expected_next = next((candidate for candidate in range(int(manifest["chunk_count"])) if candidate not in existing), int(manifest["chunk_count"]))
    if target.exists():
        if sha_file(target) != expected["sha256"] or target.stat().st_size != expected["bytes"]: raise RuntimeError("TRANSPORT_DUPLICATE_CONFLICT")
        return {"status": "IDEMPOTENT_ACK", "index": index, "next_index": expected_next, "chunk_sha256": expected["sha256"]}
    if index != expected_next: raise RuntimeError(f"TRANSPORT_OUT_OF_ORDER_EXPECTED_{expected_next}")
    with tempfile.NamedTemporaryFile("wb", dir=chunks, delete=False) as handle:
        handle.write(chunk); handle.flush(); os.fsync(handle.fileno()); tmp = Path(handle.name)
    if sha_file(tmp) != expected["sha256"]: tmp.unlink(missing_ok=True); raise RuntimeError("TRANSPORT_TEMP_HASH_MISMATCH")
    os.replace(tmp, target)
    return {"status": "ACK", "index": index, "next_index": index + 1, "chunk_sha256": expected["sha256"]}


def finalize(staging_root: Path, destination: Path, manifest: dict[str, Any], final_audit: bool) -> dict[str, Any]:
    if destination.exists():
        if destination.is_file() and destination.stat().st_size == manifest["uncompressed_bytes"] and sha_file(destination) == manifest["uncompressed_sha256"]:
            return {"status": "ALREADY_CORRECT", "filename": manifest["filename"], "content_deserialized": False}
        raise RuntimeError("TRANSPORT_DESTINATION_CONFLICT_REFUSE_OVERWRITE")
    stage = stage_dir(staging_root, manifest); chunks = stage / "chunks"
    expected_paths = [chunks / f"{index:08d}.chunk" for index in range(int(manifest["chunk_count"]))]
    if set(chunks.glob("*.chunk")) != set(expected_paths): raise RuntimeError("TRANSPORT_CHUNK_SET_INCOMPLETE_OR_EXTRA")
    envelope = b"".join(path.read_bytes() for path in expected_paths)
    if sha_bytes(envelope) != manifest["envelope_sha256"]: raise RuntimeError("TRANSPORT_ENVELOPE_HASH_MISMATCH")
    temporary: Path | None = None
    try:
        raw = gzip.decompress(envelope)  # byte-stream only; never JSON-deserializes content
        if len(raw) != manifest["uncompressed_bytes"] or sha_bytes(raw) != manifest["uncompressed_sha256"]: raise RuntimeError("TRANSPORT_UNCOMPRESSED_IDENTITY_MISMATCH")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno()); temporary = Path(handle.name)
        if sha_file(temporary) != manifest["uncompressed_sha256"]: raise RuntimeError("TRANSPORT_TEMP_DESTINATION_HASH_MISMATCH")
        os.replace(temporary, destination); temporary = None
        if final_audit: os.chmod(destination, 0o444)
        return {"status": "PUBLISHED", "filename": manifest["filename"], "bytes": manifest["uncompressed_bytes"], "sha256": manifest["uncompressed_sha256"], "model_accessed": False, "optimizer_accessed": False, "content_deserialized": False, "read_only": final_audit}
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build"); b.add_argument("--source", type=Path, required=True); b.add_argument("--manifest", type=Path, required=True); b.add_argument("--envelope", type=Path, required=True); b.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    r = sub.add_parser("receive"); r.add_argument("--staging-root", type=Path, required=True); r.add_argument("--manifest-b64", required=True); r.add_argument("--index", type=int, required=True); r.add_argument("--chunk-b64", required=True)
    f = sub.add_parser("finalize"); f.add_argument("--staging-root", type=Path, required=True); f.add_argument("--destination", type=Path, required=True); f.add_argument("--manifest-b64", required=True); f.add_argument("--final-audit", action="store_true")
    args = p.parse_args()
    if args.mode == "build": value = build(args.source, args.manifest, args.envelope, args.chunk_size); print(json.dumps({"status":"BUILT", "manifest_sha256":sha_file(args.manifest), "chunk_count":value["chunk_count"]}, sort_keys=True)); return 0
    manifest = decode_manifest(args.manifest_b64)
    value = receive(args.staging_root, manifest, args.index, args.chunk_b64) if args.mode == "receive" else finalize(args.staging_root, args.destination, manifest, args.final_audit)
    print(json.dumps(value, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
