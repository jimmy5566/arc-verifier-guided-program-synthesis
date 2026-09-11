from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from inference.llama_cpp_backend import locate_llama_cpp_artifact, materialize_llama_server, server_log_has_backend_error


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_llama_cpp_artifact_requires_manifest_commit_sm89_and_binary_hash(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"; runtime.mkdir()
    model = tmp_path / "qwen3-14b.Q4_K_M.gguf"; model.write_bytes(b"qwen")
    server = runtime / "llama-server"; server.write_bytes(b"server")
    digest = _hash(server)
    monkeypatch.setattr("inference.llama_cpp_backend.EXPECTED_SERVER_SHA256", digest)
    monkeypatch.setattr("inference.llama_cpp_backend.EXPECTED_LLAMA_COMMIT", "commit")
    (runtime / "build_manifest.json").write_text(json.dumps({"commit": "commit", "cuda_architecture": "89", "binaries": {"llama-server": {"sha256": digest}}}), encoding="utf-8")
    artifact = locate_llama_cpp_artifact(tmp_path)
    assert artifact.server_sha256 == digest
    assert artifact.model_sha256 == _hash(model)
    copied = materialize_llama_server(artifact, tmp_path / "working")
    assert _hash(copied) == digest


def test_llama_cpp_artifact_rejects_wrong_sm_or_checksum(monkeypatch, tmp_path: Path):
    runtime = tmp_path / "runtime"; runtime.mkdir()
    (tmp_path / "qwen3-14b.Q4_K_M.gguf").write_bytes(b"qwen")
    (runtime / "llama-server").write_bytes(b"server")
    monkeypatch.setattr("inference.llama_cpp_backend.EXPECTED_LLAMA_COMMIT", "commit")
    (runtime / "build_manifest.json").write_text(json.dumps({"commit": "commit", "cuda_architecture": "75", "binaries": {"llama-server": {"sha256": "wrong"}}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="commit/SM89"):
        locate_llama_cpp_artifact(tmp_path)


def test_backend_error_log_audit_is_explicit(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text("model loaded successfully", encoding="utf-8")
    assert not server_log_has_backend_error(log)
    log.write_text("fatal: unsupported model architecture", encoding="utf-8")
    assert server_log_has_backend_error(log)
