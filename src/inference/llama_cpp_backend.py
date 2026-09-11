"""Offline llama.cpp transport for the frozen Qwen3 V2 inference setup.

This module owns only model-serving infrastructure.  It deliberately does not
change ARC prompts, Macro DSL validation, compiler behaviour, or scoring.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


EXPECTED_SERVER_SHA256 = "3a362477770fc26cc1c127f0d6feccad1315d052f0464b197f251d36d701387e"
EXPECTED_LLAMA_COMMIT = "368b24c355d2e2cf3de2f36f71df18cd910357b1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LlamaCppArtifact:
    model_path: str
    model_sha256: str
    server_path: str
    server_sha256: str
    llama_commit: str
    cuda_architecture: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def locate_llama_cpp_artifact(input_root: Path, *, quantization: str = "Q4_K_M") -> LlamaCppArtifact:
    """Find and checksum the attached model and pinned CUDA llama-server."""
    models = sorted(path for path in input_root.rglob("*.gguf") if "qwen3" in path.name.lower() and quantization.lower() in path.name.lower())
    if not models:
        raise FileNotFoundError(f"missing attached Qwen3 {quantization} GGUF")
    servers = sorted(path for path in input_root.rglob("llama-server") if path.is_file())
    if not servers:
        raise FileNotFoundError("missing attached llama-server binary")
    server = servers[0]
    manifests = sorted(path for path in input_root.rglob("build_manifest.json") if path.is_file() and path.parent == server.parent)
    if not manifests:
        raise FileNotFoundError("llama-server build_manifest.json must be attached alongside the binary")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if manifest.get("commit") != EXPECTED_LLAMA_COMMIT or str(manifest.get("cuda_architecture")) != "89":
        raise RuntimeError("attached llama.cpp build does not match the frozen commit/SM89 provenance")
    recorded_hash = manifest.get("binaries", {}).get("llama-server", {}).get("sha256")
    actual_hash = sha256_file(server)
    if recorded_hash != EXPECTED_SERVER_SHA256 or actual_hash != EXPECTED_SERVER_SHA256:
        raise RuntimeError("attached llama-server SHA-256 failed provenance validation")
    model = models[0]
    return LlamaCppArtifact(str(model), sha256_file(model), str(server), actual_hash, str(manifest["commit"]), str(manifest["cuda_architecture"]))


def materialize_llama_server(artifact: LlamaCppArtifact, destination: Path) -> Path:
    """Copy from a possibly noexec Kaggle input mount to writable storage."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "llama-server"
    if not target.exists() or sha256_file(target) != artifact.server_sha256:
        shutil.copy2(artifact.server_path, target)
    target.chmod(target.stat().st_mode | 0o111)
    if sha256_file(target) != artifact.server_sha256:
        raise RuntimeError("materialized llama-server checksum mismatch")
    return target


class LlamaCppServer:
    """One GPU-pinned, local-only llama-server process."""

    def __init__(self, *, binary: Path, model: Path, endpoint: str, gpu_id: int, context_window: int, log_path: Path) -> None:
        self.binary, self.model, self.endpoint = binary, model, endpoint
        self.gpu_id, self.context_window, self.log_path = gpu_id, context_window, log_path
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any | None = None

    @property
    def environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        return environment

    def start(self, timeout_seconds: float = 180.0) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab")
        port = self.endpoint.rsplit(":", 1)[1]
        command = [str(self.binary), "--model", str(self.model), "--host", "127.0.0.1", "--port", port, "--ctx-size", str(self.context_window), "--n-gpu-layers", "99", "--jinja", "--no-webui"]
        self.process = subprocess.Popen(command, env=self.environment, stdout=self._log_handle, stderr=self._log_handle)
        started = time.perf_counter()
        while time.perf_counter() - started < timeout_seconds:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited before health check (exitcode={self.process.returncode})")
            try:
                with urlopen(f"{self.endpoint}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(0.25)
        raise TimeoutError("llama-server did not pass /health")

    def chat(self, *, prompt: str, max_tokens: int, temperature: float, top_p: float, seed: int) -> dict[str, Any]:
        request_body = {"messages": [{"role": "user", "content": prompt}], "temperature": temperature, "top_p": top_p, "seed": seed, "max_tokens": max_tokens, "stream": False}
        request = Request(f"{self.endpoint}/v1/chat/completions", data=json.dumps(request_body).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=240) as response:
                if response.status != 200:
                    raise RuntimeError(f"llama-server returned unexpected HTTP {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"llama-server HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


def server_log_has_backend_error(log_path: Path) -> bool:
    if not log_path.exists():
        return True
    contents = log_path.read_text(encoding="utf-8", errors="replace")
    return bool(re.search(r"\b(fatal|unsupported|failed to load|error loading|error:|assertion failed)\b", contents, flags=re.IGNORECASE))
