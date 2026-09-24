"""Local controller for the bounded one-RTX-5090 Phase-1 experiment.

The GPU Pod has no durable volume.  This controller deliberately dispatches
one predeclared run at a time, fetches its immutable remote artifact through
the SSH terminal channel, verifies the archive SHA256 locally, then performs
the post-freeze development score locally before it dispatches the next run.
It never opens evaluation solutions on the GPU Pod.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_5090_blackwell_unleashed_phase1_queue import _candidate_grid_set, _jaccard, _load_queue


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _quote(value: str | Path) -> str:
    import shlex

    return shlex.quote(str(value))


def _ssh_base(args: argparse.Namespace) -> list[str]:
    # -F NUL bypasses a user-local SSH config with unrelated ACLs.  -tt is
    # required by the RunPod terminal gateway; no key material is logged.
    return ["ssh", "-F", "NUL", "-tt", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-i", str(args.identity_file), args.ssh_target]


def _remote(args: argparse.Namespace, command: str, *, timeout: int | None = None) -> str:
    completed = subprocess.run(_ssh_base(args) + [command], text=True, capture_output=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"REMOTE_COMMAND_FAILED exit={completed.returncode}: {completed.stdout}{completed.stderr}")
    return completed.stdout


def _copy_bytes_to_remote(args: argparse.Namespace, *, payload: bytes, remote_path: Path) -> None:
    encoded = base64.b64encode(payload).decode("ascii")
    command = f"mkdir -p {_quote(remote_path.parent)}; umask 077; base64 -d > {_quote(remote_path)}"
    completed = subprocess.run(_ssh_base(args) + [command], input=encoded, text=True, capture_output=True, timeout=300)
    if completed.returncode:
        raise RuntimeError(f"REMOTE_UPLOAD_FAILED path={remote_path}: {completed.stdout}{completed.stderr}")


def _stage_source(args: argparse.Namespace, commit: str) -> None:
    check = _remote(args, f"if [ -e {_quote(args.remote_repo)} ]; then echo EXISTS; fi")
    if "EXISTS" in check:
        current = _remote(args, f"test -f {_quote(args.remote_repo / '.arc2-source-commit')} && cat {_quote(args.remote_repo / '.arc2-source-commit')} || true").strip()
        if current == commit:
            return
        raise RuntimeError(f"REMOTE_SOURCE_PATH_EXISTS_WITH_DIFFERENT_OR_UNKNOWN_CONTENT: {args.remote_repo}")
    archive = subprocess.check_output(["git", "archive", "--format=tar", commit], cwd=ROOT)
    encoded = base64.b64encode(archive).decode("ascii")
    command = (
        f"mkdir -p {_quote(args.remote_repo)}; "
        f"base64 -d | tar -xf - -C {_quote(args.remote_repo)}; "
        f"printf %s {_quote(commit)} > {_quote(args.remote_repo / '.arc2-source-commit')}"
    )
    completed = subprocess.run(_ssh_base(args) + [command], input=encoded, text=True, capture_output=True, timeout=600)
    if completed.returncode:
        raise RuntimeError(f"REMOTE_SOURCE_STAGE_FAILED: {completed.stdout}{completed.stderr}")


def _stage_challenge(args: argparse.Namespace) -> None:
    local_digest = _sha256(args.local_challenge)
    probe = _remote(args, f"if [ -f {_quote(args.remote_challenge)} ]; then sha256sum {_quote(args.remote_challenge)} | awk '{{print $1}}'; fi").strip()
    if probe == local_digest:
        return
    if probe:
        raise RuntimeError("REMOTE_CHALLENGE_HASH_MISMATCH: refusing to overwrite existing input")
    _copy_bytes_to_remote(args, payload=args.local_challenge.read_bytes(), remote_path=args.remote_challenge)
    verified = _remote(args, f"sha256sum {_quote(args.remote_challenge)} | awk '{{print $1}}'").strip()
    if verified != local_digest:
        raise RuntimeError("REMOTE_CHALLENGE_UPLOAD_HASH_MISMATCH")


def _fetch_archive(args: argparse.Namespace, remote_run: Path, destination: Path) -> str:
    marker_begin = "__ARC2_PHASE1_ARCHIVE_BEGIN__"
    marker_end = "__ARC2_PHASE1_ARCHIVE_END__"
    command = (
        "set -e; tmp=$(mktemp); "
        f"tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner -C {_quote(remote_run.parent)} -cf - {_quote(remote_run.name)} | gzip -n > \"$tmp\"; "
        "printf '%s' '" + marker_begin + "'; base64 -w0 \"$tmp\"; "
        "printf '\n%s %s\n' '" + marker_end + "' \"$(sha256sum \"$tmp\" | awk '{print $1}')\"; rm -f \"$tmp\""
    )
    output = _remote(args, command, timeout=600)
    matched = re.search(marker_begin + r"\s*([A-Za-z0-9+/=\s]+?)\s*" + marker_end + r"\s+([0-9a-f]{64})", output)
    if matched is None:
        raise RuntimeError("REMOTE_ARCHIVE_PROTOCOL_FAILED: terminal output did not contain a complete archive marker")
    payload = re.sub(r"\s+", "", matched.group(1))
    data = base64.b64decode(payload, validate=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    digest = _sha256(destination)
    if digest != matched.group(2):
        raise RuntimeError(f"REMOTE_ARCHIVE_SHA256_MISMATCH expected={matched.group(2)} actual={digest}")
    return digest


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        handle.extractall(destination, filter="data")


def _score_local(args: argparse.Namespace, run_dir: Path) -> None:
    evaluation = run_dir / "evaluation"
    remote_report = evaluation / "report_remote_pending.json"
    report = evaluation / "report.json"
    if report.is_file():
        report.replace(remote_report)
    command = [
        str(args.local_python), str(ROOT / "scripts" / "score_eval3_blackwell_unleashed_phase1.py"),
        "--manifest", str(ROOT / "governance" / "benchmarks" / "legacy-eval3" / "eval3_manifest.json"),
        "--candidates", str(run_dir / "candidates_frozen.json"),
        "--challenge-path", str(args.local_challenge),
        "--solutions-path", str(args.local_solutions),
        "--output", str(report),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"LOCAL_POST_FREEZE_SCORE_FAILED: {completed.stdout}{completed.stderr}")
    _atomic_json(
        run_dir / "offpod_backup.json",
        {
            "archive_sha256": _sha256(run_dir.parent / f"{run_dir.name}.tar.gz"),
            "candidate_sha256": _sha256(run_dir / "candidates_frozen.json"),
            "local_post_freeze_score": str(report.relative_to(run_dir)),
            "remote_pending_report_preserved": str(remote_report.relative_to(run_dir)),
        },
    )


def _verify_remote_run_hashes(run_dir: Path) -> None:
    hashes = json.loads((run_dir / "hashes.json").read_text(encoding="utf-8")).get("files", {})
    candidate = "candidates_frozen.json"
    if hashes.get(candidate) != _sha256(run_dir / candidate):
        raise RuntimeError("LOCAL_CANDIDATE_HASH_DOES_NOT_MATCH_REMOTE_FROZEN_HASH")
    required = {"manifest.json", "config_resolved.json", "environment.json", "telemetry.json", "events.jsonl", "evaluation/report.json"}
    missing = required - set(hashes)
    if missing:
        raise RuntimeError(f"REMOTE_FROZEN_ARTIFACT_HASHES_MISSING: {sorted(missing)}")


def _local_summary(local_root: Path, run_ids: list[str]) -> None:
    records = {run_id: json.loads((local_root / run_id / "candidates_frozen.json").read_text(encoding="utf-8")) for run_id in run_ids}
    serial = _candidate_grid_set(records[run_ids[0]])
    runs: dict[str, Any] = {}
    for run_id in run_ids:
        telemetry = json.loads((local_root / run_id / "telemetry.json").read_text(encoding="utf-8"))["aggregate"]
        report = json.loads((local_root / run_id / "evaluation" / "report.json").read_text(encoding="utf-8"))
        runs[run_id] = {
            "generation_seconds": telemetry["generation_seconds"],
            "total_wall_seconds": telemetry["total_wall_seconds"],
            "generation_tokens_per_second": telemetry["generation_tokens_per_second"],
            "peak_allocated_vram_mb": telemetry["peak_allocated_bytes"] / (1024 * 1024),
            "peak_reserved_vram_mb": telemetry["peak_reserved_bytes"] / (1024 * 1024),
            "ANY_OF_K": report["ANY_OF_K"],
            "candidate_jaccard_vs_serial": _jaccard(serial, _candidate_grid_set(records[run_id])),
        }
    promoted = min(runs.items(), key=lambda item: (item[1]["generation_seconds"], item[1]["total_wall_seconds"], item[0]))
    _atomic_json(local_root / "phase1_local_summary.json", {"runs": runs, "promotion": {"run_id": promoted[0], "generation_speedup_vs_serial": runs[run_ids[0]]["generation_seconds"] / max(promoted[1]["generation_seconds"], 1e-9)}})


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--remote-repo", type=Path, required=True)
    parser.add_argument("--remote-python", type=Path, required=True)
    parser.add_argument("--remote-runtime-root", type=Path, default=Path("/root/arc-runtime"))
    parser.add_argument("--remote-model", type=Path, required=True)
    parser.add_argument("--remote-challenge", type=Path, required=True)
    parser.add_argument("--remote-ptxas", type=Path, required=True)
    parser.add_argument("--local-challenge", type=Path, required=True)
    parser.add_argument("--local-solutions", type=Path, required=True)
    parser.add_argument("--local-artifact-root", type=Path, required=True)
    parser.add_argument("--local-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--stage-source", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    for path in (args.identity_file, args.local_challenge, args.local_solutions, args.local_python):
        if not path.is_file():
            raise FileNotFoundError(f"local controller input is missing: {path}")
    queue_path = ROOT / "governance" / "queues" / "5090-unleashed-phase1-single-gpu-v1.json"
    queue, specs = _load_queue(queue_path)
    if args.stage_source:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        _stage_source(args, commit)
    required_remote = [args.remote_repo / "scripts" / "run_5090_blackwell_unleashed_phase1_queue.py", args.remote_python, args.remote_model / "config.json", args.remote_ptxas]
    _remote(args, "test " + " -a ".join(f"-f {_quote(path)}" for path in required_remote))
    _stage_challenge(args)
    local_root = args.local_artifact_root / queue["queue_id"]
    local_root.mkdir(parents=True, exist_ok=False)
    for spec in sorted(specs, key=lambda item: item.queue_position):
        remote_run = args.remote_runtime_root / "active_run" / queue["queue_id"] / spec.run_id
        command = " ".join(
            _quote(part)
            for part in [
                args.remote_python,
                args.remote_repo / "scripts" / "run_5090_blackwell_unleashed_phase1_queue.py",
                "--queue-config", args.remote_repo / "governance" / "queues" / "5090-unleashed-phase1-single-gpu-v1.json",
                "--bootstrap-script", args.remote_repo / "scripts" / "bootstrap_5090_research_env.sh",
                "--sync-script", args.remote_repo / "scripts" / "sync_arc2_run_to_persistent.sh",
                "--manifest", args.remote_repo / "governance" / "benchmarks" / "legacy-eval3" / "eval3_manifest.json",
                "--reference-config", args.remote_repo / "governance" / "benchmarks" / "legacy-eval3" / "reference_ttt_config_frozen.json",
                "--challenge-path", args.remote_challenge,
                "--solutions-path", args.remote_challenge,
                "--model-path", args.remote_model,
                "--native-config-dir", args.remote_repo / "configs" / "nvarc_native_846d0198",
                "--ptxas-path", args.remote_ptxas,
                "--runtime-root", args.remote_runtime_root,
                "--repo-dir", args.remote_repo,
                "--python", args.remote_python,
                "--run-id", spec.run_id,
                "--skip-persistent-sync",
                "--defer-local-scoring",
            ]
        )
        # The remote runner is target-blind until its candidates are frozen. It
        # receives the challenge path in the solutions slot only because local
        # scoring is explicitly deferred and performed from the verified local
        # archive below.
        _remote(args, command)
        archive = local_root / f"{spec.run_id}.tar.gz"
        _fetch_archive(args, remote_run, archive)
        _safe_extract(archive, local_root)
        run_dir = local_root / spec.run_id
        _verify_remote_run_hashes(run_dir)
        _score_local(args, run_dir)
        print(json.dumps({"event": "OFFPOD_BACKUP_AND_LOCAL_SCORE_PASS", "run_id": spec.run_id, "archive": str(archive), "archive_sha256": _sha256(archive)}, sort_keys=True), flush=True)
    _local_summary(local_root, [spec.run_id for spec in sorted(specs, key=lambda item: item.queue_position)])
    print(json.dumps({"event": "OFFPOD_PHASE1_QUEUE_COMPLETE", "queue_id": queue["queue_id"], "local_artifacts": str(local_root)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
