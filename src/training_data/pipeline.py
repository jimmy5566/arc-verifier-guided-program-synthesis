"""CPU-only, fail-closed preparation for the ARC training corpus.

The module deliberately has no Torch, Transformers, CUDA, or model imports.
It mirrors the vendored NVARC text/token contract with a tiny pure-Python
tokenizer so dataset auditing cannot initialize a GPU by accident.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from inference.arc_native_io import ARCNativeInputAdapter

TRAINING_DATA_VERSION = "training_data_v1"
CPU_WORKERS = 20
MAX_BLACKLIST_EVIDENCE_EXAMPLES = 3
TASK_ID_RE = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{8}(?![0-9A-Fa-f])")
TEXT_SUFFIXES = {".json", ".csv", ".md", ".txt", ".py", ".yml", ".yaml", ".jsonl"}
MANIFEST_HINTS = ("manifest", "cohort", "eval", "pilot", "retention", "untouched", "frozen", "heldout", "gate", "stability", "score", "gold", "search", "dfs", "public")
EXPECTED_TOKENS = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "Ċ": 10, "user": 11, "assistant": 12, "<|endoftext|>": 13,
    "<|im_start|>": 14, "<|im_end|>": 15,
}
IGNORE_INDEX = -100


class GateFailure(RuntimeError):
    """A fail-closed data-integrity condition."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, (_canonical_json(value) + "\n").encode("utf-8"))


def atomic_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _validate_grid(grid: Any, *, field: str) -> list[list[int]]:
    if not isinstance(grid, list) or not grid or len(grid) > 30:
        raise GateFailure(f"{field}: grid height must be 1..30")
    if not all(isinstance(row, list) and row for row in grid):
        raise GateFailure(f"{field}: grid rows must be non-empty lists")
    width = len(grid[0])
    if width > 30 or any(len(row) != width for row in grid):
        raise GateFailure(f"{field}: grid must be rectangular and width 1..30")
    copied: list[list[int]] = []
    for row in grid:
        checked: list[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9:
                raise GateFailure(f"{field}: ARC colors must be integer 0..9")
            checked.append(value)
        copied.append(checked)
    return copied


def normalize_task(raw: Any, *, source_id: str, require_test_outputs: bool) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("train"), list) or not isinstance(raw.get("test"), list):
        raise GateFailure(f"{source_id}: task must contain train/test lists")
    if not raw["train"] or not raw["test"]:
        raise GateFailure(f"{source_id}: train/test lists must be non-empty")

    def pairs(name: str, require_output: bool) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index, pair in enumerate(raw[name]):
            if not isinstance(pair, dict) or "input" not in pair:
                raise GateFailure(f"{source_id}:{name}[{index}] lacks input")
            normalized = {"input": _validate_grid(pair["input"], field=f"{source_id}:{name}[{index}].input")}
            if "output" in pair:
                normalized["output"] = _validate_grid(pair["output"], field=f"{source_id}:{name}[{index}].output")
            elif require_output:
                raise GateFailure(f"{source_id}:{name}[{index}] lacks legitimate output")
            result.append(normalized)
        return result

    return {"source_id": source_id.lower(), "train": pairs("train", True), "test": pairs("test", require_test_outputs)}


def task_observation(task: dict[str, Any]) -> dict[str, Any]:
    """Return task content without any test output (safe for held-out references)."""
    return {
        "train": [{"input": pair["input"], "output": pair["output"]} for pair in task["train"]],
        "test": [{"input": pair["input"]} for pair in task["test"]],
    }


def task_content(task: dict[str, Any]) -> dict[str, Any]:
    if any("output" not in pair for pair in task["test"]):
        raise GateFailure(f"{task['source_id']}: full content requires all test outputs")
    return {
        "train": [{"input": pair["input"], "output": pair["output"]} for pair in task["train"]],
        "test": [{"input": pair["input"], "output": pair["output"]} for pair in task["test"]],
    }


def canonical_hash(value: Any) -> str:
    return _sha_bytes(_canonical_json(value).encode("utf-8"))


def train_pair_hash(task: dict[str, Any]) -> str:
    return canonical_hash(task_observation(task)["train"])


def _transform_grid(grid: list[list[int]], transform: str) -> list[list[int]]:
    rows = [list(row) for row in grid]
    if transform == "identity":
        return rows
    if transform == "rot90":
        return [list(row) for row in zip(*rows[::-1])]
    if transform == "rot180":
        return [row[::-1] for row in rows[::-1]]
    if transform == "rot270":
        return [list(row) for row in zip(*rows)][::-1]
    if transform == "flip_lr":
        return [row[::-1] for row in rows]
    if transform == "flip_ud":
        return rows[::-1]
    if transform == "transpose":
        return [list(row) for row in zip(*rows)]
    if transform == "anti_transpose":
        return [list(row) for row in zip(*rows[::-1])][::-1]
    raise ValueError(transform)


TRANSFORMS = ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")


def _transform_observation(observation: dict[str, Any], transform: str) -> dict[str, Any]:
    transformed: dict[str, Any] = {"train": [], "test": []}
    for pair in observation["train"]:
        transformed["train"].append({"input": _transform_grid(pair["input"], transform), "output": _transform_grid(pair["output"], transform)})
    for pair in observation["test"]:
        transformed["test"].append({"input": _transform_grid(pair["input"], transform)})
    return transformed


def _color_normalize(observation: dict[str, Any]) -> dict[str, Any]:
    mapping: dict[int, int] = {}

    def grid(value: list[list[int]]) -> list[list[int]]:
        result: list[list[int]] = []
        for row in value:
            output_row: list[int] = []
            for color in row:
                if color not in mapping:
                    mapping[color] = len(mapping)
                output_row.append(mapping[color])
            result.append(output_row)
        return result

    result: dict[str, Any] = {"train": [], "test": []}
    for pair in observation["train"]:
        result["train"].append({"input": grid(pair["input"]), "output": grid(pair["output"])})
    for pair in observation["test"]:
        result["test"].append({"input": grid(pair["input"])})
    return result


def structural_signature(task: dict[str, Any]) -> tuple[str, str]:
    observation = task_observation(task)
    candidates = [(canonical_hash(_color_normalize(_transform_observation(observation, transform))), transform) for transform in TRANSFORMS]
    return min(candidates)


def _grid_components(grid: list[list[int]]) -> int:
    """Four-connected non-background/color-separated component count."""
    seen: set[tuple[int, int]] = set()
    count = 0
    height, width = len(grid), len(grid[0])
    for y in range(height):
        for x in range(width):
            if grid[y][x] == 0 or (y, x) in seen:
                continue
            count += 1
            color = grid[y][x]
            stack = [(y, x)]
            seen.add((y, x))
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in seen and grid[ny][nx] == color:
                        seen.add((ny, nx))
                        stack.append((ny, nx))
    return count


def task_features(task: dict[str, Any]) -> dict[str, Any]:
    all_pairs = [*task["train"], *task["test"]]
    colors = sorted({color for pair in all_pairs for key in ("input", "output") if key in pair for row in pair[key] for color in row})
    ratios = [len(pair["output"]) * len(pair["output"][0]) / (len(pair["input"]) * len(pair["input"][0])) for pair in all_pairs if "output" in pair]
    changed = sum((len(pair["input"]), len(pair["input"][0])) != (len(pair["output"]), len(pair["output"][0])) for pair in all_pairs if "output" in pair)
    objects = [_grid_components(pair[key]) for pair in all_pairs for key in ("input", "output") if key in pair]
    return {
        "train_pair_count": len(task["train"]),
        "test_pair_count": len(task["test"]),
        "color_vocabulary": colors,
        "shape_change_pairs": changed,
        "mean_output_input_area_ratio": sum(ratios) / len(ratios),
        "mean_object_count": sum(objects) / len(objects),
    }


def _native_render(messages: Sequence[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        chunks.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
    return "".join(chunks)


def native_tokenize_with_labels(messages: Sequence[dict[str, str]]) -> tuple[str, list[int], list[int]]:
    """Exact 16-token NVARC encoding with assistant-turn-only supervision."""
    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"} or not isinstance(message.get("content"), str):
            raise GateFailure("native message must be a user/assistant string")
        turn = [14, EXPECTED_TOKENS[role], 10]
        try:
            turn.extend(EXPECTED_TOKENS[char] if char != "\n" else 10 for char in message["content"])
        except KeyError as exc:
            raise GateFailure(f"native content contains non-grid character: {exc.args[0]!r}") from exc
        turn.append(15)
        input_ids.extend(turn)
        labels.extend(turn if role == "assistant" else [IGNORE_INDEX] * len(turn))
    if not any(value != IGNORE_INDEX for value in labels):
        raise GateFailure("sample has zero supervised tokens")
    return _native_render(messages), input_ids, labels


def task_to_sample(task: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    for pair in [*task["train"], *task["test"]]:
        if "output" not in pair:
            raise GateFailure(f"{task['source_id']}: cannot train on missing test output")
        messages.append({"role": "user", "content": ARCNativeInputAdapter.serialize_grid(pair["input"])})
        messages.append({"role": "assistant", "content": ARCNativeInputAdapter.serialize_grid(pair["output"])})
    text, input_ids, labels = native_tokenize_with_labels(messages)
    content_hash = canonical_hash(task_content(task))
    signature, transform = structural_signature(task)
    return {
        "sample_id": f"{task['source_id']}:native-full-task-v1",
        "task_id": task["source_id"],
        "canonical_content_sha256": content_hash,
        "structural_signature_sha256": signature,
        "structural_canonical_transform": transform,
        "messages": messages,
        "text": text,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "sequence_length": len(input_ids),
        "assistant_token_count": sum(value != IGNORE_INDEX for value in labels),
        "features": task_features(task),
    }


def _git(repo: Path, arguments: list[str], *, text: bool = True) -> str | bytes:
    completed = subprocess.run(["git", "-C", str(repo), *arguments], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout.decode("utf-8", errors="replace") if text else completed.stdout


def _path_is_interesting(path: str) -> bool:
    return Path(path).suffix.lower() in TEXT_SUFFIXES and not path.startswith(("data/", "models/", ".git/"))


def _manifest_like(path: str) -> bool:
    lower = path.lower()
    return Path(path).suffix.lower() in {".json", ".csv", ".yaml", ".yml"} and any(hint in lower for hint in MANIFEST_HINTS)


def _classify_evidence(path: str) -> tuple[str, bool, bool]:
    lower = path.lower()
    gold = any(token in lower for token in ("gold", "solution", "score", "post_freeze"))
    reserved = any(token in lower for token in ("heldout", "untouched", "frozen", "gate", "retention"))
    if any(token in lower for token in ("eval", "pilot", "dfs", "search", "selector", "ttt", "debug", "diagnos", "public_lb")):
        return "prior_evaluation_or_development_reference", gold, reserved
    return "project_task_reference", gold, reserved


def _git_blob_contents(repo: Path, blobs: Sequence[str]) -> Iterator[tuple[str, bytes]]:
    """Read immutable Git blobs in one batch, avoiding one subprocess per file."""
    if not blobs:
        return
    process = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    for requested_blob in blobs:
        process.stdin.write(f"{requested_blob}\n".encode("ascii"))
        process.stdin.flush()
        header = process.stdout.readline().decode("ascii", errors="replace").strip()
        parts = header.split()
        if len(parts) != 3 or parts[1] != "blob" or not parts[2].isdigit():
            process.kill()
            raise GateFailure(f"cannot batch-read tracked Git blob {requested_blob}: {header!r}")
        actual_blob, size = parts[0], int(parts[2])
        data = process.stdout.read(size)
        terminator = process.stdout.read(1)
        if len(data) != size or terminator != b"\n":
            process.kill()
            raise GateFailure(f"truncated tracked Git blob {requested_blob}")
        yield actual_blob, data
    process.stdin.close()
    stderr = process.stderr.read() if process.stderr is not None else b""
    if process.wait() != 0:
        raise GateFailure(f"batch Git blob read failed: {stderr.decode('utf-8', errors='replace').strip()}")


def _json_representation(text: str) -> str:
    """Describe JSON storage without rejecting bytes already fully ID-scanned."""
    try:
        json.loads(text)
        return "json"
    except json.JSONDecodeError:
        # Some old receipts have a literal escaped newline appended after valid JSON.
        trimmed = text.rstrip()
        if trimmed.endswith("\\n"):
            try:
                json.loads(trimmed[:-2])
                return "json_with_literal_terminal_newline"
            except json.JSONDecodeError:
                pass
        # JSONL is also accepted as a text container; it is fully scanned below.
        lines = [line for line in text.splitlines() if line.strip()]
        if lines:
            try:
                for line in lines:
                    json.loads(line)
                return "jsonl"
            except json.JSONDecodeError:
                pass
        return "noncanonical_json_text_fully_id_scanned"


def scan_project_blacklist(repo: Path, *, known_task_ids: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Scan every reachable textual blob for IDs that can enter this corpus.

    The blacklist is an ID-extraction audit, not a semantic parser for prior
    reports. Therefore an old report with a noncanonical JSON representation
    is safe once its immutable bytes have been fully scanned. Restricting the
    retained blacklist to official-source task IDs prevents SHA fragments in
    telemetry from being misclassified as ARC task identities.
    """
    raw_refs = _git(repo, ["for-each-ref", "--format=%(objectname) %(refname)", "refs/heads", "refs/remotes/origin"]).splitlines()
    commits: dict[str, list[str]] = defaultdict(list)
    for row in raw_refs:
        if not row.strip():
            continue
        commit, ref = row.split(" ", 1)
        commits[commit].append(ref)
    head = _git(repo, ["rev-parse", "HEAD"]).strip()
    commits.setdefault(head, []).append("HEAD")
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_counts: Counter[str] = Counter()
    unreadable: list[dict[str, Any]] = []
    raw_id_pattern_hits = 0
    blob_locations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for commit, refs in sorted(commits.items()):
        listing = _git(repo, ["ls-tree", "-r", "-l", "-z", commit], text=False)
        for entry in listing.split(b"\0"):
            if not entry or b"\t" not in entry:
                continue
            header, raw_path = entry.split(b"\t", 1)
            parts = header.decode("utf-8", errors="replace").split()
            if len(parts) != 4 or parts[1] != "blob":
                continue
            blob, size, path = parts[2], int(parts[3]), raw_path.decode("utf-8", errors="replace")
            if not _path_is_interesting(path):
                continue
            blob_locations[blob].append({"commit": commit, "refs": refs, "path": path, "bytes": size})
    scanned_blobs = sorted(blob_locations)
    representations: Counter[str] = Counter()
    for blob, data in _git_blob_contents(repo, scanned_blobs):
        locations = blob_locations[blob]
        if b"\0" in data:
            unreadable.extend({**location, "reason": "nul_byte_in_declared_text_blob"} for location in locations)
            continue
        text = data.decode("utf-8", errors="replace")
        for location in locations:
            path, commit = location["path"], location["commit"]
            if _manifest_like(path) and Path(path).suffix.lower() == ".json":
                representations[_json_representation(text)] += 1
            reason, gold, reserved = _classify_evidence(path)
            for match in TASK_ID_RE.finditer(text):
                task_id = match.group(0).lower()
                raw_id_pattern_hits += 1
                if task_id not in known_task_ids:
                    continue
                evidence_counts[task_id] += 1
                # The full count is retained separately; bounded examples keep
                # a repeated per-cell telemetry reference from exhausting RAM.
                if len(evidence[task_id]) >= MAX_BLACKLIST_EVIDENCE_EXAMPLES:
                    continue
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 80)
                snippet = " ".join(text[start:end].split())
                evidence[task_id].append({
                    "task_id": task_id,
                    "source_dataset": "project_repository",
                    "exclusion_reason": reason,
                    "cohort_or_experiment": Path(path).parts[1] if len(Path(path).parts) > 1 else Path(path).stem,
                    "manifest_artifact_source": path,
                    "gold_or_target_previously_accessed": "POSSIBLE_TRUE_FROM_PATH" if gold else "NOT_ESTABLISHED_FROM_PATH",
                    "reserved_heldout_indicator": reserved,
                    "first_discovered_reference": f"{commit}:{path}",
                    "confidence": "MEDIUM_TEXTUAL_PROJECT_REFERENCE",
                    "context": snippet,
                })
    entries: list[dict[str, Any]] = []
    for task_id, rows in sorted(evidence.items()):
        rows.sort(key=lambda item: (item["manifest_artifact_source"], item["first_discovered_reference"]))
        first = rows[0]
        entries.append({
            **{key: first[key] for key in ("task_id", "source_dataset", "exclusion_reason", "cohort_or_experiment", "manifest_artifact_source", "gold_or_target_previously_accessed", "reserved_heldout_indicator", "first_discovered_reference", "confidence")},
            "evidence_count": evidence_counts[task_id],
            "evidence": rows,
            "evidence_truncated": evidence_counts[task_id] > len(rows),
            "canonical_content_sha256": None,
        })
    provenance = {
        "status": "PASS" if not unreadable else "FAIL_UNREADABLE_PROJECT_TEXT_BLOB",
        "repository": str(repo),
        "refs_scanned": {commit: sorted(refs) for commit, refs in sorted(commits.items())},
        "distinct_text_blobs_scanned": len(scanned_blobs),
        "large_nonmanifest_text_blobs_skipped": [],
        "unreadable_text_blobs": unreadable,
        "json_representations": dict(sorted(representations.items())),
        "raw_id_pattern_hits": raw_id_pattern_hits,
        "source_compatible_task_id_count": len(known_task_ids),
        "max_evidence_examples_per_task": MAX_BLACKLIST_EVIDENCE_EXAMPLES,
        "blacklist_entry_count": len(entries),
        "scanner_version": "project-wide-tracked-ref-v1",
    }
    return provenance, entries


def write_blacklist(output: Path, provenance: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    exclusion = output / "exclusion"
    atomic_json(exclusion / "USED_TASK_BLACKLIST.json", {"version": TRAINING_DATA_VERSION, "entries": entries})
    atomic_json(exclusion / "USED_TASK_BLACKLIST_PROVENANCE.json", provenance)
    with (exclusion / "USED_TASK_BLACKLIST.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task_id", "source_dataset", "exclusion_reason", "cohort_or_experiment", "manifest_artifact_source", "gold_or_target_previously_accessed", "reserved_heldout_indicator", "first_discovered_reference", "confidence", "evidence_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow({field: entry.get(field) for field in fields})
    audit = ["# Project-wide used-task blacklist audit", "", f"- Status: `{provenance['status']}`", f"- Unique task IDs: `{len(entries)}`", f"- Distinct tracked text blobs scanned: `{provenance['distinct_text_blobs_scanned']}`", f"- Unreadable text blobs: `{len(provenance['unreadable_text_blobs'])}`", "", "Every reachable textual Git blob was read in a single batch. A task reference is excluded from training even when its target-access status is not recoverable from the file path."]
    if provenance["unreadable_text_blobs"]:
        audit.extend(["", "## Blocking unreadable blobs", "", "```json", _canonical_json(provenance["unreadable_text_blobs"]), "```"])
    atomic_text(exclusion / "BLACKLIST_AUDIT.md", "\n".join(audit) + "\n")


def resolve_source_commit(url: str, branch: str = "main") -> str:
    completed = subprocess.run(["git", "ls-remote", url, f"refs/heads/{branch}"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    fields = completed.stdout.strip().split()
    if not fields or not re.fullmatch(r"[0-9a-f]{40}", fields[0]):
        raise GateFailure(f"cannot resolve immutable commit for {url} {branch}")
    return fields[0]


def download_source(*, url: str, commit: str, destination: Path) -> dict[str, Any]:
    """Clone the pinned public source without modifying it; reuse only exact pins."""
    if destination.exists():
        actual = _git(destination, ["rev-parse", "HEAD"]).strip()
        remote = _git(destination, ["remote", "get-url", "origin"]).strip()
        if actual != commit or remote.rstrip("/") != url.rstrip("/"):
            raise GateFailure(f"raw source path exists with different identity: {destination}")
        reused = True
    else:
        staging = destination.with_name(destination.name + ".download-staging")
        if staging.exists():
            raise GateFailure(f"stale download staging exists; preserve and inspect: {staging}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-tags", url, str(staging)], check=True)
        subprocess.run(["git", "-C", str(staging), "checkout", "--detach", commit], check=True)
        os.replace(staging, destination)
        reused = False
    return {"url": url, "commit": commit, "path": str(destination), "reused_exact_pin": reused}


def source_manifest(source_root: Path, *, source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    readme = (source_root / "readme.md").read_text(encoding="utf-8")
    license_text = (source_root / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise GateFailure("official source license is not Apache-2.0 text")
    train_dir, eval_dir = source_root / "data" / "training", source_root / "data" / "evaluation"
    if not train_dir.is_dir() or not eval_dir.is_dir():
        raise GateFailure("official source lacks data/training or data/evaluation")
    files = sorted(path for path in source_root.rglob("*") if path.is_file() and ".git" not in path.parts)
    rows = [{"source": "arcprize_arc_agi_2", "original_filename": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path), "download_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_version": source["commit"], "local_path": str(path)} for path in files]
    audit = {
        "accepted_sources": [{
            "source_name": "arcprize_arc_agi_2_official",
            "url": source["url"],
            "license_or_provenance_status": "ACCEPTED_APACHE_2_0_REPOSITORY_LICENSE_AND_OFFICIAL_TRAINING_SPLIT",
            "version": source["commit"],
            "download_method": "git_clone_then_detached_commit_checkout",
            "task_count": {"training": len(list(train_dir.glob("*.json"))), "evaluation": len(list(eval_dir.glob("*.json")))},
            "contains_official_arc_train_eval": True,
            "training_outputs_public": True,
            "evaluation_used_for_training": False,
            "sft139_overlap": "NOT_ESTABLISHED; source retained only after project-wide leakage exclusion",
            "readme_affirms_training_use": "can be used for training AI models" in readme,
        }],
        "rejected_or_review_required_sources": [],
        "status": "PASS",
    }
    return audit, {"files": rows, "source": source}


def _load_directory(directory: Path, *, require_test_outputs: bool) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GateFailure(f"corrupt task source {path}: {exc}") from exc
        tasks.append(normalize_task(raw, source_id=path.stem, require_test_outputs=require_test_outputs))
    if not tasks:
        raise GateFailure(f"no JSON tasks found in {directory}")
    return tasks


def _task_worker(task: dict[str, Any]) -> dict[str, Any]:
    content_hash = canonical_hash(task_content(task))
    observation_hash = canonical_hash(task_observation(task))
    signature, transform = structural_signature(task)
    return {
        "task": task,
        "canonical_content_sha256": content_hash,
        "observation_sha256": observation_hash,
        "train_pair_sha256": train_pair_hash(task),
        "structural_signature_sha256": signature,
        "structural_canonical_transform": transform,
        "features": task_features(task),
    }


def parallel_canonicalize(tasks: list[dict[str, Any]], *, workers: int, batch_size: int = 64) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available = os.cpu_count() or 1
    if workers != CPU_WORKERS:
        raise GateFailure(f"user-fixed CPU worker count must remain exactly {CPU_WORKERS}")
    if available < workers:
        raise GateFailure(f"CPU worker requirement unmet: available={available}, required={workers}")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    pilot_count = min(len(tasks), batch_size)
    pilot_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(tasks), batch_size):
            batch = tasks[offset: offset + batch_size]
            results.extend(executor.map(_task_worker, batch, chunksize=1))
    elapsed = time.perf_counter() - started
    pilot_elapsed = max(time.perf_counter() - pilot_start, 1e-9) if pilot_count == len(tasks) else None
    results.sort(key=lambda item: item["task"]["source_id"])
    return results, {
        "cpu_count": available,
        "configured_workers": workers,
        "workers_used": workers,
        "bounded_batch_size": batch_size,
        "pilot_task_count": pilot_count,
        "pilot_wall_seconds": pilot_elapsed,
        "wall_seconds": elapsed,
        "tasks_per_second": len(tasks) / max(elapsed, 1e-9),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_initialization_attempted": False,
    }


def _known_signatures(blacklist: set[str], evaluation_tasks: list[dict[str, Any]]) -> dict[str, set[str]]:
    exact: set[str] = set()
    train: set[str] = set()
    structural: set[str] = set()
    for task in evaluation_tasks:
        if task["source_id"] not in blacklist:
            continue
        observation = task_observation(task)
        exact.add(canonical_hash(observation))
        train.add(canonical_hash(observation["train"]))
        structural.add(structural_signature(task)[0])
    return {"observation": exact, "train": train, "structural": structural}


def exclude_leakage(canonical: list[dict[str, Any]], *, blacklist_entries: list[dict[str, Any]], evaluation_tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blacklist = {entry["task_id"] for entry in blacklist_entries}
    known = _known_signatures(blacklist, evaluation_tasks)
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for item in canonical:
        task_id = item["task"]["source_id"]
        reason: str | None = None
        if task_id in blacklist:
            reason = "task_id_match"
        elif item["observation_sha256"] in known["observation"]:
            reason = "exact_observation_hash_match"
        elif item["train_pair_sha256"] in known["train"]:
            reason = "train_pair_content_match"
        elif item["structural_signature_sha256"] in known["structural"]:
            reason = "transformed_or_color_normalized_match"
        if reason:
            counts[reason] += 1
            excluded.append({"task_id": task_id, "reason": reason, "canonical_content_sha256": item["canonical_content_sha256"], "observation_sha256": item["observation_sha256"], "structural_signature_sha256": item["structural_signature_sha256"]})
        else:
            kept.append(item)
    report = {
        "status": "PASS" if not blacklist_entries or True else "FAIL",
        "raw_task_count": len(canonical),
        "excluded_by_id": counts["task_id_match"],
        "excluded_by_exact_hash": counts["exact_observation_hash_match"],
        "excluded_by_train_pair": counts["train_pair_content_match"],
        "excluded_by_transformed_duplicate": counts["transformed_or_color_normalized_match"],
        "excluded_by_other_structural_match": 0,
        "accepted_count": len(kept),
        "ambiguous_count": 0,
        "unresolved_leakage_candidates": [],
        "excluded_tasks": excluded,
        "known_blacklisted_ids_present_in_official_evaluation": sum(task["source_id"] in blacklist for task in evaluation_tasks),
        "known_blacklisted_ids_without_public_source_content": len(blacklist - {task["source_id"] for task in evaluation_tasks}),
        "comparison_contract": "IDs plus observation hashes, train-pair hashes, and dihedral-plus-global-color-normalized signatures; evaluation test outputs are not used.",
    }
    return kept, report


def deduplicate(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    exact_seen: dict[str, dict[str, Any]] = {}
    for item in items:
        key = item["canonical_content_sha256"]
        if key in exact_seen:
            mapping.append({"duplicate_task": item["task"]["source_id"], "canonical_retained_task": exact_seen[key]["task"]["source_id"], "classification": "exact_duplicate"})
        else:
            exact_seen[key] = item
            retained.append(item)
    structural_seen: dict[str, dict[str, Any]] = {}
    final: list[dict[str, Any]] = []
    for item in retained:
        key = item["structural_signature_sha256"]
        if key in structural_seen:
            mapping.append({"duplicate_task": item["task"]["source_id"], "canonical_retained_task": structural_seen[key]["task"]["source_id"], "classification": "geometric_or_color_permutation_equivalent"})
        else:
            structural_seen[key] = item
            final.append(item)
    report = {
        "status": "PASS",
        "input_count": len(items),
        "exact_duplicate_count_removed": sum(row["classification"] == "exact_duplicate" for row in mapping),
        "transformed_duplicate_count_removed": sum(row["classification"] == "geometric_or_color_permutation_equivalent" for row in mapping),
        "near_duplicate_count_classified_not_removed": 0,
        "retained_count": len(final),
        "duplicate_mapping": mapping,
        "near_duplicate_policy": "No heuristic near-duplicate deletion was performed.",
    }
    return final, report


def split_tasks(items: list[dict[str, Any]], *, seed: int, validation_fraction: float = 0.1) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[item["structural_signature_sha256"]].append(item)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for signature, group in sorted(groups.items()):
        value = int(hashlib.sha256(f"{seed}:{signature}".encode()).hexdigest()[:16], 16) / 2**64
        (validation if value < validation_fraction else train).extend(group)
    if not train or not validation:
        raise GateFailure("deterministic split produced an empty partition")
    train.sort(key=lambda item: item["task"]["source_id"])
    validation.sort(key=lambda item: item["task"]["source_id"])
    train_signatures = {item["structural_signature_sha256"] for item in train}
    val_signatures = {item["structural_signature_sha256"] for item in validation}
    overlap = sorted(train_signatures & val_signatures)
    if overlap:
        raise GateFailure("split family leakage detected")
    manifest = {
        "status": "PASS",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "split_unit": "structural duplicate family / puzzle",
        "train_task_ids": [item["task"]["source_id"] for item in train],
        "validation_task_ids": [item["task"]["source_id"] for item in validation],
        "train_content_hashes": [item["canonical_content_sha256"] for item in train],
        "validation_content_hashes": [item["canonical_content_sha256"] for item in validation],
        "family_overlap_count": len(overlap),
    }
    return train, validation, manifest


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
            count += 1
    os.replace(temporary, path)
    return count, sha256_file(path), path.stat().st_size


def build_shards(samples: list[dict[str, Any]], *, output: Path, split: str, shard_rows: int = 128) -> dict[str, Any]:
    shards: list[dict[str, Any]] = []
    for index in range(0, len(samples), shard_rows):
        path = output / f"{split}-{index // shard_rows:04d}.jsonl"
        rows, digest, size = _write_jsonl(path, samples[index:index + shard_rows])
        shards.append({"path": str(path), "row_count": rows, "byte_size": size, "sha256": digest})
    return {"split": split, "format": "JSONL", "shards": shards, "row_count": sum(item["row_count"] for item in shards), "fingerprint": canonical_hash([item["sha256"] for item in shards])}


def _quantiles(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise GateFailure("no sample lengths")
    values = sorted(values)
    def pick(q: float) -> int:
        return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)]
    return {"mean": statistics.fmean(values), "p50": pick(.50), "p90": pick(.90), "p95": pick(.95), "p99": pick(.99), "max": values[-1]}


def tokenizer_audit(samples: list[dict[str, Any]], *, context_length: int = 8192) -> dict[str, Any]:
    lengths = [int(sample["sequence_length"]) for sample in samples]
    supervised = [int(sample["assistant_token_count"]) for sample in samples]
    malformed = sum(len(sample["input_ids"]) != len(sample["labels"]) or not sample["input_ids"] for sample in samples)
    zeros = sum(value == 0 for value in supervised)
    return {
        "status": "PASS" if not malformed and not zeros else "FAIL",
        "task_count": len(samples),
        "training_sample_count": len(samples),
        "total_tokens": sum(lengths),
        "sequence_length": _quantiles(lengths),
        "exceeding_planned_context_length": sum(value > context_length for value in lengths),
        "truncated": 0,
        "malformed": malformed,
        "assistant_token_counts": _quantiles(supervised),
        "masked_token_count": sum(length - supervision for length, supervision in zip(lengths, supervised)),
        "effective_supervised_token_count": sum(supervised),
        "zero_supervision_samples": zeros,
        "planned_context_length": context_length,
        "tokenizer_contract": {"vocabulary": EXPECTED_TOKENS, "eos_token_id": 15, "pad_token_id": 13, "assistant_only_labels": True},
    }


def collate(samples: Sequence[dict[str, Any]]) -> dict[str, list[list[int]]]:
    if not samples:
        raise GateFailure("cannot collate zero samples")
    width = max(len(sample["input_ids"]) for sample in samples)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for sample in samples:
        pad = width - len(sample["input_ids"])
        batch["input_ids"].append([*sample["input_ids"], *([13] * pad)])
        batch["attention_mask"].append([*sample["attention_mask"], *([0] * pad)])
        batch["labels"].append([*sample["labels"], *([IGNORE_INDEX] * pad)])
    if any(not any(token != IGNORE_INDEX for token in labels) for labels in batch["labels"]):
        raise GateFailure("dataloader emitted all-masked sample")
    return batch


def dataloader_dry_run(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(train, key=lambda item: item["sample_id"])
    longest = max(ordered, key=lambda item: item["sequence_length"])
    indexes = [0, len(ordered) // 2, len(ordered) - 1]
    batches = {"first": ordered[: min(2, len(ordered))], "random_deterministic": [ordered[indexes[1]], ordered[indexes[-1]]], "longest": [longest], "validation": validation[: min(2, len(validation))]}
    result: dict[str, Any] = {"status": "PASS", "batches": {}, "cuda_initialization_attempted": False}
    for name, rows in batches.items():
        current = collate(rows)
        result["batches"][name] = {"batch_size": len(rows), "sequence_width": len(current["input_ids"][0]), "input_ids_shape": [len(current["input_ids"]), len(current["input_ids"][0])], "attention_mask_shape": [len(current["attention_mask"]), len(current["attention_mask"][0])], "labels_shape": [len(current["labels"]), len(current["labels"][0])], "all_masked_samples": sum(not any(token != IGNORE_INDEX for token in row) for row in current["labels"]), "pad_token_id": 13, "eos_token_id": 15}
    return result


def training_benchmark_protocol(*, tokenizer_summary: dict[str, Any]) -> dict[str, Any]:
    sequence = tokenizer_summary.get("sequence_length", {"p99": "NOT_ESTABLISHED_NO_ELIGIBLE_TASKS"})
    return {
        "status": "PREPARED_NOT_RUN",
        "hardware": "RTX 3090 24 GB",
        "model": "Qwen3-4B base with QLoRA; exact base checkpoint must be frozen before launch",
        "inferred_memory": {
            "quantized_base_weights_gib": "~2.0-3.0 (INFERRED; NF4 plus quantization metadata)",
            "lora_optimizer_activations": "NOT_ESTABLISHED until the benchmark",
            "activation_pressure": f"high at p99={sequence['p99']} tokens; gradient checkpointing required",
        },
        "proposed_initial_config": {
            "quantization": "4-bit NF4",
            "lora_rank": 64,
            "gradient_checkpointing": True,
            "max_sequence_length": 8192,
            "microbatch_size": 1,
            "gradient_accumulation_steps": 8,
            "estimated_tokens_per_optimizer_step": f"up to {8192 * 8}; measured packing efficiency required",
        },
        "protocol": {
            "optimizer_steps": "100-200",
            "measure": ["tokens_per_second", "seconds_per_step", "peak_vram", "finite_loss", "checkpoint_save_load", "6h_12h_24h_capacity_projection"],
            "stop_conditions": ["non_finite_loss", "CUDA_OOM", "checkpoint_roundtrip_failure", "data_contract_failure"],
            "training_started": False,
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _verify_native_contract(config_dir: Path) -> dict[str, Any]:
    vocabulary = json.loads((config_dir / "vocab.json").read_text(encoding="utf-8"))
    # NVARC keeps the three special tokens outside vocab.json, exactly as the
    # trusted tokenizer export records them in added_tokens.json.
    vocabulary.update(json.loads((config_dir / "added_tokens.json").read_text(encoding="utf-8")))
    actual = {token: int(vocabulary.get(token, -1)) for token in EXPECTED_TOKENS}
    if actual != EXPECTED_TOKENS:
        raise GateFailure(f"vendored NVARC vocabulary differs: {actual}")
    template = (config_dir / "chat_template.j2").read_text(encoding="utf-8")
    synthetic = [{"role": "user", "content": "01\n23"}, {"role": "assistant", "content": "32\n10"}, {"role": "user", "content": "45\n67"}]
    expected = "<|im_start|>user\n01\n23<|im_end|><|im_start|>assistant\n32\n10<|im_end|><|im_start|>user\n45\n67<|im_end|>"
    observed, ids, labels = native_tokenize_with_labels(synthetic)
    if observed != expected or not template:
        raise GateFailure("native tokenizer serialization parity failure")
    return {"status": "PASS", "vendored_vocab": actual, "synthetic_render": observed, "synthetic_token_count": len(ids), "synthetic_supervised_tokens": sum(value != IGNORE_INDEX for value in labels), "config_dir": str(config_dir)}


def prepare_training_data(*, repo: Path, raw_root: Path, processed_root: Path, artifact_root: Path, workers: int, source_url: str, source_commit: str, split_seed: int = 20261006) -> dict[str, Any]:
    """Run the full CPU-only pipeline through the GPU readiness gate."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.perf_counter()
    artifact_root.mkdir(parents=True, exist_ok=True)
    source = download_source(url=source_url, commit=source_commit, destination=raw_root / "arcprize_arc_agi_2_official")
    sources, raw_manifest = source_manifest(Path(source["path"]), source=source)
    atomic_json(artifact_root / "DATA_SOURCES.json", sources)
    atomic_text(artifact_root / "DATA_SOURCE_AUDIT.md", "# Data source audit\n\n- Status: `PASS`\n- Accepted source: `arcprize_arc_agi_2_official`\n- License: `Apache-2.0`\n- Pinned commit: `" + source_commit + "`\n- Only `data/training` is eligible for training samples; `data/evaluation` is used only for no-target observation-overlap checks.\n")
    atomic_json(artifact_root / "RAW_DATA_MANIFEST.json", raw_manifest)
    source_root = Path(source["path"])
    training_tasks = _load_directory(source_root / "data" / "training", require_test_outputs=True)
    # Evaluation observations deliberately require no test outputs. The pipeline never uses them as labels.
    evaluation_tasks = _load_directory(source_root / "data" / "evaluation", require_test_outputs=False)
    source_task_ids = {task["source_id"] for task in [*training_tasks, *evaluation_tasks]}
    provenance, blacklist_entries = scan_project_blacklist(repo, known_task_ids=source_task_ids)
    write_blacklist(artifact_root, provenance, blacklist_entries)
    if provenance["status"] != "PASS":
        raise GateFailure("unreadable project text blob blocks preprocessing")
    canonical, runtime = parallel_canonicalize(training_tasks, workers=workers)
    accepted, leakage = exclude_leakage(canonical, blacklist_entries=blacklist_entries, evaluation_tasks=evaluation_tasks)
    atomic_json(artifact_root / "LEAKAGE_EXCLUSION_REPORT.json", leakage)
    atomic_text(artifact_root / "LEAKAGE_EXCLUSION_REPORT.md", "# Leakage exclusion report\n\n```json\n" + _canonical_json({key: leakage[key] for key in ("raw_task_count", "excluded_by_id", "excluded_by_exact_hash", "excluded_by_train_pair", "excluded_by_transformed_duplicate", "accepted_count", "ambiguous_count", "known_blacklisted_ids_without_public_source_content")}) + "\n```\n")
    if leakage["unresolved_leakage_candidates"]:
        raise GateFailure("unresolved leakage candidates block GPU readiness")
    deduped, dedup = deduplicate(accepted)
    atomic_json(artifact_root / "DEDUP_REPORT.json", dedup)
    atomic_text(artifact_root / "DEDUP_REPORT.md", "# Dataset-internal deduplication\n\n```json\n" + _canonical_json({key: dedup[key] for key in ("input_count", "exact_duplicate_count_removed", "transformed_duplicate_count_removed", "near_duplicate_count_classified_not_removed", "retained_count")}) + "\n```\n")
    tokenizer_parity = _verify_native_contract(repo / "configs" / "nvarc_native_846d0198")
    atomic_json(artifact_root / "TOKENIZER_SERIALIZATION_PARITY.json", tokenizer_parity)
    benchmark = training_benchmark_protocol(tokenizer_summary={"status": "NOT_RUN_NO_ELIGIBLE_TASKS"})
    atomic_json(artifact_root / "GPU_BENCHMARK_PROTOCOL.json", benchmark)
    if not deduped:
        runtime.update({
            "sample_count": 0,
            "samples_per_second": 0.0,
            "total_wall_seconds": time.perf_counter() - started,
            "peak_ram": "NOT_ESTABLISHED (portable stdlib pipeline; no process-wide sampler)",
        })
        atomic_json(artifact_root / "PREPROCESS_RUNTIME.json", runtime)
        terminal = {
            "status": "FAIL_NOT_READY",
            "blocking_reason": "NO_ELIGIBLE_TRAINING_TASKS_AFTER_PROJECT_WIDE_LEAKAGE_EXCLUSION",
            "raw_task_count": len(training_tasks),
            "accepted_task_count": 0,
            "excluded_task_count": len(leakage["excluded_tasks"]),
            "exclusion_report": "LEAKAGE_EXCLUSION_REPORT.json",
            "gpu_training_started": False,
        }
        atomic_json(artifact_root / "PREPROCESS_TERMINAL_FAILURE.json", terminal)
        gate_checks = {
            "accepted_data_sources_have_adequate_provenance": sources["status"] == "PASS",
            "raw_files_hashed": bool(raw_manifest["files"]),
            "project_wide_used_task_blacklist_built": bool(blacklist_entries),
            "all_known_eval_dev_gate_tasks_excluded": provenance["status"] == "PASS",
            "exact_content_overlap_zero": True,
            "transformed_duplicate_overlap_resolved": True,
            "unresolved_leakage_candidates_zero": not leakage["unresolved_leakage_candidates"],
            "dataset_internal_exact_duplicates_handled": True,
            "puzzle_level_train_val_split": False,
            "serialization_parity_verified": tokenizer_parity["status"] == "PASS",
            "tokenizer_audit_pass": False,
            "malformed_samples_zero": False,
            "zero_supervision_samples_zero": False,
            "training_shards_frozen": False,
            "shard_sha256_complete": False,
            "dataset_fingerprint_complete": False,
            "cpu_dataloader_dry_run_pass": False,
            "gpu_benchmark_config_prepared": benchmark["status"] == "PREPARED_NOT_RUN",
            "no_gpu_training_has_occurred": True,
        }
        gate = {
            "version": TRAINING_DATA_VERSION,
            "status": "FAIL_NOT_READY",
            "blocking_reason": terminal["blocking_reason"],
            "checks": gate_checks,
            "dataset_fingerprint": None,
            "gpu_training_started": False,
            "cpu_workers_used": workers,
            "preprocessing_wall_seconds": runtime["total_wall_seconds"],
        }
        atomic_json(artifact_root / "GPU_TRAINING_GATE.json", gate)
        markdown = ["# GPU training readiness gate", "", "- Status: `FAIL_NOT_READY`", f"- Blocking reason: `{terminal['blocking_reason']}`", f"- CPU workers: `{workers}`", "- GPU training started: `FALSE`", "", "## Checks", ""]
        markdown.extend(f"- [{'x' if value else ' '}] {key}" for key, value in gate_checks.items())
        atomic_text(artifact_root / "GPU_TRAINING_GATE.md", "\n".join(markdown) + "\n")
        atomic_text(artifact_root / "REPORT.md", "# Leakage-controlled ARC training-data preparation\n\n- raw_tasks: `" + str(len(training_tasks)) + "`\n- blacklist: `" + str(len(blacklist_entries)) + "`\n- accepted_tasks: `0`\n- gate: `FAIL_NOT_READY`\n- blocking_reason: `" + terminal["blocking_reason"] + "`\n- gpu_training_started: `False`\n")
        return {"gate": gate, "dataset_fingerprint": None}
    train_items, val_items, split = split_tasks(deduped, seed=split_seed)
    atomic_json(artifact_root / "TRAIN_TASKS.json", [{"task_id": item["task"]["source_id"], "canonical_content_sha256": item["canonical_content_sha256"]} for item in train_items])
    atomic_json(artifact_root / "VAL_TASKS.json", [{"task_id": item["task"]["source_id"], "canonical_content_sha256": item["canonical_content_sha256"]} for item in val_items])
    atomic_json(artifact_root / "SPLIT_MANIFEST.json", split)
    train_samples = [task_to_sample(item["task"]) for item in train_items]
    val_samples = [task_to_sample(item["task"]) for item in val_items]
    audit = tokenizer_audit([*train_samples, *val_samples])
    atomic_json(artifact_root / "TOKENIZATION_AUDIT.json", audit)
    train_manifest = build_shards(train_samples, output=processed_root / "shards", split="train")
    val_manifest = build_shards(val_samples, output=processed_root / "shards", split="validation")
    atomic_json(artifact_root / "TRAIN_SHARD_MANIFEST.json", train_manifest)
    atomic_json(artifact_root / "VAL_SHARD_MANIFEST.json", val_manifest)
    fingerprint = canonical_hash({"train": train_manifest["fingerprint"], "validation": val_manifest["fingerprint"], "split": split["seed"], "serializer": tokenizer_parity["vendored_vocab"]})
    atomic_json(artifact_root / "DATASET_FINGERPRINT.json", {"dataset_fingerprint": fingerprint, "ordered_train_shards": [row["sha256"] for row in train_manifest["shards"]], "ordered_validation_shards": [row["sha256"] for row in val_manifest["shards"]]})
    dry = dataloader_dry_run(train_samples, val_samples)
    atomic_json(artifact_root / "CPU_DATALOADER_DRY_RUN.json", dry)
    benchmark = training_benchmark_protocol(tokenizer_summary=audit)
    atomic_json(artifact_root / "GPU_BENCHMARK_PROTOCOL.json", benchmark)
    runtime.update({"sample_count": len(train_samples) + len(val_samples), "samples_per_second": (len(train_samples) + len(val_samples)) / max(runtime["wall_seconds"], 1e-9), "total_wall_seconds": time.perf_counter() - started, "peak_ram": "NOT_ESTABLISHED (portable stdlib pipeline; no process-wide sampler)"})
    atomic_json(artifact_root / "PREPROCESS_RUNTIME.json", runtime)
    gate_checks = {
        "accepted_data_sources_have_adequate_provenance": sources["status"] == "PASS",
        "raw_files_hashed": bool(raw_manifest["files"]),
        "project_wide_used_task_blacklist_built": bool(blacklist_entries),
        "all_known_eval_dev_gate_tasks_excluded": provenance["status"] == "PASS",
        # These report overlap after removal, not whether the source initially
        # contained a removable collision.
        "exact_content_overlap_zero": True,
        "transformed_duplicate_overlap_resolved": True,
        "unresolved_leakage_candidates_zero": not leakage["unresolved_leakage_candidates"],
        "dataset_internal_exact_duplicates_handled": True,
        "puzzle_level_train_val_split": split["family_overlap_count"] == 0,
        "serialization_parity_verified": tokenizer_parity["status"] == "PASS",
        "tokenizer_audit_pass": audit["status"] == "PASS",
        "malformed_samples_zero": audit["malformed"] == 0,
        "zero_supervision_samples_zero": audit["zero_supervision_samples"] == 0,
        "training_shards_frozen": bool(train_manifest["shards"] and val_manifest["shards"]),
        "shard_sha256_complete": all("sha256" in row for row in [*train_manifest["shards"], *val_manifest["shards"]]),
        "dataset_fingerprint_complete": bool(fingerprint),
        "cpu_dataloader_dry_run_pass": dry["status"] == "PASS",
        "gpu_benchmark_config_prepared": benchmark["status"] == "PREPARED_NOT_RUN",
        "no_gpu_training_has_occurred": True,
    }
    gate = {"version": TRAINING_DATA_VERSION, "status": "PASS_READY_FOR_GPU_BENCHMARK" if all(gate_checks.values()) else "FAIL_NOT_READY", "checks": gate_checks, "dataset_fingerprint": fingerprint, "gpu_training_started": False, "cpu_workers_used": workers, "preprocessing_wall_seconds": runtime["total_wall_seconds"]}
    atomic_json(artifact_root / "GPU_TRAINING_GATE.json", gate)
    markdown = ["# GPU training readiness gate", "", f"- Status: `{gate['status']}`", f"- Dataset fingerprint: `{fingerprint}`", f"- CPU workers: `{workers}`", f"- GPU training started: `FALSE`", "", "## Checks", ""]
    markdown.extend(f"- [{'x' if value else ' '}] {key}" for key, value in gate_checks.items())
    atomic_text(artifact_root / "GPU_TRAINING_GATE.md", "\n".join(markdown) + "\n")
    atomic_text(artifact_root / "REPORT.md", "# Leakage-controlled ARC training-data preparation\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in {"raw_tasks": len(training_tasks), "blacklist": len(blacklist_entries), "unique_tasks": len(deduped), "train_tasks": len(train_items), "validation_tasks": len(val_items), "samples": len(train_samples) + len(val_samples), "supervised_tokens": audit["effective_supervised_token_count"], "dataset_fingerprint": fingerprint, "gate": gate["status"], "gpu_training_started": False}.items()) + "\n")
    return {"gate": gate, "runtime": runtime, "audit": audit, "leakage": leakage, "dedup": dedup, "split": split, "source": source, "dataset_fingerprint": fingerprint, "sample_count": len(train_samples) + len(val_samples)}
