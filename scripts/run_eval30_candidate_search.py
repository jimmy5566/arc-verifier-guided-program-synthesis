"""Target-blind candidate-only Beam/DFS search on a frozen Evaluation30 cohort.

This is deliberately separate from the production runner.  It reuses the
native tokenizer, prompt transport, Aug8 views, parser, inverse transforms,
and de-duplication, but never imports ARC solutions or runs a ranker.  The
paired scorer is the first component allowed to inspect targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import statistics
import sys
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json, inspect_hardware


FROZEN_STATUS = "EVAL30_SEARCH_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"
MODEL_LOAD_TIMEOUT_SECONDS = 300.0


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _task_hash(task_ids: list[str]) -> str:
    return _sha256_bytes(json.dumps(sorted(task_ids), separators=(",", ":")).encode())


def _identity(manifest: dict[str, Any], search_config: dict[str, Any], condition: str, config: Path) -> str:
    payload = {
        "manifest_hash": manifest["task_ids_hash"],
        "condition": condition,
        "condition_config": search_config["conditions"][condition],
        "frozen_source_commit": manifest["frozen_inference"]["source_commit"],
        "native_config_sha256": _sha256_bytes(config.read_bytes()),
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


@dataclass(frozen=True)
class _GridPrefix:
    """Tiny-vocabulary grammar state for a non-empty rectangular ARC grid."""

    completed_rows: int = 0
    width: int | None = None
    current_width: int = 0

    def allowed_tokens(self) -> tuple[int, ...]:
        if self.completed_rows >= 30:
            return (15,) if self.current_width == 0 else ()
        values: list[int] = []
        if self.current_width < 30:
            values.extend(range(10))
        if self.current_width:
            if self.width is None or self.current_width == self.width:
                values.extend((10, 15))
        elif self.completed_rows:
            values.append(15)
        return tuple(values)

    def consume(self, token_id: int) -> "_GridPrefix | None":
        if token_id not in self.allowed_tokens():
            return None
        if 0 <= token_id <= 9:
            return _GridPrefix(self.completed_rows, self.width, self.current_width + 1)
        if token_id == 10:
            return _GridPrefix(self.completed_rows + 1, self.width or self.current_width, 0)
        if token_id == 15 and self.terminal:
            return self
        return None

    @property
    def terminal(self) -> bool:
        return bool(self.current_width and (self.width is None or self.current_width == self.width)) or bool(
            self.completed_rows and self.width is not None and self.current_width == 0
        )


@dataclass(frozen=True)
class _DFSState:
    token_ids: tuple[int, ...]
    prefix: _GridPrefix
    depth: int
    cumulative_nll: float


@dataclass(frozen=True)
class _DecodedPath:
    token_ids: tuple[int, ...]
    cumulative_nll: float
    elapsed_seconds: float
    branches: tuple[_DFSState, ...]
    expanded_nodes: int
    timed_out: bool


def _rank_legal(logits: Any, prefix: _GridPrefix) -> list[tuple[float, int]]:
    """Return model-probability scores for grammar-legal tokens only."""
    import torch

    probabilities = torch.log_softmax(logits.float(), dim=-1)
    return sorted(
        ((-float(probabilities[token_id].item()), token_id) for token_id in prefix.allowed_tokens()),
        key=lambda item: (item[0], item[1]),
    )


def _decode_one_dfs_path(provider: Any, messages: list[dict[str, str]], state: _DFSState, spec: dict[str, Any]) -> _DecodedPath:
    """Greedily complete one prefix and expose bounded probability-only branches.

    A stack in :func:`_dfs_paths` makes the resulting traversal depth-first.
    Only one KV cache is live per completed path, so alternative branches never
    share mutable cache state.  This is slower than an unconstrained beam but
    is intentionally bounded and grammar-valid by construction.
    """
    import torch

    assert provider.model is not None and provider.tokenizer is not None
    encoded = provider.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    )
    prompt_ids = encoded["input_ids"][0]
    max_new_tokens = int(spec["max_new_tokens"])
    if int(prompt_ids.shape[-1]) + len(state.token_ids) > int(spec["context_window"]):
        raise ValueError("DFS prompt plus branch prefix exceeds frozen context window")
    started = time.perf_counter()
    token_ids, prefix, path_nll = list(state.token_ids), state.prefix, state.cumulative_nll
    branches: list[_DFSState] = []
    expanded = 0
    timed_out = False
    # Reconstruct each branch from a complete input prefix.  This avoids any
    # mutable KV-cache sharing across sibling DFS paths.
    ids = torch.tensor([list(prompt_ids.detach().cpu().tolist()) + token_ids], dtype=torch.long, device=provider.device)
    with torch.inference_mode():
        output = provider.model(input_ids=ids, use_cache=True, return_dict=True)
        logits, cache = output.logits[0, -1, :], output.past_key_values
        del output, ids
        branch_points = 0
        while len(token_ids) < max_new_tokens:
            if time.perf_counter() - started >= float(spec["max_wall_seconds_per_path"]):
                timed_out = True
                break
            ranked = _rank_legal(logits, prefix)
            if not ranked:
                break
            chosen_nll, chosen_token = ranked[0]
            # The first eligible alternates at deterministically encountered
            # positions become child states.  Their ordering is probability,
            # then token ID; no task/output information is consulted.
            if state.depth < int(spec["max_depth"]) and branch_points < int(spec["max_branch_points_per_path"]):
                for alternative_nll, alternative_token in ranked[1 : 1 + int(spec["alternatives_per_branch_point"])]:
                    alternative_prefix = prefix.consume(alternative_token)
                    if alternative_prefix is not None and alternative_token != 15:
                        branches.append(
                            _DFSState(
                                tuple(token_ids + [alternative_token]),
                                alternative_prefix,
                                state.depth + 1,
                                path_nll + alternative_nll,
                            )
                        )
                branch_points += 1
            next_prefix = prefix.consume(chosen_token)
            if next_prefix is None:
                break
            token_ids.append(chosen_token)
            path_nll += chosen_nll
            if chosen_token == 15:
                del logits, cache
                return _DecodedPath(tuple(token_ids), path_nll, time.perf_counter() - started, tuple(branches), expanded, timed_out)
            next_input = torch.tensor([[chosen_token]], dtype=torch.long, device=provider.device)
            output = provider.model(input_ids=next_input, past_key_values=cache, use_cache=True, return_dict=True)
            logits, cache = output.logits[0, -1, :], output.past_key_values
            del output, next_input
            prefix = next_prefix
            expanded += 1
    del logits, cache
    return _DecodedPath(tuple(token_ids), path_nll, time.perf_counter() - started, tuple(branches), expanded, timed_out)


def _dfs_paths(provider: Any, messages: list[dict[str, str]], spec: dict[str, Any]) -> tuple[list[_DecodedPath], dict[str, Any]]:
    """Fixed-budget grammar-constrained DFS, without solution-aware pruning."""
    provider.load()
    pending = [_DFSState((), _GridPrefix(), 0, 0.0)]
    complete: list[_DecodedPath] = []
    expanded = 0
    timed_out = 0
    while pending and len(complete) < int(spec["max_completed_paths_per_view"]):
        state = pending.pop()
        decoded = _decode_one_dfs_path(provider, messages, state, spec)
        expanded += decoded.expanded_nodes
        timed_out += int(decoded.timed_out)
        if decoded.token_ids and decoded.token_ids[-1] == 15:
            complete.append(decoded)
        # Push reverse-sorted children so LIFO pop visits the lower-NLL child
        # first.  Capping applies before new states are ever evaluated.
        remaining = int(spec["max_completed_paths_per_view"]) - len(complete)
        children = sorted(decoded.branches, key=lambda item: (item.cumulative_nll, item.token_ids))[: max(0, remaining)]
        pending.extend(reversed(children))
        if len(pending) > int(spec["max_pending_states_per_view"]):
            pending = sorted(pending, key=lambda item: (item.cumulative_nll, item.token_ids))[: int(spec["max_pending_states_per_view"])]
    return complete, {
        "completed_paths": len(complete), "expanded_nodes": expanded, "timed_out_paths": timed_out,
        "pending_states_discarded": len(pending),
    }


def _candidate(augmentation: Any, prediction: list[list[list[int]]], completion_tokens: int, elapsed_seconds: float) -> Any:
    from inference.nvarc_native_candidates import NativeGridCandidate

    return NativeGridCandidate(
        augmentation=augmentation,
        prediction=tuple(tuple(tuple(int(value) for value in row) for row in grid) for grid in prediction),
        completion_tokens=completion_tokens,
        generation_seconds=elapsed_seconds,
    )


def _generate_task(provider: Any, task: Any, *, condition: str, spec: dict[str, Any], settings: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    from inference.dynamic_task_scheduler import task_seed
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix, parse_native_grid
    from inference.nvarc_native_augmentation import bounded_native_augmentations, transform_tasks_for_augmentations

    augmentations = bounded_native_augmentations(
        color_offsets=tuple(settings["color_offsets"]), pair_orders=tuple(settings["train_pair_orders"])
    )[:8]
    transformed = transform_tasks_for_augmentations(task, augmentations)
    prefixes = [native_training_message_prefix(view) for view in transformed]
    raw: list[Any] = []
    invalid = 0
    generated = 0
    path_stats: list[dict[str, Any]] = []
    for augmentation_index, (augmentation, view) in enumerate(zip(augmentations, transformed, strict=True)):
        by_rank: list[list[tuple[list[list[int]] | None, int, float]]] = []
        per_test_stats: list[dict[str, Any]] = []
        for test_index, example in enumerate(view.test):
            messages = native_messages_from_training_prefix(prefixes[augmentation_index], example.input)
            if condition == "beam2":
                beams = provider.generate_beams(
                    messages,
                    max_new_tokens=int(spec["max_new_tokens"]),
                    context_window=int(spec["context_window"]),
                    beam_width=int(spec["beam_width"]),
                )
                entries = []
                for item in beams:
                    parsed = parse_native_grid(item.text)
                    entries.append((None if parsed is None else augmentation.inverse_grid(parsed), item.completion_tokens, item.elapsed_seconds))
                per_test_stats.append({"test_index": test_index, "paths": len(beams), "valid": sum(value[0] is not None for value in entries), "sequence_scores": [item.sequence_score for item in beams]})
            else:
                # Greedy decoding is deterministic, but retain the task-local
                # seed provenance used by the frozen native stack.
                import torch
                seed = task_seed(task.task_id, int(settings["decode"]["seed"]), f"augmentation:{augmentation_index}:test:{test_index}")
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                paths, stats = _dfs_paths(provider, messages, spec)
                entries = []
                for item in paths:
                    text = provider.tokenizer.decode(list(item.token_ids), skip_special_tokens=True)
                    parsed = parse_native_grid(text)
                    entries.append((None if parsed is None else augmentation.inverse_grid(parsed), len(item.token_ids), item.elapsed_seconds))
                per_test_stats.append({"test_index": test_index, **stats, "valid": sum(value[0] is not None for value in entries)})
            by_rank.append(entries)
        aligned = min((len(values) for values in by_rank), default=0)
        generated += aligned
        for rank in range(aligned):
            selected = [items[rank] for items in by_rank]
            if any(grid is None for grid, _tokens, _elapsed in selected):
                invalid += 1
                continue
            raw.append(_candidate(
                augmentation,
                [grid for grid, _tokens, _elapsed in selected if grid is not None],
                sum(tokens for _grid, tokens, _elapsed in selected),
                max(elapsed for _grid, _tokens, elapsed in selected),
            ))
        path_stats.append({"augmentation_index": augmentation_index, "augmentation": augmentation.to_dict(), "per_test": per_test_stats, "aligned_paths": aligned})
    return raw, {"generated_candidate_count": generated, "invalid_candidate_count": invalid, "search_path_stats": path_stats}


def _valid_checkpoint(path: Path, task_id: str, identity: str) -> dict[str, Any] | None:
    try:
        value = _read(path)
    except (OSError, json.JSONDecodeError):
        return None
    record = value.get("record")
    if value.get("identity") != identity or value.get("task_id") != task_id or not isinstance(record, dict):
        return None
    if record.get("status") != "SUCCESS" or not isinstance(record.get("candidates"), list) or not record["candidates"]:
        return None
    return record


def _worker(
    worker_id: int,
    task_queue: Any,
    events: Any,
    ready: Any,
    start: Any,
    challenge_path: str,
    model_path: str,
    native_config_dir: str,
    settings: dict[str, Any],
    condition: str,
    spec: dict[str, Any],
    checkpoint_root: str,
    identity: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch
        from arc.io import load_dataset
        from inference.nvarc_native import NVARCNativeProvider
        from inference.nvarc_native_candidates import deduplicate_candidates

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in candidate-search worker")
        torch.cuda.set_device(0)
        if torch.cuda.current_device() != 0:
            raise RuntimeError("worker did not bind to its local CUDA device")
        provider = NVARCNativeProvider(model_path=Path(model_path), tokenizer_config_dir=Path(native_config_dir), device="cuda:0")
        events.put({"event": "MODEL_LOAD_STARTED", "worker_id": worker_id, "physical_gpu_id": worker_id})
        load_seconds = provider.load()
        ready.put({
            "event": "MODEL_READY", "worker_id": worker_id, "physical_gpu_id": worker_id,
            "gpu_name": torch.cuda.get_device_name(0), "local_cuda_device": torch.cuda.current_device(),
            "model_load_seconds": load_seconds, "model_instances": 1,
        })
        if not start.wait(timeout=MODEL_LOAD_TIMEOUT_SECONDS):
            raise TimeoutError("candidate-search start barrier timed out")
        tasks = load_dataset(Path(challenge_path))
        while True:
            task_id = task_queue.get()
            if task_id is None:
                return
            events.put({"event": "TASK_START", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "condition": condition})
            started = time.perf_counter()
            error: str | None = None
            for retry in range(2):
                try:
                    torch.cuda.reset_peak_memory_stats()
                    raw, metadata = _generate_task(provider, tasks[task_id], condition=condition, spec=spec, settings=settings)
                    unique = deduplicate_candidates(raw)
                    if not unique:
                        raise RuntimeError("no valid native candidates")
                    record = {
                        "status": "SUCCESS", "task_id": task_id, "worker_id": worker_id, "physical_gpu_id": worker_id,
                        "condition": condition, "candidates": [item.to_dict() for item in unique],
                        "unique_candidate_count": len(unique), "elapsed_seconds": time.perf_counter() - started,
                        "peak_allocated_vram_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)), "retry_count": retry,
                        **metadata,
                    }
                    checkpoint = Path(checkpoint_root) / "tasks" / f"{task_id}.json"
                    atomic_write_json(checkpoint, {"identity": identity, "task_id": task_id, "record": record})
                    if _valid_checkpoint(checkpoint, task_id, identity) is None:
                        raise RuntimeError("atomic checkpoint failed validation")
                    events.put({"event": "TASK_FROZEN", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "record": record})
                    break
                except Exception as exc:  # retry once without changing decoding config
                    error = f"{type(exc).__name__}: {exc}"
                    torch.cuda.empty_cache()
            else:
                events.put({"event": "TASK_FAILED", "worker_id": worker_id, "physical_gpu_id": worker_id, "task_id": task_id, "error": error})
    except Exception as exc:
        ready.put({"event": "WORKER_FAILED", "worker_id": worker_id, "physical_gpu_id": worker_id, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "search_config", "condition", "challenge_path", "model_path", "native_config_dir", "config", "output", "checkpoint_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path if name not in {"condition"} else str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest, search_config = _read(args.manifest), _read(args.search_config)
    condition = str(args.condition)
    task_ids = list(manifest.get("task_ids", ()))
    if (
        manifest.get("status") != "EVAL30_COHORT_FROZEN_BEFORE_NEW_SEARCH"
        or len(task_ids) != 30 or len(task_ids) != len(set(task_ids))
        or manifest.get("task_ids_hash") != _task_hash(task_ids)
        or condition not in {"beam2", "dfs_small", "dfs_medium"}
        or set(search_config.get("conditions", ())) != {"beam2", "dfs_small", "dfs_medium"}
        or not args.challenge_path.is_file() or not args.model_path.is_dir() or not args.native_config_dir.is_dir()
    ):
        raise ValueError("invalid frozen Evaluation30 identity or required search input")
    if _sha256_bytes(args.challenge_path.read_bytes()) != manifest["source_challenge_sha256"]:
        raise ValueError("evaluation challenge differs from frozen Evaluation60 source")
    config = _read(args.config)
    settings = config["B_augmentation_search"]
    spec = dict(search_config["conditions"][condition])
    if int(spec.get("augmentation_count", -1)) != 8 or int(spec.get("max_new_tokens", -1)) != int(settings["decode"]["max_new_tokens"]):
        raise ValueError("search config must preserve exact frozen Aug8/model decode limits")
    identity = _identity(manifest, search_config, condition, args.config)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen candidate artifact: {args.output}")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "tasks").mkdir(exist_ok=True)
    resumed: dict[str, Any] = {}
    if args.resume:
        for task_id in task_ids:
            value = _valid_checkpoint(args.checkpoint_dir / "tasks" / f"{task_id}.json", task_id, identity)
            if value is not None:
                resumed[task_id] = value
    unfinished = [task_id for task_id in task_ids if task_id not in resumed]
    hardware = inspect_hardware()
    gpu_names = tuple(gpu.name for gpu in hardware.gpus)
    if len(gpu_names) != 4 or any("NVIDIA L4" not in name for name in gpu_names):
        raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs, got {gpu_names}")
    context = get_context("spawn")
    task_queue, events, ready, start = context.Queue(), context.Queue(), context.Queue(), context.Event()
    children = []
    started = time.perf_counter()
    try:
        for worker_id in range(4):
            process = context.Process(target=_worker, args=(
                worker_id, task_queue, events, ready, start, str(args.challenge_path), str(args.model_path),
                str(args.native_config_dir), settings, condition, spec, str(args.checkpoint_dir), identity,
            ))
            process.start(); children.append(process)
            state = ready.get(timeout=MODEL_LOAD_TIMEOUT_SECONDS)
            print(json.dumps(state, sort_keys=True), flush=True)
            if state.get("event") != "MODEL_READY" or state.get("physical_gpu_id") != worker_id:
                raise RuntimeError(f"worker {worker_id} failed before ready: {state}")
        for task_id in unfinished:
            task_queue.put(task_id)
        for _ in children:
            task_queue.put(None)
        start.set()
        records = dict(resumed)
        failures: list[dict[str, Any]] = []
        while len(records) + len(failures) < len(task_ids):
            try:
                event = events.get(timeout=60.0)
            except queue.Empty:
                dead = [process.pid for process in children if process.exitcode not in (None, 0)]
                if dead:
                    raise RuntimeError(f"candidate-search worker exited unexpectedly: {dead}")
                continue
            payload = {key: value for key, value in event.items() if key != "record"}
            print(json.dumps(payload, sort_keys=True), flush=True)
            if event["event"] == "TASK_FROZEN":
                record = event["record"]
                if record["task_id"] in records:
                    raise RuntimeError(f"duplicate task record: {record['task_id']}")
                records[record["task_id"]] = record
            elif event["event"] == "TASK_FAILED":
                failures.append(event)
        if failures or set(records) != set(task_ids):
            raise RuntimeError(f"candidate search failed: failures={failures}, missing={sorted(set(task_ids) - set(records))}")
    finally:
        start.set()
        for process in children:
            process.join(timeout=30)
        for process in children:
            if process.is_alive():
                process.terminate(); process.join(timeout=10)
    ordered = {task_id: records[task_id] for task_id in task_ids}
    artifact = {
        "experiment_id": "ARC2_EVAL30_CANDIDATE_SEARCH_DIAGNOSTIC",
        "status": FROZEN_STATUS,
        "protocol": "Target-blind candidate generation only. ARC evaluation solutions are not available to this executable.",
        "condition": condition, "condition_config": spec, "identity": identity,
        "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"],
        "source_challenge_sha256": manifest["source_challenge_sha256"],
        "frozen_source_commit": manifest["frozen_inference"]["source_commit"],
        "hardware": hardware.to_dict(), "worker_gpu_mapping": {str(index): index for index in range(4)},
        "resumed_task_count": len(resumed), "runtime_seconds": time.perf_counter() - started,
        "gpu_seconds": sum(float(record["elapsed_seconds"]) for record in ordered.values()),
        "generated_candidate_count": sum(int(record["generated_candidate_count"]) for record in ordered.values()),
        "unique_candidate_count": sum(int(record["unique_candidate_count"]) for record in ordered.values()),
        "invalid_candidate_count": sum(int(record["invalid_candidate_count"]) for record in ordered.values()),
        "records": ordered,
    }
    atomic_write_json(args.output, artifact)
    print(json.dumps({"event": "CONDITION_CANDIDATES_FROZEN", "condition": condition, "task_count": len(ordered), "output": str(args.output), "solutions_opened": False}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
