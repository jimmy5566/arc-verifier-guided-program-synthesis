"""Token masks for the frozen final-query NLL decomposition."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

IGNORE = -100
COMPONENTS = ("DEMONSTRATION_ASSISTANT_ALL", "FINAL_ASSISTANT_PREFIX", "FINAL_GRID_CONTENT", "FINAL_EOS")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def task_messages(task: dict) -> list[dict[str, str]]:
    from training_data.pipeline import ARCNativeInputAdapter
    messages = []
    for pair in task["train"]:
        messages.extend((
            {"role": "user", "content": ARCNativeInputAdapter.serialize_grid(pair["input"])},
            {"role": "assistant", "content": ARCNativeInputAdapter.serialize_grid(pair["output"])},
        ))
    if len(task["test"]) != 1 or "output" not in task["test"][0]:
        raise RuntimeError("EXPECTED_ONE_LABELED_DEV_TEST_PAIR")
    pair = task["test"][0]
    messages.extend((
        {"role": "user", "content": ARCNativeInputAdapter.serialize_grid(pair["input"])},
        {"role": "assistant", "content": ARCNativeInputAdapter.serialize_grid(pair["output"])},
    ))
    return messages


def masks_for_task(task: dict) -> dict:
    """Return disjoint/exhaustive assistant masks in causal-loss coordinates."""
    from training_data.pipeline import EXPECTED_TOKENS, native_tokenize_with_labels
    _, ids, labels = native_tokenize_with_labels(task_messages(task))
    masks = {name: [False] * len(ids) for name in COMPONENTS}
    turns, offset = [], 0
    for message in task_messages(task):
        size = 4 + len(message["content"])
        turn = ids[offset:offset + size]
        if len(turn) != size or turn[0] != 14 or turn[1] != EXPECTED_TOKENS[message["role"]] or turn[2] != 10 or turn[-1] != 15:
            raise RuntimeError("NATIVE_TURN_BOUNDARY_MISMATCH")
        if message["role"] == "assistant":
            turns.append((offset, size, message["content"]))
        offset += size
    if offset != len(ids) or not turns:
        raise RuntimeError("TURN_COVERAGE_MISMATCH")
    for start, size, _ in turns[:-1]:
        for pos in range(start, start + size):
            masks["DEMONSTRATION_ASSISTANT_ALL"][pos] = True
    start, size, content = turns[-1]
    for pos in range(start, start + 3):
        masks["FINAL_ASSISTANT_PREFIX"][pos] = True
    for pos in range(start + 3, start + 3 + len(content)):
        masks["FINAL_GRID_CONTENT"][pos] = True
    masks["FINAL_EOS"][start + size - 1] = True
    assistant = {i for i, label in enumerate(labels) if label != IGNORE}
    pieces = [i for name in COMPONENTS for i, enabled in enumerate(masks[name]) if enabled]
    if len(pieces) != len(set(pieces)) or set(pieces) != assistant:
        raise RuntimeError("ASSISTANT_MASK_NOT_MUTUALLY_EXHAUSTIVE")
    if any(ids[i] != labels[i] for i in assistant):
        raise RuntimeError("ASSISTANT_LABEL_ID_MISMATCH")
    content_ids = ids[start + 3:start + 3 + len(content)]
    if any(x not in range(11) for x in content_ids):
        raise RuntimeError("FINAL_GRID_TOKEN_CONTRACT_FAIL")
    loss_masks = {name: [False] * (len(ids) - 1) for name in COMPONENTS}
    for name in COMPONENTS:
        for pos, enabled in enumerate(masks[name]):
            if enabled:
                if pos == 0:
                    raise RuntimeError("UNSCORABLE_LABEL_AT_ZERO")
                loss_masks[name][pos - 1] = True
    all_loss = [labels[i + 1] != IGNORE for i in range(len(ids) - 1)]
    if sum(sum(mask) for mask in loss_masks.values()) != sum(all_loss):
        raise RuntimeError("LOSS_MASK_TOKEN_COUNT_MISMATCH")
    if any(sum(int(loss_masks[name][i]) for name in COMPONENTS) != int(all_loss[i]) for i in range(len(all_loss))):
        raise RuntimeError("LOSS_MASK_UNION_MISMATCH")
    counts = {name: sum(mask) for name, mask in loss_masks.items()}
    return {"ids": ids, "labels": labels, "loss_masks": loss_masks,
            "assistant_token_count": len(assistant), "component_token_counts": counts,
            "final_grid_plus_eos_token_count": counts["FINAL_GRID_CONTENT"] + counts["FINAL_EOS"]}


def component_sums(losses: list[float], masks: dict[str, list[bool]]) -> dict:
    if any(len(mask) != len(losses) for mask in masks.values()):
        raise RuntimeError("LOSS_VECTOR_MASK_LENGTH_MISMATCH")
    values = {}
    for name, mask in masks.items():
        selected = [float(x) for x, enabled in zip(losses, mask) if enabled]
        values[name] = {"nll_sum": sum(selected), "token_count": len(selected)}
    legacy_mask = [any(masks[name][i] for name in COMPONENTS) for i in range(len(losses))]
    legacy = sum(float(x) for x, enabled in zip(losses, legacy_mask) if enabled)
    if abs(sum(values[name]["nll_sum"] for name in COMPONENTS) - legacy) > 1e-5:
        raise RuntimeError("LEGACY_ALL_ASSISTANT_RECONSTRUCTION_FAIL")
    values["ALL_ASSISTANT_LEGACY"] = {"nll_sum": legacy, "token_count": sum(legacy_mask)}
    values["FINAL_GRID_PLUS_EOS"] = {
        "nll_sum": values["FINAL_GRID_CONTENT"]["nll_sum"] + values["FINAL_EOS"]["nll_sum"],
        "token_count": values["FINAL_GRID_CONTENT"]["token_count"] + values["FINAL_EOS"]["token_count"],
    }
    return values
