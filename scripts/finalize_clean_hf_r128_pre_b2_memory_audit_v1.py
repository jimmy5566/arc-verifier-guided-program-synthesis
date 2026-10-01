#!/usr/bin/env python3
"""CPU-only finalization for the frozen Clean-HF R128 first-B2 memory audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _first(rows: list[dict[str, str]], checkpoint: str) -> dict[str, str]:
    for row in rows:
        if row["checkpoint"] == checkpoint:
            return row
    raise RuntimeError(f"missing checkpoint {checkpoint}")


def _gib(value: int) -> float:
    return round(value / (1024 ** 3), 4)


def _correct_r32_projection(path: Path) -> dict[str, Any]:
    rows = _rows(path / "KV_MEMORY_WATERFALL.csv")
    live = [row for row in rows if len(json.loads(row["cells"])) == 4]
    if not live:
        raise RuntimeError("R32 audit has no simultaneous four-cell snapshot")
    source = max(live, key=lambda row: int(row["total_unique_kv_bytes"]))
    cells = json.loads(source["cells"])
    four = sum(int(cell["unique_kv_bytes"]) for cell in cells)
    per_view = max((int(cell["unique_kv_bytes"]) for cell in cells), default=0)
    decision = json.loads((path / "DECISION.json").read_text(encoding="utf-8"))
    persistent = decision["persistent"]
    merged = per_view * 2
    split = per_view * 2
    projected = int(persistent["model_parameter_bytes"]) + int(persistent["model_buffer_bytes"]) + four + merged + split
    projection = {
        "corrected_end_state_projection_bug": True,
        "projection_source_snapshot": source["snapshot"],
        "four_current_kv_bytes": four,
        "estimated_b2_merged_kv_bytes": merged,
        "estimated_b2_split_kv_bytes": split,
        "projected_peak_expected_bytes": projected,
        "total_gpu_bytes": int(source["total_bytes"]),
        "fits_24gb": projected <= int(source["total_bytes"]),
        "assumption": "one pair is merged and split while the maximum simultaneous four-view snapshot remains live",
    }
    _atomic_json(path / "FOUR_VIEW_B2_MEMORY_PROJECTION.json", projection)
    hashes = {item.name: _sha256(item) for item in path.iterdir() if item.is_file() and item.name != "HASHES.json"}
    _atomic_json(path / "HASHES.json", hashes)
    return projection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--r32-audit", type=Path, required=True)
    args = parser.parse_args()

    rows = _rows(args.audit / "PRE_B2_MEMORY_WATERFALL.csv")
    p1, p5, p6 = (_first(rows, item) for item in ("P1", "P5", "P6"))
    b5, b6, b8, b9, b10 = (_first(rows, item) for item in ("B5", "B6", "B8", "B9", "B10"))
    alias = json.loads((args.audit / "CACHE_CONVERSION_STORAGE_ALIAS.json").read_text(encoding="utf-8"))
    persistent = json.loads(p1["extra"])
    p6_unattributed = int(p6["allocated_bytes"]) - int(persistent["model_parameter_bytes"]) - int(persistent["model_buffer_bytes"]) - int(p6["total_live_kv_bytes"])
    merged_delta = int(b5["allocated_bytes"]) - int(p6["allocated_bytes"])
    forward_delta = int(b6["allocated_bytes"]) - int(b5["allocated_bytes"])
    released_delta = int(b8["allocated_bytes"]) - int(b9["allocated_bytes"])
    post_reply_kv_delta = int(b10["total_live_kv_bytes"]) - int(p6["total_live_kv_bytes"])
    previous_oom_allocated = int(round(19.65 * (1024 ** 3)))
    previous_extra = previous_oom_allocated - int(p6["allocated_bytes"])
    split_bytes = int(alias["split_copy_bytes"])
    equivalent_splits = previous_extra / split_bytes
    projection = _correct_r32_projection(args.r32_audit)

    root = {
        "classification": "SPLIT_CONTIGUOUS_COPY",
        "classification_reason": "The first B2 leaves two independent split caches live after logical reply; their measured physical storage accounts for the historical allocation delta at about twenty retained B2 generations.",
        "secondary_observation": "The historical OOM also had 3.59 GiB reserved-but-unallocated memory, so allocator fragmentation was material but not the primary allocated-byte explanation.",
        "fresh_process": True,
        "p6_unattributed_bytes": p6_unattributed,
        "merged_legacy_allocation_bytes": merged_delta,
        "forward_allocation_delta_bytes": forward_delta,
        "split_copy_bytes": split_bytes,
        "transient_release_after_b8_bytes": released_delta,
        "post_reply_live_kv_delta_bytes": post_reply_kv_delta,
        "from_legacy_cache": alias["from_legacy_cache"],
        "prior_oom_allocated_gib": 19.65,
        "prior_extra_over_fresh_p6_bytes_estimate": previous_extra,
        "equivalent_retained_b2_split_generations_estimate": equivalent_splits,
        "micro_gate_retention": "NOT_RECONSTRUCTABLE_NO_COMMITTED_DEDICATED_RUNNER",
        "smallest_safe_fix": "Do not launch long dynamic-B2 R128 work on 24GB until cache ownership/lifetime is redesigned and parity-validated; release merged intermediates immediately after reply construction, then validate a cache-sharing or bounded-lifetime design separately.",
        "r32_projection": projection,
    }
    _atomic_json(args.audit / "ROOT_CAUSE.json", root)
    report = f"""# CLEAN_HF_R128_PRE_B2_MEMORY_AUDIT_V1

Target-blind, fresh-process, four-view R128 audit through exactly one physical B2 forward.

| checkpoint | allocated GiB | reserved GiB | live KV bytes |
|---|---:|---:|---:|
| P1 model + adapter | {_gib(int(p1['allocated_bytes']))} | {_gib(int(p1['reserved_bytes']))} | {p1['total_live_kv_bytes']} |
| P5 four prefills | {_gib(int(p5['allocated_bytes']))} | {_gib(int(p5['reserved_bytes']))} | {p5['total_live_kv_bytes']} |
| P6 before first B2 | {_gib(int(p6['allocated_bytes']))} | {_gib(int(p6['reserved_bytes']))} | {p6['total_live_kv_bytes']} |
| B5 before B2 forward | {_gib(int(b5['allocated_bytes']))} | {_gib(int(b5['reserved_bytes']))} | {b5['total_live_kv_bytes']} |
| B6 after B2 forward | {_gib(int(b6['allocated_bytes']))} | {_gib(int(b6['reserved_bytes']))} | {b6['total_live_kv_bytes']} |
| B8 after split/reconstruct | {_gib(int(b8['allocated_bytes']))} | {_gib(int(b8['reserved_bytes']))} | {b8['total_live_kv_bytes']} |
| B9 transient references released | {_gib(int(b9['allocated_bytes']))} | {_gib(int(b9['reserved_bytes']))} | {b9['total_live_kv_bytes']} |
| B10 logical replies resumed | {_gib(int(b10['allocated_bytes']))} | {_gib(int(b10['reserved_bytes']))} | {b10['total_live_kv_bytes']} |

## Attribution

- P6 unattributed allocation: {p6_unattributed} bytes ({_gib(p6_unattributed)} GiB).
- Merge before forward: {merged_delta} bytes ({_gib(merged_delta)} GiB).
- `DynamicCache.from_legacy_cache`: {alias['from_legacy_cache']} ({alias['shared_storage_count']}/{alias['legacy_storage_count']} storages shared).
- First B2 forward delta: {forward_delta} bytes ({_gib(forward_delta)} GiB).
- Split independent-cache copies: {split_bytes} bytes ({_gib(split_bytes)} GiB), sharing {alias['split_shared_with_merged_storage_count']} merged storages.
- After B9 releases the merged temporary representation, allocation drops {released_delta} bytes.  After B10 the live-KV set remains higher by {post_reply_kv_delta} bytes: the split copies are now owned by resumed DFS states.

## Classification

`SPLIT_CONTIGUOUS_COPY` is the primary explanation.  The historical 19.65-GiB allocated OOM exceeds fresh P6 by about {previous_extra} bytes, equivalent to {equivalent_splits:.2f} first-B2 split-copy increments.  The historical 3.59-GiB reserved-but-unallocated value is a secondary allocator-fragmentation observation.  No committed dedicated B2 micro/lane-swap/repeat runner was found, so stale micro-gate references cannot be quantitatively reconstructed.

The corrected R32 four-view+B2 projection uses the maximum simultaneous four-cell snapshot `{projection['projection_source_snapshot']}`, not END: {projection['projected_peak_expected_bytes']} bytes ({_gib(projection['projected_peak_expected_bytes'])} GiB), `fits_24gb={projection['fits_24gb']}`.  That projection covers a single B2 pair only; it does not prove a full R128 dynamic B2 search fits 24GB.
"""
    (args.audit / "REPORT.md").write_text(report, encoding="utf-8")
    hashes = {item.name: _sha256(item) for item in args.audit.iterdir() if item.is_file() and item.name != "HASHES.json"}
    _atomic_json(args.audit / "HASHES.json", hashes)
    print(json.dumps({"classification": root["classification"], "r32_projected_peak_bytes": projection["projected_peak_expected_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
