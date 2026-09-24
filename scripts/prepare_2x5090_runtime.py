"""Validate a two-RTX-5090 independent-worker plan without model loading."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def worker_plan(inventory: list[dict[str, object]]) -> dict[str, object]:
    if len(inventory) != 2:
        raise ValueError(f"GPU_INVENTORY_MISMATCH: expected exactly 2 devices, found {len(inventory)}")
    workers = []
    for physical_id, gpu in enumerate(inventory):
        if gpu.get("name") != "NVIDIA GeForce RTX 5090" or gpu.get("capability") != [12, 0]:
            raise ValueError(f"BLACKWELL_GPU_MISMATCH: physical_gpu_id={physical_id}, gpu={gpu}")
        workers.append(
            {
                "worker_id": physical_id,
                "physical_gpu_id": physical_id,
                "cuda_visible_devices": str(physical_id),
                "execution": "independent_arc_task_worker",
                "model_instances": 1,
                "tensor_parallelism": False,
                "task_micro_batching": False,
                "scientific_semantics": "one task-local model/adapter trajectory per worker",
            }
        )
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "ARC2_2X5090_RUNTIME_V1",
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "worker_count": 2,
        "workers": workers,
        "dynamic_queue": True,
        "scientific_code_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if not isinstance(inventory, list):
        raise SystemExit("GPU_INVENTORY_SCHEMA_INVALID")
    plan = worker_plan(inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, plan)
    print(json.dumps({"event": "TWO_5090_RUNTIME_READY", **plan}, sort_keys=True))


if __name__ == "__main__":
    main()
