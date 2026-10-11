"""Shared, target-blind execution contracts for the frozen E04-C pilot.

This module has no torch or transformers imports.  It validates the pinned
TRAIN schedules and constructs the same native samples used by the worker.
The worker performs the separate production-tokenizer/collator gate before it
constructs an optimizer.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/e04_c_matched_fixed_turn_rotation_repair_pilot_v1"
SCHEDULE = BASE / "schedule_freeze_v1"
PROTOCOL_ID = "E04_C_MATCHED_FIXED_TURN_ROTATION_REPAIR_PILOT_V1"
ARMS = ("CONTROL_NO_TRANSFORM", "TREATMENT_FIXED_TURN_ROTATION")
ARM_FILE = {ARMS[0]: "CONTROL_SCHEDULE.json", ARMS[1]: "TREATMENT_SCHEDULE.json"}
PER_ARM_CAP_SECONDS = 3600
JOINT_CAP_SECONDS = 7200
IGNORE_INDEX = -100


class E04CFailure(RuntimeError):
    pass


def fail(code: str) -> None:
    raise E04CFailure(code)


def sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _task_for(arm: str, episode: dict[str, Any]) -> dict[str, Any]:
    if arm == ARMS[0]:
        return episode["control_task"]
    return episode["treatment_task"]


def _task_hash_for(arm: str, episode: dict[str, Any]) -> str:
    return episode["control_task_sha256"] if arm == ARMS[0] else episode["treatment_task_sha256"]


def load_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = read_json(BASE / "E04_C_PRELAUNCH_PROTOCOL_V1.json")
    schedule_binding = read_json(SCHEDULE / "MATCHED_SCHEDULE_BINDING_V1.json")
    static = read_json(SCHEDULE / "NATIVE_TOKEN_STATIC_PREFLIGHT_V1.json")
    schedules = {arm: read_json(SCHEDULE / ARM_FILE[arm]) for arm in ARMS}
    if protocol.get("protocol_id") != PROTOCOL_ID or schedule_binding.get("protocol_id") != PROTOCOL_ID:
        fail("E04C_PROTOCOL_IDENTITY")
    if sha_path(SCHEDULE / "MATCHED_SCHEDULE_BINDING_V1.json") != protocol["schedule"]["binding_sha256"]:
        fail("E04C_SCHEDULE_BINDING_HASH")
    if sha_path(SCHEDULE / "NATIVE_TOKEN_STATIC_PREFLIGHT_V1.json") != protocol["schedule"]["native_token_preflight_sha256"]:
        fail("E04C_STATIC_PREFLIGHT_HASH")
    if static.get("status") != "PASS_NATIVE_RUNTIME_SERIALIZATION_CONTRACT":
        fail("E04C_STATIC_PREFLIGHT_STATUS")
    return protocol, schedule_binding, static, schedules


def build_arm_samples(arm: str, schedule: dict[str, Any]) -> list[dict[str, Any]]:
    """Build ordered native samples while checking every frozen schedule field."""
    if arm not in ARMS or schedule.get("arm") != ("CONTROL" if arm == ARMS[0] else "TREATMENT"):
        fail("E04C_ARM_IDENTITY")
    episodes = schedule.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 384:
        fail("E04C_SCHEDULE_ROWS")
    from training_data.pipeline import task_to_sample
    samples: list[dict[str, Any]] = []
    for slot, episode in enumerate(episodes):
        if episode.get("slot") != slot or not isinstance(episode.get("pair_id"), str):
            fail("E04C_SCHEDULE_SLOT")
        task = _task_for(arm, episode)
        if hashlib.sha256(canon(task)).hexdigest() != _task_hash_for(arm, episode):
            fail("E04C_SCHEDULE_TASK_HASH:" + str(slot))
        sample = task_to_sample({"source_id": "e04c-" + arm.lower() + "-" + episode["pair_id"], **task})
        if len(sample["input_ids"]) != len(sample["labels"]) or sample["attention_mask"] != [1] * len(sample["input_ids"]):
            fail("E04C_NATIVE_SAMPLE_SHAPE:" + str(slot))
        supervised = sum(value != IGNORE_INDEX for value in sample["labels"])
        if supervised != sample["assistant_token_count"] or supervised < 1:
            fail("E04C_NATIVE_SAMPLE_LABELS:" + str(slot))
        samples.append({"slot": slot, "episode": episode, "sample": sample})
    return samples


def static_schedule_preflight() -> dict[str, Any]:
    """CPU-only schedule proof, used by tests and before production imports."""
    protocol, binding, static, schedules = load_frozen_inputs()
    control = build_arm_samples(ARMS[0], schedules[ARMS[0]])
    treatment = build_arm_samples(ARMS[1], schedules[ARMS[1]])
    static_rows = static.get("per_slot_rows")
    if not isinstance(static_rows, list) or len(static_rows) != 384:
        fail("E04C_STATIC_ROWS")
    protected = 0
    intervention = 0
    for c, t, expected in zip(control, treatment, static_rows, strict=True):
        slot = c["slot"]
        if t["slot"] != slot or c["episode"]["pair_id"] != t["episode"]["pair_id"]:
            fail("E04C_CROSS_ARM_SLOT_ALIGNMENT:" + str(slot))
        for key, observed in (("control_sequence_length", len(c["sample"]["input_ids"])),
                              ("treatment_sequence_length", len(t["sample"]["input_ids"])),
                              ("control_supervised_token_count", c["sample"]["assistant_token_count"]),
                              ("treatment_supervised_token_count", t["sample"]["assistant_token_count"])):
            if expected.get(key) != observed:
                fail("E04C_STATIC_COUNT_MISMATCH:" + str(slot))
        if slot >= 96:
            protected += 1
            if (c["episode"]["control_task"] != t["episode"]["treatment_task"] or
                    c["sample"]["input_ids"] != t["sample"]["input_ids"] or
                    c["sample"]["labels"] != t["sample"]["labels"]):
                fail("E04C_PROTECTED_REPLAY_MISMATCH:" + str(slot))
        else:
            intervention += 1
            if c["episode"]["control_task"] == t["episode"]["treatment_task"]:
                fail("E04C_INTERVENTION_NOT_REPLACED:" + str(slot))
    totals = {
        arm: {"total_tokens": sum(len(row["sample"]["input_ids"]) for row in values),
              "supervised_tokens": sum(row["sample"]["assistant_token_count"] for row in values)}
        for arm, values in ((ARMS[0], control), (ARMS[1], treatment))
    }
    expected_tokens = protocol["schedule"]
    if any(value["total_tokens"] != expected_tokens["per_arm_total_tokens"] or value["supervised_tokens"] != expected_tokens["per_arm_supervised_tokens"] for value in totals.values()):
        fail("E04C_ARM_TOTALS")
    if protected != 288 or intervention != 96 or binding.get("optimizer_steps") != 96:
        fail("E04C_FROZEN_SLOT_ALLOCATION")
    return {"status": "PASS_CPU_NATIVE_SCHEDULE", "protocol_id": PROTOCOL_ID, "slots": 384,
            "intervention_slots": intervention, "protected_replay_slots": protected, "arm_totals": totals,
            "model_imported": False, "tokenizer_loaded": False, "optimizer_constructed": False,
            "target_sidecar_accessed": False}
