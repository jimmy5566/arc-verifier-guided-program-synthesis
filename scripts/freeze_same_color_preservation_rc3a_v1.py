#!/usr/bin/env python3
"""Freeze RC3A-matched, CPU-only Stage-A schedules.

The only permitted arm difference is the Director-approved minimum transformer
token delta forced by one mandatory rotate control sample.  This program does
not import model/runtime libraries and cannot launch scientific work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training_data.pipeline import IGNORE_INDEX, task_to_sample


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf8", newline="\n") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def totals(entries):
    return {
        "episodes": len(entries),
        "transformer_tokens": sum(x["sequence_length"] for x in entries),
        "supervised_tokens": sum(x["supervised_token_count"] for x in entries),
        "by_role": dict(sorted(Counter(x["role"] for x in entries).items())),
        "by_family": dict(sorted(Counter(x["family"] for x in entries).items())),
    }


def load_pool(train):
    pools = defaultdict(lambda: defaultdict(list))
    for line in Path(train).read_text(encoding="utf8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        sample = task_to_sample({"source_id": record["episode_id"], **record["task"]})
        if not any(x != IGNORE_INDEX for x in sample["labels"]):
            raise RuntimeError("ZERO_SUPERVISION")
        pools[record["role"]][record["family"]].append({
            "episode_id": record["episode_id"], "role": record["role"], "family": record["family"],
            "sequence_length": len(sample["input_ids"]), "supervised_token_count": sample["assistant_token_count"],
            "token_label_sha256": digest({"input_ids": sample["input_ids"], "labels": sample["labels"]}),
        })
    for family_pool in pools.values():
        for rows in family_pool.values():
            rows.sort(key=lambda x: x["episode_id"])
    return pools


def smallest_matching_intervention(pools, counts):
    """Find the global minimum token delta with exact supervision equality.

    Each family may reuse a frozen source row in a finite schedule; this is the
    same sampling convention used by the original 4004-row source corpus.
    The deterministic DP searches all possible supervised-token totals for the
    mandated 16 control positions and compatible sixteen same-color positions.
    """
    positions = []
    for family in sorted(counts):
        positions.extend([family] * int(counts[family]))
    if len(positions) != 16:
        raise RuntimeError("INTERVENTION_COUNT_NOT_16")
    control = {family: pools["RETENTION_TRAIN"][family] for family in counts}
    treatment = pools["ATOMIC_REPAIR"]["same color"]
    if not treatment or any(not control[f] for f in counts):
        raise RuntimeError("REQUIRED_POOL_UNAVAILABLE")

    treatment_by_supervision = defaultdict(list)
    for row in treatment:
        treatment_by_supervision[row["supervised_token_count"]].append(row)
    # Repetition is deliberate and deterministic when only three 279-token
    # same-color source rows exist; no row bytes are changed or fabricated.
    treatment_sums = {0: []}
    for _ in range(16):
        updated = {}
        for prior, chosen in treatment_sums.items():
            for supervision, rows in sorted(treatment_by_supervision.items()):
                total = prior + supervision
                if total not in updated:
                    updated[total] = chosen + [rows[0]]
        treatment_sums = updated

    # Sort controls by (invariant, episode id) so the DP's first solution is a
    # deterministic witness.  The outer candidate scan minimizes abs delta.
    options = []
    for family in positions:
        options.append(sorted(control[family], key=lambda x: (x["sequence_length"] - 2 * x["supervised_token_count"], x["episode_id"])))
    states = {0: []}
    for rows in options:
        updated = {}
        for prior, chosen in states.items():
            for row in rows:
                total = prior + row["supervised_token_count"]
                if total not in updated:
                    updated[total] = chosen + [row]
        states = updated

    candidates = []
    for supervision, chosen_control in states.items():
        chosen_treatment = treatment_sums.get(supervision)
        if not chosen_treatment:
            continue
        delta = sum(x["sequence_length"] for x in chosen_control) - sum(x["sequence_length"] for x in chosen_treatment)
        candidates.append((abs(delta), delta, supervision, chosen_control, chosen_treatment))
    if not candidates:
        raise RuntimeError("NO_EXACT_SUPERVISION_MATCH")
    # tie-break full frozen row identity to make a reproducible global solver.
    candidates.sort(key=lambda x: (x[0], x[1], x[2], tuple(r["episode_id"] for r in x[3]), tuple(r["episode_id"] for r in x[4])))
    return candidates[0]


def common_schedule(config, pools):
    shared = config["shared_training"]
    cycle = shared["role_cycle"]
    target = int(shared["nominal_transformer_tokens_per_arm"])
    alignment = int(shared["schedule_alignment_slots"])
    accum = int(shared["gradient_accumulation"])
    if len(cycle) != 20 or cycle.count("RETENTION_TRAIN") != 4 or alignment % accum:
        raise RuntimeError("INVALID_SHARED_SCHEDULE")
    weights = {"ATOMIC_REPAIR": shared["atomic_family_weights"], "COMPOSITION_REPAIR": shared["composition_family_weights"]}
    seed = int(shared["seed"])
    cursor = Counter()
    output, slot, replay, token_total = [], 0, 0, 0
    families = {role: sorted(value) for role, value in weights.items()}
    while token_total < target or slot % accum or slot % alignment:
        role = cycle[slot % len(cycle)]
        if role == "RETENTION_TRAIN":
            # Non-intervention retention source is irrelevant at designated
            # positions and will be replaced later; retain the V2 weighted
            # source for every other retention position.
            source_weights = config["control_replay_family_weights"]
        else:
            source_weights = weights[role]
        ordered = sorted(source_weights)
        selector = int(hashlib.sha256(f"{seed}:{role}:{replay if role == 'RETENTION_TRAIN' else slot}".encode()).hexdigest(), 16)
        cumulative = 0.0
        point = (selector % 10_000_000) / 10_000_000
        family = ordered[-1]
        for candidate in ordered:
            cumulative += float(source_weights[candidate])
            if point < cumulative:
                family = candidate
                break
        rows = pools[role][family]
        row = dict(rows[cursor[(role, family)] % len(rows)])
        cursor[(role, family)] += 1
        row.update({"slot": slot, "intervention_slot": role == "RETENTION_TRAIN" and replay % 5 == 0})
        output.append(row)
        token_total += row["sequence_length"]
        if role == "RETENTION_TRAIN":
            replay += 1
        slot += 1
    if replay != 80 or slot != 400 or slot // accum != 100:
        raise RuntimeError("UNEXPECTED_SCHEDULE_SHAPE")
    return output


def build(config, pools):
    counts = config["matching_contract"]["proportional_control_reduction_counts"]
    minimum, signed, supervision, selected_control, selected_treatment = smallest_matching_intervention(pools, counts)
    common = common_schedule(config, pools)
    intervention_indices = [i for i, row in enumerate(common) if row["intervention_slot"]]
    if len(intervention_indices) != 16:
        raise RuntimeError("REPLAY_INTERVENTION_ALLOCATION_FAIL")
    control = [dict(x) for x in common]
    treatment = [dict(x) for x in common]
    for index, control_row, treatment_row in zip(intervention_indices, selected_control, selected_treatment):
        c, t = dict(control_row), dict(treatment_row)
        c.update({"slot": index, "intervention_slot": True, "paired_replacement_for": t["episode_id"]})
        t.update({"slot": index, "intervention_slot": True, "source_role": t["role"], "role": "RETENTION_TRAIN", "paired_replacement_for": c["episode_id"]})
        control[index], treatment[index] = c, t
    control_totals, treatment_totals = totals(control), totals(treatment)
    if control_totals["supervised_tokens"] != treatment_totals["supervised_tokens"]:
        raise RuntimeError("SUPERVISION_NOT_EQUAL")
    actual_signed = control_totals["transformer_tokens"] - treatment_totals["transformer_tokens"]
    if actual_signed != signed or abs(actual_signed) != minimum:
        raise RuntimeError("SOLVER_WITNESS_MISMATCH")
    ratio = minimum / min(control_totals["transformer_tokens"], treatment_totals["transformer_tokens"])
    limit = float(config["matching_contract"]["transformer_token_difference_rule"]["maximum_fraction_of_smaller_arm_total"])
    if ratio > limit:
        raise RuntimeError("TRANSFORMER_DELTA_EXCEEDS_BOUND")
    if Counter(x["family"] for x in (control[i] for i in intervention_indices)) != Counter(counts):
        raise RuntimeError("PROPORTIONAL_FAMILY_COUNT_FAIL")
    if any(treatment[i]["family"] != "same color" for i in intervention_indices):
        raise RuntimeError("TREATMENT_FAMILY_FAIL")
    if any(control[i]["role"] != treatment[i]["role"] for i in range(len(control))):
        raise RuntimeError("ROLE_SLOT_FAIL")
    for i, (c, t) in enumerate(zip(control, treatment)):
        if i not in intervention_indices and c != t:
            raise RuntimeError("NON_INTERVENTION_DRIFT")
    return control, treatment, {"minimum_absolute_transformer_token_delta": minimum, "signed_control_minus_treatment_transformer_tokens": signed, "exact_matched_intervention_supervised_tokens": supervision, "transformer_token_difference_ratio": ratio, "intervention_indices": intervention_indices}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf8"))
    if config.get("scientific_training_authorized") is not False or config.get("model_loading_authorized") is not False:
        raise RuntimeError("PREPARATION_ONLY_REQUIRED")
    train = Path(config["train"]["path"])
    if sha(train) != config["train"]["sha256"]:
        raise RuntimeError("TRAIN_HASH_MISMATCH")
    if args.out.exists():
        raise RuntimeError("OUTPUT_ALREADY_EXISTS")
    control, treatment, solver = build(config, load_pool(train))
    common = {"schema_version": 1, "protocol_id": config["protocol_id"], "config_sha256": sha(args.config), "train_sha256": sha(train), "checkpoint_manifest": config["checkpoint_manifest"], "shared_training": config["shared_training"], "matching_contract": config["matching_contract"], "cpu_only": True, "model_loaded": False, "optimizer_constructed": False, "scientific_training_started": False}
    args.out.mkdir(parents=True)
    control_path, treatment_path = args.out / "CONTROL_TRAINING_SCHEDULE.json", args.out / "TREATMENT_TRAINING_SCHEDULE.json"
    atomic(control_path, {**common, "arm": "CONTROL", "episodes": control, "totals": totals(control)})
    atomic(treatment_path, {**common, "arm": "TREATMENT", "episodes": treatment, "totals": totals(treatment)})
    binding = {"schema_version": 1, **common, "solver_identity": "freeze_same_color_preservation_rc3a_v1.py:deterministic-global-DP-v1", "control_schedule_sha256": sha(control_path), "treatment_schedule_sha256": sha(treatment_path), "control_totals": totals(control), "treatment_totals": totals(treatment), "solver_result": solver, "status": "FROZEN_CPU_ONLY_RC3A_MATCHED_SCHEDULES"}
    atomic(args.out / "RC3A_MATCHED_SCHEDULE_BINDING_V1.json", binding)
    print(json.dumps({"status": binding["status"], **solver, "control_transformer_tokens": binding["control_totals"]["transformer_tokens"], "treatment_transformer_tokens": binding["treatment_totals"]["transformer_tokens"]}, sort_keys=True))


if __name__ == "__main__":
    main()
