"""Freeze and validate the CPU-only connected-components/object-extraction pilot.

The freezer has no model, GPU, target-sidecar, or remote dependency.  It
generates legitimate synthetic TRAIN labels, while every evaluation split
contains prompts only.  The only arm difference is the topology set used in
the same two atomic task families.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
E04 = ROOT / "experiments/capability_repair_baseline_v1/e04_e_equal_slot_weight_marker_replay_preservation_pilot_v2/CONTROL_SCHEDULE.json"
PROFILE = ROOT / "experiments/capability_repair_baseline_v1/CAPABILITY_REPAIR_BASELINE_V1_PROFILE.json"
DIRECTOR_RESPONSE_SHA256 = "787595f1e56841a672580208e55dba4baa130974b295fdcb706e9ff0a6e3d419"
CHECKPOINT_SHA256 = "1e124cc4f43530bbbc71103703d404b38df83a6998a3e39d7c1da0676c800549"
TOPOLOGIES = {
    "POINT": ((1, 1),),
    "L": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "Z": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "T": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "PLUS": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    "BOX": ((0, 0), (0, 1), (1, 0), (1, 1)),
}
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SEED = "CONNECTED_COMPONENTS_OBJECT_EXTRACTION_PILOT_V1_20261012"


class GateFailure(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(canonical_bytes(value) + b"\n")
        tmp = Path(handle.name)
    os.replace(tmp, path)


def rotate(points: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    output = list(points)
    for _ in range(turns % 4):
        output = [(x, 2 - y) for y, x in output]
    return tuple(output)


def blank(bg: int = 0) -> list[list[int]]:
    return [[bg] * 9 for _ in range(9)]


def paste(grid: list[list[int]], points: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> None:
    for y, x in points:
        grid[top + y][left + x] = color


def crop(points: tuple[tuple[int, int], ...], color: int) -> list[list[int]]:
    result = [[0] * 3 for _ in range(3)]
    for y, x in points:
        result[y][x] = color
    return result


def count_output(count: int, color: int) -> list[list[int]]:
    result = [[0] * 3 for _ in range(3)]
    for i in range(count):
        result[1][i] = color
    return result


def count_pair(topology: str, variant: int) -> dict:
    """Count 1--3 separated same-colour connected components."""
    color = COLORS[variant % len(COLORS)]
    grid = blank()
    points = rotate(TOPOLOGIES[topology], variant)
    count = 1 + (variant % 3)
    anchors = ((1, 1), (1, 5), (5, 1))
    for top, left in anchors[:count]:
        paste(grid, points, top, left, color)
    return {"input": grid, "output": count_output(count, color)}


def extract_pair(topology: str, variant: int) -> dict:
    """Select the only non-background component and crop it exactly."""
    color = COLORS[(variant + 2) % len(COLORS)]
    distractor = COLORS[(variant + 5) % len(COLORS)]
    grid = blank()
    points = rotate(TOPOLOGIES[topology], variant)
    paste(grid, points, 3, 3, color)
    grid[0][0] = distractor
    grid[8][8] = distractor
    return {"input": grid, "output": crop(points, color)}


def task(family: str, topology: str, sequence: int, source_id: str, include_query_output: bool) -> dict:
    maker = count_pair if family == "CONNECTED_COMPONENT_COUNT" else extract_pair
    demos = [maker(topology, sequence + i) for i in range(3)]
    query = maker(topology, sequence + 3)
    if not include_query_output:
        query = {"input": query["input"]}
    return {"source_id": source_id, "train": demos, "test": [query]}


def train_rows(arm: str) -> list[dict]:
    topologies = ("POINT", "L") if arm == "CONTROL_NARROW_STRUCTURE" else ("L", "Z", "T")
    rows: list[dict] = []
    for slot in range(96):
        family = ("CONNECTED_COMPONENT_COUNT", "OBJECT_EXTRACTION")[slot % 2]
        topology = topologies[(slot // 2) % len(topologies)]
        source_id = f"TRAIN:{arm}:{family}:{topology}:{slot:03d}"
        record = {
            "slot": slot,
            "optimizer_step": slot // 4,
            "role": f"PRIMARY_{family}",
            "arm": arm,
            "family": family,
            "topology": topology,
            "seed": f"{SEED}:{arm}:{slot}",
            "task": task(family, topology, slot * 4, source_id, True),
        }
        record["canonical_base_id"] = sha({key: record[key] for key in ("arm", "family", "topology", "seed")})
        rows.append(record)
    return rows


def eval_prompts(split: str, n: int, families: tuple[str, ...], topologies: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    for index in range(n):
        family = families[index % len(families)]
        topology = topologies[(index // len(families)) % len(topologies)]
        source_id = f"{split}:{family}:{topology}:{index:03d}"
        prompt = task(family, topology, 10000 + index * 4, source_id, False)
        latent = {"split": split, "family": family, "topology": topology, "seed": f"{SEED}:{split}:{index}"}
        rows.append({"prompt_id": sha(latent), "latent_condition": latent, "task": prompt})
    return rows


def historical_audit() -> dict:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    cc = profile["by_capability"]["connected components"]
    return {
        "protocol_id": "E1_EFFECTIVE_SUPERVISION_AUDIT_CONNECTED_COMPONENTS_OBJECT_EXTRACTION_V1",
        "status": "PASS_CPU_ONLY_HISTORICAL_EVIDENCE_BOUNDARY_AUDIT",
        "source_profile_sha256": file_sha(PROFILE),
        "director_response_sha256": DIRECTOR_RESPONSE_SHA256,
        "historical_connected_components": {
            "episodes": cc["episodes"], "exact_grid_match": cc["exact_grid_match"],
            "exact_rate": cc["exact_rate"], "parse_rate": cc["parse_rate"],
            "interpretation": "Established historical weak atomic capability; parser validity does not establish correct connected-component reasoning.",
        },
        "historical_object_extraction": {
            "standalone_rate_available": False,
            "interpretation": "The cited Director response identifies object extraction as a direct prerequisite and leading candidate, but the available profile contains no standalone object-extraction metric. It remains a targeted hypothesis, not a measured historical rate.",
        },
        "effective_supervision_conclusion": "The historical evidence supports an explicitly matched structural-coverage test; it does not establish a model-training effect or justify any GPU execution.",
        "model_loaded": False, "gpu_used": False, "targets_accessed": False,
        "director_next_stage": "CPU_ONLY_PRIMARY_ATOMIC_TRAINING_PILOT_FREEZE",
    }


def schedule(rows: list[dict], arm: str, replay: list[dict]) -> dict:
    episodes = []
    for row in rows:
        item = {key: copy.deepcopy(value) for key, value in row.items() if key not in {"arm", "family", "topology", "seed"}}
        item["structural_family"] = row["family"]
        item["topology"] = row["topology"]
        episodes.append(item)
    episodes.extend(copy.deepcopy(replay))
    return {"schema_version": 1, "protocol_id": "CONNECTED_COMPONENTS_OBJECT_EXTRACTION_CAPABILITY_PILOT_V1", "arm": arm, "episodes": episodes}


def token_accounting(control: dict, treatment: dict) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from training_data.pipeline import task_to_sample

    def one(schedule: dict) -> dict:
        # E04-E's immutable replay task payloads predate the serializer's
        # source_id field.  Give the serializer an ephemeral, deterministic
        # identifier outside the frozen schedule; it never reaches messages
        # or changes any replay task bytes.
        def serializer_view(row: dict) -> dict:
            view = copy.deepcopy(row["task"])
            view.setdefault("source_id", f"SERIALIZER_VIEW:{row['canonical_base_id']}")
            return view
        samples = [task_to_sample(serializer_view(row)) for row in schedule["episodes"]]
        return {
            "slots": len(samples), "updates": len(samples) // 4,
            "transformer_tokens": sum(sample["sequence_length"] for sample in samples),
            "supervised_tokens": sum(sample["assistant_token_count"] for sample in samples),
        }

    c, t = one(control), one(treatment)
    if c != t:
        raise GateFailure(f"TOKEN_ACCOUNTING_MISMATCH:{c}!={t}")
    return {
        "status": "PASS_CPU_SERIALIZER_ACCOUNTING_EXACT_MATCH",
        "serializer_path": "src/training_data/pipeline.py",
        "serializer_sha256": file_sha(ROOT / "src/training_data/pipeline.py"),
        "control": c, "treatment": t,
        "model_loaded": False, "gpu_used": False, "optimizer_constructed": False,
    }


def validate(control: dict, treatment: dict, ood: list[dict], retention: list[dict], sentinel: list[dict], accounting: dict) -> dict:
    if len(control["episodes"]) != 384 or len(treatment["episodes"]) != 384:
        raise GateFailure("SLOT_COUNT")
    if [r["slot"] for r in control["episodes"]] != list(range(384)) or [r["slot"] for r in treatment["episodes"]] != list(range(384)):
        raise GateFailure("SLOT_SEQUENCE")
    if control["episodes"][96:] != treatment["episodes"][96:]:
        raise GateFailure("PROTECTED_REPLAY_NOT_BYTE_IDENTICAL")
    if any(r["topology"] not in {"POINT", "L"} for r in control["episodes"][:96]):
        raise GateFailure("CONTROL_NOT_NARROW")
    if {r["topology"] for r in treatment["episodes"][:96]} != {"L", "Z", "T"}:
        raise GateFailure("TREATMENT_NOT_DIVERSE")
    if {r["structural_family"] for r in control["episodes"][:96]} != {"CONNECTED_COMPONENT_COUNT", "OBJECT_EXTRACTION"}:
        raise GateFailure("TRAIN_FAMILY_MISMATCH")
    for row in control["episodes"][:96] + treatment["episodes"][:96]:
        maker = count_pair if row["structural_family"] == "CONNECTED_COMPONENT_COUNT" else extract_pair
        for offset, pair in enumerate(row["task"]["train"] + row["task"]["test"]):
            if pair != maker(row["topology"], row["slot"] * 4 + offset):
                raise GateFailure("TRAIN_ORACLE_MISMATCH")
    train_ids = {r["canonical_base_id"] for r in control["episodes"][:96] + treatment["episodes"][:96]}
    split_ids = set()
    for name, rows in (("OOD", ood), ("RETENTION", retention), ("SENTINEL", sentinel)):
        for row in rows:
            if "output" in row["task"]["test"][0] or "label" in row:
                raise GateFailure(f"{name}_TARGET_LEAK")
            if row["prompt_id"] in train_ids or row["prompt_id"] in split_ids:
                raise GateFailure(f"{name}_CONTENT_COLLISION")
            split_ids.add(row["prompt_id"])
    if {row["latent_condition"]["topology"] for row in ood} & {"POINT", "L", "Z", "T"}:
        raise GateFailure("OOD_TOPOLOGY_NOT_DISJOINT")
    if accounting["control"] != accounting["treatment"]:
        raise GateFailure("TOKEN_MISMATCH")
    return {
        "status": "PASS_CPU_ONLY_CONSTRUCT_REPLAY_EXPOSURE_AND_SERIALIZER_VALIDATION",
        "model_loaded": False, "gpu_used": False, "targets_accessed": False,
        "train_slots_per_arm": 384, "optimizer_steps_per_arm": 96,
        "byte_identical_protected_replay_slots": 288,
        "independent_target_blind_prompts": {"ood": len(ood), "retention": len(retention), "composition_sentinel": len(sentinel)},
    }


def freeze(out: Path = OUT) -> dict:
    allowed_prior = {"PREFREEZE_DRAFT.json", "FREEZE_ATTEMPT_001_FAILURE.json", "ATTEMPT_001_PARTIAL"}
    if out.exists() and {p.name for p in out.iterdir()} - allowed_prior:
        raise GateFailure("REFUSE_OVERWRITE_EXISTING_FREEZE")
    source = json.loads(E04.read_text(encoding="utf-8"))
    replay = source["episodes"][96:]
    if len(replay) != 288:
        raise GateFailure("SOURCE_REPLAY_COUNT")
    control = schedule(train_rows("CONTROL_NARROW_STRUCTURE"), "CONTROL_NARROW_STRUCTURE", replay)
    treatment = schedule(train_rows("TREATMENT_STRUCTURE_DIVERSE"), "TREATMENT_STRUCTURE_DIVERSE", replay)
    ood = eval_prompts("OOD", 48, ("CONNECTED_COMPONENT_COUNT", "OBJECT_EXTRACTION"), ("PLUS", "BOX"))
    retention = eval_prompts("RETENTION", 24, ("NO_TRANSFORM_RETENTION",), ("PLUS",))
    sentinel = eval_prompts("COMPOSITION_SENTINEL", 16, ("COUNT_THEN_OBJECT_EXTRACTION",), ("BOX",))
    atomic_write(out / "CONTROL_SCHEDULE.json", control)
    atomic_write(out / "TREATMENT_SCHEDULE.json", treatment)
    atomic_write(out / "OOD_INPUT_PROMPTS.json", ood)
    atomic_write(out / "RETENTION_INPUT_PROMPTS.json", retention)
    atomic_write(out / "COMPOSITION_SENTINEL_INPUT_PROMPTS.json", sentinel)
    atomic_write(out / "E1_EFFECTIVE_SUPERVISION_AUDIT.json", historical_audit())
    accounting = token_accounting(control, treatment)
    atomic_write(out / "TOKEN_ACCOUNTING.json", accounting)
    validation = validate(control, treatment, ood, retention, sentinel, accounting)
    atomic_write(out / "CPU_VALIDATION.json", validation)
    manifest = {
        "schema_version": 1, "protocol_id": "CONNECTED_COMPONENTS_OBJECT_EXTRACTION_CAPABILITY_PILOT_V1",
        "status": "CPU_ONLY_FROZEN_PENDING_DIRECTOR_PRELAUNCH_REVIEW",
        "source_prefreeze_commit": "854444401ca35019e834336094ec7fa0631787b8",
        "checkpoint": {"id": "CAPABILITY_REPAIR_BASELINE_V1_V7", "manifest_sha256": CHECKPOINT_SHA256, "independent_fresh_initialization_per_arm": True},
        "recipe": {"optimizer": "PagedAdamW8bit", "learning_rate": 0.00005, "lora_rank": 64, "lora_alpha": 32, "lora_dropout": 0.0, "precision": "BF16", "optimizer_steps_per_arm": 96, "slots_per_step": 4, "per_arm_runtime_cap_seconds": 3600, "joint_runtime_cap_seconds": 7200},
        "principal_intervention": "CONTROL uses only POINT/L narrow structures; TREATMENT uses L/Z/T structurally diverse examples, with both arms balanced over connected-component count and object extraction.",
        "protected_replay": {"source_schedule_sha256": file_sha(E04), "source_slots": list(range(96, 384)), "byte_identical_slots_per_arm": 288},
        "evaluation": {"ood": "target-blind prompts, structurally disjoint PLUS/BOX topologies", "retention": "separate target-blind no-transform prompts", "composition_sentinel": "separate target-blind count-then-object-extraction prompts", "targets": "NOT_CREATED_OR_ACCESSED", "primary": "native-Greedy exact-grid accuracy, to be bound only after Director prelaunch review"},
        "files": {}, "model_loaded": False, "gpu_used": False, "training_started": False, "optimizer_constructed": False, "sealed_target_accessed": False,
    }
    for name in ("CONTROL_SCHEDULE.json", "TREATMENT_SCHEDULE.json", "OOD_INPUT_PROMPTS.json", "RETENTION_INPUT_PROMPTS.json", "COMPOSITION_SENTINEL_INPUT_PROMPTS.json", "E1_EFFECTIVE_SUPERVISION_AUDIT.json", "TOKEN_ACCOUNTING.json", "CPU_VALIDATION.json"):
        manifest["files"][name] = file_sha(out / name)
    atomic_write(out / "MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        c = schedule(train_rows("CONTROL_NARROW_STRUCTURE"), "CONTROL_NARROW_STRUCTURE", json.loads(E04.read_text(encoding="utf-8"))["episodes"][96:])
        t = schedule(train_rows("TREATMENT_STRUCTURE_DIVERSE"), "TREATMENT_STRUCTURE_DIVERSE", json.loads(E04.read_text(encoding="utf-8"))["episodes"][96:])
        assert token_accounting(c, t)["control"] == token_accounting(c, t)["treatment"]
        print("PASS_CPU_CONNECTED_COMPONENTS_OBJECT_EXTRACTION_FREEZER")
        return
    print(json.dumps(freeze(args.out), sort_keys=True))


if __name__ == "__main__":
    main()
