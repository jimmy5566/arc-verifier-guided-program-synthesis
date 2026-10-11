"""Create one fresh, SHA-bound E04-C launch binding after Director approval."""
from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.e04_c_matched_rotation_execution import (ARMS, BASE, JOINT_CAP_SECONDS, PER_ARM_CAP_SECONDS,
    SCHEDULE, atomic_json, relative, sha_path, static_schedule_preflight)

RESPONSE = ROOT / "orchestration/director/responses/E04_C_RUN_002_REPLACEMENT_EXECUTION_AUTHORIZATION_BRIEF_V1_RESPONSE.json"
WORKER = ROOT / "scripts/run_e04_c_matched_rotation_repair_pilot.py"
EVALUATOR = ROOT / "scripts/score_e04_c_matched_rotation_repair_pilot.py"
LAUNCHER = ROOT / "scripts/launch_e04_c_matched_rotation_repair_pilot.py"
V7_REFERENCE = ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/e04_v3_full768_b1_no_update_remeasurement_v1/run_001_2ee668738df64d558f666f5989d09add/PRIMARY_B1_RAW.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root-prefix", default="/workspace/arc2/e04_c_matched_fixed_turn_rotation_repair_pilot_v1")
    parser.add_argument("--worker-source-commit")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("E04C_BINDING_REFUSE_OVERWRITE")
    proof = static_schedule_preflight()
    response = json.loads(RESPONSE.read_text(encoding="utf-8"))
    if response.get("decision") != "CONTINUE_CONTROLLER":
        raise SystemExit("E04C_DIRECTOR_DECISION_NOT_AUTHORIZING")
    auth = response.get("replacement_execution_authorization", {})
    if not (auth.get("gpu_training_authorized") and auth.get("post_training_b1_generation_authorized") and auth.get("post_freeze_cpu_scoring_authorized")):
        raise SystemExit("E04C_DIRECTOR_EXECUTION_SCOPE")
    worker_source_commit = args.worker_source_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    nonce = secrets.token_hex(16)
    run_root = args.run_root_prefix.rstrip("/") + "/run_001_" + nonce
    protocol = BASE / "E04_C_PRELAUNCH_PROTOCOL_V1.json"
    control = BASE / "CONTROL_NO_TRANSFORM_TRAIN.jsonl"
    treatment = BASE / "TREATMENT_FIXED_TURN_ROTATION_TRAIN.jsonl"
    binding = {
        "schema_version": 1,
        "protocol_id": proof["protocol_id"],
        "authorization_id": response.get("response_id", "E04_C_REPLACEMENT_EXECUTION_AUTHORIZATION_BRIEF_V1_RESPONSE"),
        "director_response_path": relative(RESPONSE), "director_response_sha256": sha_path(RESPONSE),
        # The binding is committed after this worker snapshot.  A launcher
        # supplies its exact binding-containing launch commit at runtime, so
        # no field needs to cryptographically name its own future Git commit.
        "execution_authorized": True, "worker_source_commit": worker_source_commit,
        "worker_path": relative(WORKER), "worker_sha256": sha_path(WORKER),
        "evaluator_path": relative(EVALUATOR), "evaluator_sha256": sha_path(EVALUATOR),
        "launcher_path": relative(LAUNCHER), "launcher_sha256": sha_path(LAUNCHER),
        "protocol_path": relative(protocol), "protocol_sha256": sha_path(protocol),
        "schedule_binding_path": relative(SCHEDULE / "MATCHED_SCHEDULE_BINDING_V1.json"), "schedule_binding_sha256": sha_path(SCHEDULE / "MATCHED_SCHEDULE_BINDING_V1.json"),
        "static_preflight_path": relative(SCHEDULE / "NATIVE_TOKEN_STATIC_PREFLIGHT_V1.json"), "static_preflight_sha256": sha_path(SCHEDULE / "NATIVE_TOKEN_STATIC_PREFLIGHT_V1.json"),
        "control_cohort_path": relative(control), "control_cohort_sha256": sha_path(control),
        "treatment_cohort_path": relative(treatment), "treatment_cohort_sha256": sha_path(treatment),
        "checkpoint_manifest_path": "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json", "checkpoint_manifest_sha256": sha_path(ROOT / "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"),
        "runtime_config_path": "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_FULL768_B1_NO_UPDATE_REMEASUREMENT_V1_CONFIG.json", "runtime_config_sha256": sha_path(ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_FULL768_B1_NO_UPDATE_REMEASUREMENT_V1_CONFIG.json"),
        "v7_reference_raw_path": relative(V7_REFERENCE), "v7_reference_raw_sha256": sha_path(V7_REFERENCE),
        "arm_order": list(ARMS), "seed": 2000031, "arms": 2, "jobs": 1, "retry": False,
        "nonce": nonce, "output_root": run_root,
        "arm_output_roots": {arm: run_root + "/arms/" + arm for arm in ARMS},
        "per_arm_runtime_cap_seconds": PER_ARM_CAP_SECONDS, "joint_runtime_cap_seconds": JOINT_CAP_SECONDS,
        "static_preflight": proof,
    }
    atomic_json(args.output, binding)
    print(json.dumps({"status": "FRESH_BINDING_CREATED", "binding_sha256": sha_path(args.output), "nonce": nonce, "worker_source_commit": worker_source_commit}, sort_keys=True))


if __name__ == "__main__":
    main()
