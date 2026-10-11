"""Validate frozen E05 inputs without opening or scoring the target sidecar."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e05_structurally_held_out_marker_conditioned_rotation_confirmation_v1"
CONDS = {"FIXED_TURN_ROTATION_CONTROL", "MARKER_BINDING_CONTROL", "NO_TRANSFORM_RETENTION_CONTROL", "MARKER_CONDITIONED_ROTATION"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    prompts = [json.loads(line) for line in (OUT / "INPUT_PROMPTS.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(prompts) != 768 or any("target" in row for row in prompts):
        raise RuntimeError("E05_INPUT_TARGET_ISOLATION")
    if Counter(row["condition"] for row in prompts) != Counter({name: 192 for name in CONDS}):
        raise RuntimeError("E05_CONDITION_ALLOCATION")
    by_base = {}
    for row in prompts:
        by_base.setdefault(row["canonical_base_id"], []).append(row)
    if len(by_base) != 48 or any(len(rows) != 16 or {row["turn"] for row in rows} != {0, 1, 2, 3} or {row["condition"] for row in rows} != CONDS for rows in by_base.values()):
        raise RuntimeError("E05_CANONICAL_BASE_COVERAGE")
    if digest(OUT / "INPUT_PROMPTS.jsonl") != manifest["prompt_sha256"]:
        raise RuntimeError("E05_PROMPT_SHA")
    audit = json.loads((OUT / "STRUCTURAL_DISJOINTNESS_AUDIT.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["source"]["sealed_target_sidecar_opened"] is not False:
        raise RuntimeError("E05_DISJOINTNESS_AUDIT")
    result = {"status": "PASS_CPU_ONLY_INPUT_FREEZE_VALIDATION", "input_prompts": len(prompts), "canonical_bases": len(by_base), "target_sidecar_opened": False, "model_loaded": False, "gpu_used": False, "prompt_sha256": manifest["prompt_sha256"], "structural_audit_sha256": manifest["structural_audit_sha256"]}
    (OUT / "CPU_FREEZE_VALIDATION.json").write_bytes(json.dumps(result, sort_keys=True, indent=2).encode() + b"\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
