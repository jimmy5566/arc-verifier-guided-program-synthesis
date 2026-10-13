"""Compatibility closure for the frozen code-pair raw generation.

The code-pair binding binds the common deterministic scorer by exact source
hash, but stores its dataset and construct identities under the code-pair
schema.  This adapter projects only those already-bound values into the
scorer's schema.  It validates the whole original binding and all frozen raw
bytes before the common scorer is allowed to open target context.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
COMMON_SCORER = "scripts/score_connected_components_object_extraction_pilot_v1.py"
CONSTRUCT = f"{BASE}/scientific_successor_v2/SCIENTIFIC_CONSTRUCT_VALIDATION.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(canon(value) + b"\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project(binding_path: Path) -> tuple[dict, dict]:
    original = json.loads(binding_path.read_text())
    preclaim = load("scripts/run_connected_component_count_decision_token_supervision_pilot_v1.py", "code_pair_preclaim_for_score")
    preclaim.validate_binding(original)
    try:
        scorer_hash = original["source_files"][COMMON_SCORER]
        paths = original["inputs"]
        projected = dict(original)
        projected.update({
            "score_entrypoint_path": COMMON_SCORER,
            "score_entrypoint_sha256": scorer_hash,
            "scientific_construct_validation_path": CONSTRUCT,
            "scientific_construct_validation_sha256": sha(ROOT / CONSTRUCT),
            "dataset_paths": {
                "OOD": paths["ood_prompts"]["path"],
                "RETENTION": paths["retention_prompts"]["path"],
                "COMPOSITION": paths["composition_prompts"]["path"],
            },
        })
    except KeyError as exc:
        raise RuntimeError(f"CODE_PAIR_SCORE_COMPATIBILITY:MISSING_BOUND_FIELD:{exc.args[0]}") from exc
    if projected["scientific_construct_validation_sha256"] != "622a91fcfe07c40b4a2c7c58bd25e423d0a1d83aeaf9e7a8b7068e6b54eb50d7":
        raise RuntimeError("CODE_PAIR_SCORE_COMPATIBILITY:CONSTRUCT_IDENTITY")
    meta = {
        "status": "POSTFREEZE_SCORE_COMPATIBILITY_PROJECTED",
        "original_binding_sha256": sha(binding_path),
        "common_scorer_path": COMMON_SCORER,
        "common_scorer_sha256": scorer_hash,
        "compatibility_entrypoint_path": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "compatibility_entrypoint_sha256": sha(Path(__file__)),
        "scientific_construct_validation_path": CONSTRUCT,
        "scientific_construct_validation_sha256": projected["scientific_construct_validation_sha256"],
        "dataset_paths": projected["dataset_paths"],
        "targets_opened": False,
        "model_loaded": False,
        "gpu_used": False,
    }
    return projected, meta


def validate_projection(binding_path: Path, output: Path) -> tuple[dict, dict]:
    projected, meta = project(binding_path)
    scorer = load(COMMON_SCORER, "common_scorer_for_code_pair")
    freeze, _ = scorer._load_raw(projected, output)
    meta["raw_freeze_sha256"] = sha(output / "RAW_FREEZE_RECEIPT.json")
    meta["raw_sha256"] = freeze["raw_sha256"]
    write(output / "POSTFREEZE_SCORE_COMPATIBILITY_RECEIPT.json", meta)
    return projected, meta


def score(binding_path: Path, output: Path) -> dict:
    projected, meta = validate_projection(binding_path, output)
    scorer = load(COMMON_SCORER, "common_scorer_for_code_pair_execute")
    temporary = output / ".POSTFREEZE_SCORE_COMPATIBILITY_BINDING.json"
    write(temporary, projected)
    try:
        result = scorer.score(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result["code_pair_score_compatibility"] = meta
    result["original_binding_sha256"] = meta["original_binding_sha256"]
    write(output / "SCORE_RECEIPT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-projection-only", action="store_true")
    args = parser.parse_args()
    if args.validate_projection_only:
        _, receipt = validate_projection(args.binding, args.output_root)
        print(json.dumps(receipt, sort_keys=True))
    else:
        print(json.dumps(score(args.binding, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
