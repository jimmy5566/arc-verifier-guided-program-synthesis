"""Read-only validation for the frozen connected-components/object-extraction pilot."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
FREEZER = ROOT / "scripts/freeze_connected_components_object_extraction_capability_pilot_v1.py"


def read(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    spec = importlib.util.spec_from_file_location("cc_freezer", FREEZER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    control = read("CONTROL_SCHEDULE.json")
    treatment = read("TREATMENT_SCHEDULE.json")
    ood = read("OOD_INPUT_PROMPTS.json")
    retention = read("RETENTION_INPUT_PROMPTS.json")
    sentinel = read("COMPOSITION_SENTINEL_INPUT_PROMPTS.json")
    accounting = read("TOKEN_ACCOUNTING.json")
    result = module.validate(control, treatment, ood, retention, sentinel, accounting)
    manifest = read("MANIFEST.json")
    for name, expected in manifest["files"].items():
        if sha(OUT / name) != expected:
            raise RuntimeError(f"MANIFEST_HASH_MISMATCH:{name}")
    if manifest["evaluation"]["targets"] != "NOT_CREATED_OR_ACCESSED":
        raise RuntimeError("TARGET_BOUNDARY")
    if manifest["checkpoint"]["manifest_sha256"] != module.CHECKPOINT_SHA256:
        raise RuntimeError("CHECKPOINT_IDENTITY")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
