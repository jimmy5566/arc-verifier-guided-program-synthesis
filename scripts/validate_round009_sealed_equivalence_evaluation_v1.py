"""CPU-only structural validation for the frozen sealed-evaluation package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "foundation_v2_reconstruction_and_targeted_repair_v2" / "round_009_equivalence_evaluation"


def load(name: str) -> dict:
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def sha(name: str) -> str:
    return hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()


def main() -> int:
    protocol = load("ROUND_009_SEALED_EQUIVALENCE_EVALUATION_V1_PROTOCOL.json")
    audit = load("ROUND_009_SEALED_REFERENCE_PROVENANCE_AUDIT.json")
    phase_a = load("ROUND_009_SEALED_EQUIVALENCE_PHASE_A_CONTRACT.json")
    phase_b = load("ROUND_009_SEALED_EQUIVALENCE_PHASE_B_SCORER_CONTRACT.json")
    metrics = load("ROUND_009_SEALED_EQUIVALENCE_METRIC_SEMANTICS.json")
    nonces = load("ROUND_009_SEALED_EQUIVALENCE_EVALUATION_V1_NONCES.json")
    assert protocol["protocol_id"] == "ROUND_009_SEALED_EQUIVALENCE_EVALUATION_V1"
    assert protocol["execution_authorization"] == {"model_loading": False, "generation": False, "scoring": False, "gpu_inference": False, "scientific_training": False, "final_audit": False}
    assert audit["conclusion"] == "FAIL_CLOSED_REFERENCE_PROVENANCE_UNAVAILABLE"
    assert all(value == "MISSING" for value in audit["canonical_reference_mapping"].values())
    assert audit["canonical_baseline_outputs"] == "MISSING"
    assert audit["major_regression_definition"] == "MISSING"
    assert phase_a["execution_authorized"] is False and phase_b["execution_authorized"] is False
    assert phase_a["nonce"] != phase_b["nonce"] == nonces["phase_b_nonce"]
    assert phase_a["nonce"] == nonces["phase_a_nonce"]
    assert phase_a["output_root"] != phase_b["output_root"]
    assert phase_b["model_loading_forbidden"] is True and phase_b["phase_a_write_forbidden"] is True
    assert metrics["surfaces"]["NOVEL"]["minimum_exact_rate"] == 0.80
    assert metrics["surfaces"]["REPLAY"]["minimum_exact_rate"] == 0.78
    assert metrics["surfaces"]["PROTECTED_CAPABILITY"]["maximum_major_regressions"] == 0
    for manifest in protocol["input_manifests"].values():
        path = ROOT / manifest["path"]
        raw = path.read_text(encoding="utf-8")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
        assert '"target"' not in raw.lower() and '"output"' not in raw.lower()
    print(json.dumps({"CPU_ONLY_SEALED_EVALUATION_PREFLIGHT": "PASS_TO_FAIL_CLOSED_BOUNDARY", "execution_authorized": False, "reference_provenance": "MISSING", "package_hashes": {name: sha(name) for name in ("ROUND_009_SEALED_EQUIVALENCE_EVALUATION_V1_PROTOCOL.json", "ROUND_009_SEALED_REFERENCE_PROVENANCE_AUDIT.json", "ROUND_009_SEALED_EQUIVALENCE_PHASE_A_CONTRACT.json", "ROUND_009_SEALED_EQUIVALENCE_PHASE_B_SCORER_CONTRACT.json", "ROUND_009_SEALED_EQUIVALENCE_METRIC_SEMANTICS.json")}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
