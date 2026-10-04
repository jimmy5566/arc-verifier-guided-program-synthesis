import hashlib
import json
import tempfile
from pathlib import Path

from scripts.audit_search_order_adapters import audit_adapter_state


def _adapter(path: Path, weights: bytes) -> dict:
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(weights)
    config = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 256, "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return {
        "adapter_sha256": hashlib.sha256(weights).hexdigest(),
        "adapter_config_sha256": hashlib.sha256((path / "adapter_config.json").read_bytes()).hexdigest(),
        "adapter_config_semantics": {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 256, "lora_alpha": 32,
                                     "target_modules": sorted(config["target_modules"])},
        "status": "PASS",
    }


def test_adapter_audit_distinguishes_exact_and_mismatch() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "adapter"
        expected = _adapter(path, b"exact weights")
        cohort = {"cohort_sha256": "cohort", "outputs": [{"task_id": "abc", "output_id": "abc:o0", "adapter_path": str(path), "adapter_identity": expected}]}
        reader = lambda _path: expected
        assert audit_adapter_state(cohort, identity_reader=reader)["adapter_state"] == "EXACT_HISTORICAL"
        cohort["outputs"][0]["adapter_identity"] = {**expected, "adapter_sha256": "not-the-real-hash"}
        assert audit_adapter_state(cohort, identity_reader=reader)["adapter_state"] == "MISSING_OR_MISMATCH"
