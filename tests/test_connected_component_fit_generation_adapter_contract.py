import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.connected_component_fit_generation_runtime_contract import resolve_adapter_directory


class AdapterContractTest(unittest.TestCase):
    def runtime(self, root: Path):
        weights = root / "adapter_model.safetensors"
        config = root / "adapter_config.json"
        weights.write_bytes(b"weights")
        config.write_bytes(b'{"peft_type":"LORA"}\n')
        return {
            "adapter_path": str(weights),
            "adapter_model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "adapter_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        }

    def test_weight_file_identity_resolves_peft_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(resolve_adapter_directory(self.runtime(root)), root)

    def test_missing_or_unbound_config_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self.runtime(root)
            runtime.pop("adapter_config_sha256")
            with self.assertRaisesRegex(RuntimeError, "ADAPTER_CONFIG_IDENTITY_UNBOUND"):
                resolve_adapter_directory(runtime)
            runtime = self.runtime(root)
            (root / "adapter_config.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "ADAPTER_CONFIG_MISSING"):
                resolve_adapter_directory(runtime)
