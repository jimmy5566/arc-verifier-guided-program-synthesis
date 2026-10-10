import hashlib
import json
import math
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn.functional as torch_f
    from torch.utils.checkpoint import checkpoint
except ModuleNotFoundError:
    torch = None
    torch_f = None
    checkpoint = None

ROOT = Path(__file__).resolve().parents[1]
E03 = ROOT / 'experiments/capability_repair_baseline_v1/e03_v7_lora_gradient_interference_diagnostic_v1'
WORKER = ROOT / 'scripts/run_e03_v7_lora_gradient_interference_v2_lower_memory.py'
LAUNCHER = ROOT / 'scripts/launch_e03_v7_lora_gradient_interference_v2_lower_memory.py'
CONFIG = E03 / 'E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V2_LOWER_MEMORY_CONFIG_V1.json'
BINDING = E03 / 'E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V2_LOWER_MEMORY_PRELAUNCH_BINDING_V1.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))

def ce_sum(logits, labels, ignore=-100):
    value = 0.0
    gradient = [[0.0 for _ in row] for row in logits]
    for row_index, (row, label) in enumerate(zip(logits, labels)):
        if label == ignore:
            continue
        denominator = logsumexp(row)
        value += denominator - row[label]
        for column, logit in enumerate(row):
            gradient[row_index][column] = math.exp(logit - denominator) - (1.0 if column == label else 0.0)
    return value, gradient

class E03V2LowerMemoryTests(unittest.TestCase):
    PYTORCH_CHECKPOINT_ATOL = 1e-7
    PYTORCH_CHECKPOINT_RTOL = 1e-6

    def test_static_fail_closed_contract(self):
        source = WORKER.read_text(encoding='utf-8')
        launcher = LAUNCHER.read_text(encoding='utf-8')
        for required in (
            'gradient_checkpointing_enable', "'use_reentrant':False", 'model.config.use_cache=False',
            'E03_GRADIENT_CHECKPOINTING_UNSUPPORTED', 'shifted_selected_ce',
            'selected_logits=shift_logits[selected].float()', 'E03_USE_CACHE_MUST_BE_FALSE',
            'peak_allocated_bytes', 'peak_reserved_bytes', 'E03_FRESH_OUTPUT_REQUIRED',
            'E03_EXTERNAL_CAP_LAUNCHER_REQUIRED',
        ):
            self.assertIn(required, source)
        for forbidden in ('torch.optim', '.generate(', 'load_in_4bit', 'offload', 'fallback'):
            self.assertNotIn(forbidden, source)
        self.assertIn('x.output_root.parent.mkdir(parents=True,exist_ok=True)', launcher)
        self.assertIn('os.killpg', launcher)
        self.assertIn("'retry':False", launcher)

    def test_selected_shifted_ce_reconstructs_legacy_loss_and_lora_like_gradient(self):
        # The second path recomputes the toy decoder activations, modelling checkpoint recomputation.
        ordinary = [[0.2, -0.3, 0.7], [1.1, 0.4, -0.2], [-0.5, 0.9, 0.1], [0.3, 0.6, -0.7]]
        recomputed = [[float(value) for value in row] for row in ordinary]
        shifted_labels = [2, -100, 1, 0]
        legacy_loss, legacy_gradient = ce_sum(ordinary, shifted_labels)
        selected_indices = [index for index, label in enumerate(shifted_labels) if label != -100]
        selected_loss, selected_gradient = ce_sum(
            [recomputed[index] for index in selected_indices],
            [shifted_labels[index] for index in selected_indices],
        )
        self.assertEqual(selected_indices, [0, 2, 3])
        self.assertAlmostEqual(legacy_loss, selected_loss, places=12)
        reconstructed = [[0.0 for _ in row] for row in ordinary]
        for source, target in enumerate(selected_indices):
            reconstructed[target] = selected_gradient[source]
        for expected, actual in zip(legacy_gradient, reconstructed):
            for expected_value, actual_value in zip(expected, actual):
                self.assertAlmostEqual(expected_value, actual_value, places=12)
        features = [[1.0, 0.5], [0.2, -0.1], [0.4, 0.3], [-0.3, 0.6]]
        def lora_like_gradient(logit_gradient):
            return [[sum(features[row][feature] * logit_gradient[row][column] for row in range(4)) for column in range(3)] for feature in range(2)]
        for expected, actual in zip(lora_like_gradient(legacy_gradient), lora_like_gradient(reconstructed)):
            for expected_value, actual_value in zip(expected, actual):
                self.assertAlmostEqual(expected_value, actual_value, places=12)

    @unittest.skipIf(torch is None, 'PyTorch is unavailable in this local CPU environment')
    def test_pytorch_nonreentrant_checkpoint_matches_ordinary_selected_ce_and_autograd(self):
        self.assertFalse(torch.cuda.is_initialized())

        class TinyLoRALike(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.5, -0.2]]), requires_grad=False)
                self.lora_a = torch.nn.Parameter(torch.tensor([[0.1, -0.4], [0.3, 0.2]]))
                self.lora_b = torch.nn.Parameter(torch.tensor([[0.2, -0.3, 0.1], [-0.5, 0.4, 0.6]]))

            def forward(self, inputs):
                return inputs @ self.base + (inputs @ self.lora_a) @ self.lora_b

        ordinary = TinyLoRALike()
        checkpointed = TinyLoRALike()
        checkpointed.load_state_dict(ordinary.state_dict())
        inputs = torch.tensor([[[0.1, 0.3], [0.2, -0.1], [0.5, 0.4], [-0.2, 0.6]], [[0.4, 0.1], [0.3, 0.2], [-0.4, 0.5], [0.7, -0.3]]], requires_grad=True)
        labels = torch.tensor([[-100, 2, 1, 0], [-100, 1, -100, 2]])

        def selected_loss(logits):
            shifted_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            selected = shifted_labels.ne(-100)
            loss = torch_f.cross_entropy(shifted_logits[selected].float(), shifted_labels[selected], reduction='sum')
            return loss / selected.sum(), selected

        ordinary_loss, ordinary_selected = selected_loss(ordinary(inputs))
        ordinary_loss.backward()
        checkpoint_inputs = inputs.detach().clone().requires_grad_(True)
        checkpointed_logits = checkpoint(lambda value: checkpointed(value), checkpoint_inputs, use_reentrant=False)
        checkpointed_loss, checkpointed_selected = selected_loss(checkpointed_logits)
        checkpointed_loss.backward()
        self.assertTrue(torch.equal(ordinary_selected, checkpointed_selected))
        torch.testing.assert_close(ordinary_loss, checkpointed_loss, atol=self.PYTORCH_CHECKPOINT_ATOL, rtol=self.PYTORCH_CHECKPOINT_RTOL)
        self.assertEqual(
            [name for name, parameter in ordinary.named_parameters() if parameter.requires_grad],
            [name for name, parameter in checkpointed.named_parameters() if parameter.requires_grad],
        )
        for (_, expected), (_, actual) in zip(ordinary.named_parameters(), checkpointed.named_parameters()):
            if expected.requires_grad:
                torch.testing.assert_close(expected.grad, actual.grad, atol=self.PYTORCH_CHECKPOINT_ATOL, rtol=self.PYTORCH_CHECKPOINT_RTOL)
        self.assertFalse(torch.cuda.is_initialized())

    def test_frozen_identity_and_no_fallback(self):
        config = json.loads(CONFIG.read_text(encoding='utf-8'))
        binding = json.loads(BINDING.read_text(encoding='utf-8'))
        self.assertFalse(config['execution_authorized'])
        self.assertFalse(binding['execution_authorized'])
        self.assertEqual(config['protocol_id'], 'E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V2_LOWER_MEMORY')
        self.assertEqual(config['numerics']['primary_microbatch_size'], 8)
        self.assertEqual(config['numerics']['batch1_sensitivity_rows'], 72)
        self.assertEqual(config['manifest_sha256'], '901ba1aaf692ebaa004ce69b5997064b0b52515e49eacc291762c40ff15a86be')
        self.assertFalse(config['lower_memory_v2']['scientific_estimand_change'])
        self.assertEqual(binding['runtime_cap_seconds'], 1500)
        self.assertEqual(binding['jobs'], 1)
        self.assertFalse(binding['retry'])
        required = ('B4', 'B2', 'B1 primary', 'serial gradient accumulation', 'quantization', 'CPU/NVMe offload', 'different precision', 'different objective')
        self.assertTrue(all(item in binding['prohibited_fallbacks'] for item in required))
        self.assertEqual(binding['bound_files'][str(CONFIG.relative_to(ROOT)).replace('\\', '/')], sha(CONFIG))
        self.assertEqual(binding['bound_files'][str(WORKER.relative_to(ROOT)).replace('\\', '/')], sha(WORKER))
        self.assertEqual(binding['bound_files'][str(LAUNCHER.relative_to(ROOT)).replace('\\', '/')], sha(LAUNCHER))

if __name__ == '__main__':
    unittest.main()
