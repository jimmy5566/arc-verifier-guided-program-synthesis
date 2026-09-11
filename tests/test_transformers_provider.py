from __future__ import annotations

import json

from arc.task import ARCExample, ARCGrid, ARCTask
from llm.macro_generator_v2 import TransformersMacroHypothesisGeneratorV2
from llm.models import GenerationConfig
from llm.providers import discover_providers
from llm.transformers_provider import TextGeneration, TransformersProvider


def _write_config(tmp_path, architectures: list[str]) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": architectures}), encoding="utf-8")


def test_transformers_provider_requires_local_qwen3_config_and_runtime_packages(tmp_path, monkeypatch):
    _write_config(tmp_path, ["Qwen3ForCausalLM"])
    monkeypatch.setattr("llm.transformers_provider.importlib.util.find_spec", lambda _: None)
    unavailable = TransformersProvider(model_path=tmp_path).availability()
    assert not unavailable.available
    assert "torch and transformers" in unavailable.reason


def test_transformers_provider_rejects_wrong_local_architecture(tmp_path, monkeypatch):
    _write_config(tmp_path, ["OtherModel"])
    monkeypatch.setattr("llm.transformers_provider.importlib.util.find_spec", lambda _: object())
    unavailable = TransformersProvider(model_path=tmp_path).availability()
    assert not unavailable.available
    assert "unexpected local architecture" in unavailable.reason


def test_transformers_provider_discovers_only_an_explicit_offline_model_path(tmp_path, monkeypatch):
    _write_config(tmp_path, ["Qwen3ForCausalLM"])
    monkeypatch.setattr("llm.transformers_provider.importlib.util.find_spec", lambda _: object())
    monkeypatch.setenv("ARC2_TRANSFORMERS_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("ARC2_TRANSFORMERS_DEVICE", "cuda:0")
    providers = discover_providers()
    assert len(providers) == 1
    assert isinstance(providers[0], TransformersProvider)
    assert providers[0].availability().available


def test_transformers_macro_generator_preserves_existing_macro_prompt_and_parser():
    class FakeProvider:
        def generate_text(self, prompt, config):
            self.prompt = json.loads(prompt)
            self.config = config
            return TextGeneration('{"hypotheses":[]}', 0.25, 12, 3)

    fake = FakeProvider()
    config = GenerationConfig("qwen3-8b", hypothesis_budget=5)
    task = ARCTask("fixture", (ARCExample(ARCGrid([[0]]), ARCGrid([[0]])),), (ARCExample(ARCGrid([[0]])),))
    result = TransformersMacroHypothesisGeneratorV2(fake, config).generate(task)
    assert result.hypotheses == ()
    assert result.raw_response == '{"hypotheses":[]}'
    assert result.input_tokens == 12 and result.output_tokens == 3
    # The frozen V2 task context intentionally contains only ARC examples;
    # it must not leak a task identifier into the model prompt.
    assert "task_id" not in fake.prompt["task"]
    assert fake.prompt["task"]["train"][0]["input"] == [[0]]
    assert fake.prompt["task"]["train"][0]["output"] == [[0]]
    assert "macro_program_schema" in fake.prompt
    assert "REG_FILL_INTERIOR_V1" not in json.dumps(fake.prompt["macro_program_schema"])
