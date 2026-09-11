"""One solution-blind V2 task execution shared by sequential and parallel runners."""
from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import numpy as np

from arc.task import ARCTask
from capabilities.execution import Status
from capabilities.pipeline import CapabilityProgram
from primitives.program import Step
from .macro_compiler_v1 import MacroProgramCompilerV1, MacroProgramExecutorV1
from .macro_dsl import MacroStatus, validate_macro_hypothesis
from .macro_generator_v2 import MacroHypothesisGeneratorV2


def serialized_program(program: CapabilityProgram | None) -> dict[str, Any] | None:
    if program is None:
        return None
    return {"steps": [{"primitive_id": step.primitive_id, "params": step.params} for step in program.steps], "provenance": program.provenance, "parameter_provenance": list(program.parameter_provenance)}


def execute_v2_task(task: ARCTask, generator: MacroHypothesisGeneratorV2, *, parameter_mode: str, worker_id: int | str | None = None, gpu_id: int | None = None) -> dict[str, Any]:
    """Generate, compile, hard-verify, and freeze one task without solutions."""
    if parameter_mode not in {"symbolic", "direct"}:
        raise ValueError(f"unsupported parameter mode: {parameter_mode}")
    started = perf_counter()
    compiler = MacroProgramCompilerV1(allow_direct_literals=parameter_mode == "direct")
    executor = MacroProgramExecutorV1()
    try:
        response = generator.generate(task, direct_parameter_mode=parameter_mode == "direct")
        hypotheses, raw = response.hypotheses, response.raw_response
        inference_seconds, prompt_tokens, output_tokens = response.elapsed_seconds, response.input_tokens, response.output_tokens
    except TimeoutError as exc:
        hypotheses, raw, inference_seconds, prompt_tokens, output_tokens = (), None, perf_counter() - started, 0, 0
        candidate_results = [{"hypothesis_id": f"timeout:{task.task_id}", "status": "TIMEOUT", "stage": "provider", "reason": str(exc), "macro_validation": None, "parameter_resolutions": [], "compiled_program": None, "executor_result": None, "verifier_result": None}]
    except Exception as exc:  # provider errors are per-task; never kill a worker
        hypotheses, raw, inference_seconds, prompt_tokens, output_tokens = (), None, perf_counter() - started, 0, 0
        candidate_results = [{"hypothesis_id": f"provider_error:{task.task_id}", "status": "PROVIDER_FAILED", "stage": "provider", "reason": f"provider_error:{type(exc).__name__}: {exc}", "macro_validation": None, "parameter_resolutions": [], "compiled_program": None, "executor_result": None, "verifier_result": None}]
    else:
        candidate_results = []
        for hypothesis in hypotheses:
            validation = validate_macro_hypothesis(hypothesis, allow_direct_literals=parameter_mode == "direct")
            item: dict[str, Any] = {"hypothesis_id": hypothesis.hypothesis_id, "status": validation.status.value, "stage": "macro_validation", "reason": validation.reason, "macro_validation": {"status": validation.status.value, "reason": validation.reason}, "parameter_resolutions": [], "compiled_program": None, "executor_result": None, "verifier_result": None}
            if validation.status == MacroStatus.COMPILED:
                compiled = compiler.compile(hypothesis, task)
                item["status"], item["stage"], item["reason"] = compiled.status.value, "compiler", compiled.reason
                if compiled.parameter_result is not None:
                    item["parameter_resolutions"] = [{"step_index": value.step_index, "parameter": value.parameter, "symbolic_source": value.symbolic_source, "concrete_value": value.concrete_value, "evidence": value.evidence, "train_pair_consistent": value.train_pair_consistent, "status": value.status.value} for value in compiled.parameter_result.resolutions]
                item["compiled_program"] = serialized_program(compiled.program)
                if compiled.program is not None:
                    execution = executor.execute(compiled.program, task.train[0].input.values)
                    item["executor_result"] = {"status": execution.status.value, "reason": execution.reason, "output_shape": list(execution.value.shape) if execution.status == Status.SUCCESS and isinstance(execution.value, np.ndarray) else None}
                    if execution.status == Status.SUCCESS:
                        verified = executor.verify_result(compiled.program, task)
                        item["verifier_result"] = {"status": verified.status.value, "reason": verified.reason}
                        item["status"], item["stage"], item["reason"] = verified.status.value, "hard_verifier", verified.reason
                    else:
                        item["status"], item["stage"], item["reason"] = execution.status.value, "executor", execution.reason
            candidate_results.append(item)
    valid = next((item for item in candidate_results if item["status"] == Status.TRAIN_CONSISTENT.value), None)
    prediction = None
    if valid and valid["compiled_program"]:
        program_data = valid["compiled_program"]
        program = CapabilityProgram(tuple(Step(step["primitive_id"], step["params"]) for step in program_data["steps"]), program_data["provenance"], tuple(program_data["parameter_provenance"]))
        rendered = [executor.execute(program, example.input.values) for example in task.test]
        if all(result.status == Status.SUCCESS for result in rendered):
            prediction = [np.asarray(result.value).astype(int).tolist() for result in rendered]
        else:
            valid["status"], valid["stage"], valid["reason"] = "EXECUTION_ERROR", "test_execution", "test execution failed"
    return {"task_id": task.task_id, "worker_id": worker_id, "gpu_id": gpu_id, "status": "SUCCESS", "raw_llm_response": raw, "macro_hypotheses": [hypothesis.to_dict() for hypothesis in hypotheses], "candidate_results": candidate_results, "prediction": prediction, "inference_seconds": inference_seconds, "total_task_seconds": perf_counter() - started, "prompt_tokens": prompt_tokens or 0, "output_tokens": output_tokens or 0, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
