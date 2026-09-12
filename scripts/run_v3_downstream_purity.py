"""Run only Validator → Binder → Executor → HardVerifier over frozen RuleSpecs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from v3.binding import BindingError, InstanceBinder
from v3.execution.rule_executor import RuleExecutor
from v3.recognition.recognizer_interface import parse_complete_rulespec_hypotheses
from v3.validation import RuleSpecPreflightValidator
from v3.verification.verifier import HardVerifier


def _spec_from_manifest(value: dict[str, Any]) -> Any:
    skeleton = value["skeleton"]
    hypothesis = {
        "family": skeleton["family"],
        "operations": [step["operation"] for step in skeleton["steps"]],
        "parameters": value["rule_parameters"],
        "roles": value["role_selectors"],
        "repeat": value["repeat_semantics"],
    }
    parsed, status = parse_complete_rulespec_hypotheses(json.dumps({"hypotheses": [hypothesis]}), limit=1)
    if status != "SUCCESS": raise ValueError(f"invalid frozen complete RuleSpec: {status}")
    return parsed[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite purity result")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")); tasks = load_dataset(args.challenge_path)
    validator, binder, executor, verifier = RuleSpecPreflightValidator(), InstanceBinder(), RuleExecutor(), HardVerifier()
    records: dict[str, dict[str, object]] = {}
    for case in manifest["cases"]:
        task_id, spec, task = case["task_id"], _spec_from_manifest(case["rule_spec"]), tasks[case["task_id"]]
        train = tuple((item.input.values, item.output.values) for item in task.train)
        preflight = validator.validate(spec)
        binding_ok = execution_ok = False
        if preflight.passed:
            try:
                bound = [binder.bind(spec, source) for source, _target in train]
                binding_ok = True
                actual = [executor.execute_bound(item, source) for item, (source, _target) in zip(bound, train)]
                execution_ok = True
            except (BindingError, TypeError, ValueError):
                actual = []
        exact = execution_ok and verifier.verify(spec, train).passed
        records[task_id] = {"preflight": preflight.passed, "binding": binding_ok, "execution": execution_ok, "exact": exact}
    result = {
        "experiment_id": "ARC2_V3_REPEAT_SEMANTICS_EXPANSION",
        "status": "TRUE_DOWNSTREAM_PURITY_COMPLETE",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "protocol": "Only frozen complete RuleSpecs enter this runner. It performs no evidence extraction, parameter inference, candidate generation, model call, test output or solution access.",
        "denominator": len(records),
        "coverage": {name: sum(bool(item[name]) for item in records.values()) for name in ("preflight", "binding", "execution", "exact")},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"denominator": result["denominator"], "coverage": result["coverage"]}, sort_keys=True))


if __name__ == "__main__": main()
