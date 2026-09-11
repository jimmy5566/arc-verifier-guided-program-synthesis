import json
import importlib.util
from pathlib import Path

from inference.macro_api_benchmark_runner import summarize
from llm.macro_api_benchmark import CATEGORIES, benchmark_definition, build_cases, compile_canonical, model_prompt, registry_inventory, score_response


def test_benchmark_has_sixty_cases_with_fixed_category_counts():
    definition = benchmark_definition()
    assert definition["case_count"] == 60
    assert definition["category_counts"] == {category: 15 for category in CATEGORIES}


def test_every_canonical_program_is_schema_type_and_compile_valid():
    for case in build_cases():
        ok, reason = compile_canonical(case)
        assert ok, (case.case_id, reason)


def test_every_canonical_program_scores_compile_valid_without_arc_execution():
    for case in benchmark_definition()["cases"]:
        score = score_response(case, json.dumps(case["canonical_program"]))
        assert score["schema_valid"] and score["type_valid"] and score["compile_valid"]


def test_benchmark_hash_is_stable_and_registry_derived():
    first, second = benchmark_definition(), benchmark_definition()
    assert first["benchmark_hash"] == second["benchmark_hash"]
    assert first["registry_hash"] == registry_inventory()["registry_hash"]


def test_frozen_config_matches_benchmark_definition():
    frozen = json.loads((Path(__file__).parents[1] / "configs" / "MACRO_API_COMPREHENSION_BENCHMARK_V1_FROZEN_CONFIG.json").read_text())
    definition = benchmark_definition()
    for field in ("benchmark_hash", "registry_hash", "schema_hash"):
        assert frozen[field] == definition[field]
    assert frozen["generation"]["candidate_budget"] == 1


def test_benchmark_contains_no_arc_data_solutions_or_task_ids():
    definition = benchmark_definition()
    rendered = json.dumps(definition["cases"], sort_keys=True)
    def contains_grid(value):
        if isinstance(value, list) and value and all(isinstance(row, list) and row and all(isinstance(cell, int) for cell in row) for row in value):
            return True
        if isinstance(value, dict):
            return any(contains_grid(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_grid(item) for item in value)
        return False
    assert definition["no_arc_data_used"] and definition["no_arc_solutions_used"]
    assert not contains_grid(definition["cases"])
    assert "solution" not in rendered.lower()
    assert "task_id" not in rendered


def test_canonical_program_never_enters_model_prompt():
    case = benchmark_definition()["cases"][0]
    prompt = model_prompt(case)
    assert json.dumps(case["canonical_program"], sort_keys=True) not in prompt
    assert "canonical_program" not in prompt
    assert case["natural_language_instruction"] in prompt


def test_scoring_accepts_compile_valid_alternate_even_without_canonical_sequence_match():
    case = benchmark_definition()["cases"][0]
    alternate = {"hypotheses": [{"hypothesis_id": "alternate", "steps": [{"macro_id": "COMPLETE_SYMMETRY", "params": {"axis": {"literal": "HORIZONTAL"}}}]}]}
    score = score_response(case, json.dumps(alternate))
    assert score["compile_valid"]
    assert not score["canonical_macro_sequence_match"]
    assert not score["instruction_constraints_valid"]


def test_failure_taxonomy_distinguishes_json_and_contract_failures():
    case = benchmark_definition()["cases"][0]
    malformed = score_response(case, '{"hypotheses": [}')
    unknown = score_response(case, json.dumps({"hypotheses": [{"hypothesis_id": "x", "steps": [{"macro_id": "NOT_A_MACRO", "params": {}}]}]}))
    assert malformed["failure_type"] == "JSON_PARSE_FAILURE/MALFORMED_JSON"
    assert unknown["failure_type"] == "MACRO_API_FAILURE/UNKNOWN_MACRO_ID"


def test_inventory_is_machine_readable_and_has_all_registry_macros():
    inventory = registry_inventory()
    assert inventory["macro_count"] == len(inventory["macros"])
    assert {item["macro_id"] for item in inventory["macros"]}


def test_no_llm_judge_is_used_by_scoring_source():
    import inspect
    import llm.macro_api_benchmark as module
    source = inspect.getsource(module.score_response).lower()
    assert "llm judge" not in source and "judge(" not in source


def test_runner_summary_applies_no_go_threshold_without_arc_data():
    benchmark = benchmark_definition()
    record = {"case_id": benchmark["cases"][0]["case_id"], "score": {"response_received": True, "json_parseable": True, "schema_valid": True, "macro_ids_valid": True, "argument_contract_valid": True, "type_valid": False, "parameter_valid": False, "composition_valid": False, "compile_valid": False, "canonical_macro_sequence_match": False, "instruction_constraints_valid": False, "failure_type": "TYPE_FAILURE/MACRO_INPUT_OUTPUT_CHAIN_MISMATCH"}}
    frozen = {"model": {"model_source": "fixture"}, "sampling": {"temperature": 0}, "generation": {"max_new_tokens": 1}, "prompt_version": "fixture"}
    result = summarize([record], frozen, {**benchmark, "cases": [benchmark["cases"][0]]}, [{"worker_id": 0, "gpu_id": 0}], {"bytes": 0})
    assert result["go_decision"] == "NO_GO_API"
    assert result["arc_data_used"] is False


def test_runner_summary_does_not_expose_model_mount_path():
    benchmark = benchmark_definition()
    record = {"case_id": benchmark["cases"][0]["case_id"], "score": {"response_received": True, "json_parseable": True, "schema_valid": True, "macro_ids_valid": True, "argument_contract_valid": True, "type_valid": True, "parameter_valid": True, "composition_valid": True, "compile_valid": True, "canonical_macro_sequence_match": True, "instruction_constraints_valid": True, "failure_type": None}}
    frozen = {"model": {"model_source": "fixture"}, "sampling": {"temperature": 0}, "generation": {"max_new_tokens": 1}, "prompt_version": "fixture"}
    result = summarize([record], frozen, {**benchmark, "cases": [benchmark["cases"][0]]}, [{"worker_id": 0, "gpu_id": 0}], {"model_path": "/kaggle/input/private-model", "bytes_read": 1, "seconds": 2})
    assert "model_path" not in result["model_warmup"]


def test_public_finalizer_accepts_only_safe_aggregate(tmp_path):
    path = Path(__file__).parents[1] / "scripts" / "finalize_macro_api_comprehension_benchmark.py"
    spec = importlib.util.spec_from_file_location("macro_finalizer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    benchmark = benchmark_definition()
    fields = ("response_received", "json_parseable", "schema_valid", "macro_ids_valid", "argument_contract_valid", "type_valid", "parameter_valid", "composition_valid", "compile_valid")
    categories = {category: {"case_count": 15, "schema_valid": 15, "type_valid": 15, "compile_valid": 15, "instruction_constraints_valid": 15} for category in CATEGORIES}
    aggregate = {"experiment_id": "MACRO_API_COMPREHENSION_BENCHMARK_V1", "status": "COMPLETE_FROZEN_ONE_RUN", "case_count": 60, "benchmark_hash": benchmark["benchmark_hash"], "arc_data_used": False, "arc_solutions_used": False, "candidate_budget": 1, "funnel": {field: 60 for field in fields}, "rates": {field: 1.0 for field in fields}, "category_metrics": categories, "top_failure_types": [], "go_decision": "GO_API_LEARNED", "recommended_next_experiment": "ORACLE_LADDER", "max_new_tokens": 256, "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"], "frozen_config_sha256": "fixture", "model": {"model_source": "fixture"}, "model_warmup": {"seconds": 1}, "completed_at_utc": "2026-09-11T00:00:00+00:00", "run_wall_seconds": 2}
    module.validate_aggregate(aggregate, benchmark)
    report = module.report_text(aggregate, benchmark, tests_passed=104)
    assert "Raw completions" in report and "/kaggle/" not in report
    ledger = tmp_path / "experiments.csv"
    ledger.write_text("experiment_id,date,git_commit,solver,representation,search_method,llm_model,candidate_budget,validation_split,tasks_solved,accuracy,runtime_seconds,gpu_hours,notes\n", encoding="utf-8")
    module.update_ledger(ledger, aggregate, "fixture")
    assert "MACRO_API_COMPREHENSION_BENCHMARK_V1" in ledger.read_text(encoding="utf-8")
