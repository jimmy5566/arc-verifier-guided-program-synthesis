import json
from pathlib import Path

from foundation_capability_bank_v3.pipeline import (
    CORE_EXPANSION, SOURCE_COMMIT, atomic_programs, composition_programs,
    episode, semantic_predicates,
)

ROOT=Path(__file__).parents[1]
ART=ROOT/"artifacts/foundation_capability_bank_v3"


def read(name): return json.loads((ART/name).read_text(encoding="utf-8"))


def test_source_pin_and_full_core_coverage():
    assert SOURCE_COMMIT=="b0dab8e48fc0b4b514a6a3571685f0912d85e25a"
    covered={c for p in atomic_programs() for c in p.capabilities}
    assert CORE_EXPANSION <= covered


def test_program_ids_and_signatures_unique():
    programs=atomic_programs()
    assert len(programs)==64
    assert len({p.program_id for p in programs})==len(programs)
    assert len({p.signature for p in programs})==len(programs)


def test_new_programs_execute_mechanical_semantics():
    for program in atomic_programs()[32:]:
        task=episode(program,7,True)
        checks=semantic_predicates(program,task)
        assert not [k for k,v in checks.items() if v is False], (program.name,checks)


def test_composition_partitions_and_exact_isolation():
    atomic=atomic_programs(); two,three,meta=composition_programs()
    assert (len(two),len(three),len(meta))==(20,10,5)
    assert not ({p.signature for p in atomic}&{p.signature for p in two+three})
    assert {p.partition for p in two+three+meta}=={"C1_TRAIN_CANDIDATES","C1_DEV_ISOLATED","C2_TRAIN_CANDIDATES","C2_DEV_ISOLATED","META_COMPOSITION_HOLDOUT"}


def test_frozen_gate_and_no_gpu_or_holdout_access():
    gate=read("FOUNDATION_CAPABILITY_BANK_V3_GATE.json")
    assert gate["GPU_BENCHMARK_READY"] is True
    assert gate["GPU_TRAINING_STARTED"] is False
    assert gate["DIAGNOSTIC_EPISODES_IN_TRAINING"]==0
    assert gate["DIAGNOSTIC_MODEL_EVALUATION_STARTED"] is False
    assert gate["META_HOLDOUT_MODEL_ACCESSED"] is False
    assert gate["ROUND1_HOLDOUT_TRAINING_ROWS"]==0
    assert gate["EVAL60_GOLD_ACCESSED"] is gate["KAGGLE_GOLD_ACCESSED"] is gate["ARC_HIDDEN_GOLD_ACCESSED"] is False


def test_required_v3_audits_pass():
    for name in ("PROGRAM_DISTINGUISHABILITY_AUDIT_V3.json","SEMANTIC_NONDEGENERACY_AUDIT_V3.json","DEMONSTRATION_IDENTIFIABILITY_AUDIT_V3.json","CAPABILITY_FACTORIZATION_AUDIT.json","PARAMETER_DOMAIN_SEPARATION_AUDIT.json","COMPOSITION_ISOLATION_BANK_V3.json","COMPOSITION_GENERALIZATION_MATRIX_V3.json","V3_OVERLAP_AUDIT.json","V3_TOKENIZER_PARITY_AUDIT.json","V3_CONTEXT_LENGTH_AUDIT.json","V3_WORKER_DETERMINISM_AUDIT.json","V3_CPU_DATALOADER_DRY_RUN.json"):
        assert read(name)["status"]=="PASS", name


def test_identifiability_claim_is_scoped():
    audit=read("DEMONSTRATION_IDENTIFIABILITY_AUDIT_V3.json")
    assert audit["claim_scope"]=="UNIQUE_WITHIN_FROZEN_PROGRAM_BANK"
    assert audit["non_unique_or_inconsistent"]==[]
