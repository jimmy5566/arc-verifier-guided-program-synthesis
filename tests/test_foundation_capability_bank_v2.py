import json
from pathlib import Path

from foundation_capability_bank_v2.pipeline import (
    SOURCE_COMMIT, apply_program, atomic_programs, compatible_programs,
    composition_programs, components, episode, make_input, semantic_errors,
)

ROOT=Path(__file__).parents[1]
ART=ROOT/"artifacts/foundation_capability_bank_v2"


def read(name): return json.loads((ART/name).read_text(encoding="utf-8"))


def test_source_pin_and_distinct_program_ids():
    assert SOURCE_COMMIT=="6170605158bb5380c5eed58e2af5f8d4838dcf30"
    programs=atomic_programs(); assert len(programs)==32
    assert len({p.program_id for p in programs})==len(programs)
    assert len({p.signature for p in programs})==len(programs)


def test_known_semantic_repairs():
    by={p.name:p for p in atomic_programs()}
    task=episode(by["connect_same_color"],0)
    for pair in task["train"]:
        counts={}
        for obj in components(pair["input"]): counts[obj["color"]]=counts.get(obj["color"],0)+1
        assert len([c for c,n in counts.items() if n==2])==1
    count_task=episode(by["count_components"],1)
    for pair in count_task["train"]:
        assert len(components(pair["input"])) != len({o["color"] for o in components(pair["input"])})
    for name in ("largest_recolor","smallest_recolor","repeat_horizontal","repeat_vertical","align_row","align_column","conditional_count_parity","conditional_touching"):
        assert semantic_errors(by[name],episode(by[name],2))==[]


def test_composition_closure_and_exact_isolation():
    atomic=atomic_programs(); two,three,meta=composition_programs(); atomic_steps={s for p in atomic for s in p.steps}
    assert all(set(p.steps)<=atomic_steps for p in two+three+meta)
    assert not ({p.signature for p in atomic}&{p.signature for p in two+three})


def test_demonstrations_identify_intended_program():
    atomic=atomic_programs(); two,three,meta=composition_programs(); hypotheses=atomic+two+three+meta
    for program in atomic[:8]:
        task=episode(program,3)
        assert compatible_programs(task,hypotheses)==[program.program_id]


def test_frozen_gate_and_forbidden_access():
    gate=read("FOUNDATION_CAPABILITY_BANK_V2_GATE.json")
    assert gate["GPU_BENCHMARK_READY"] is True
    assert gate["GPU_TRAINING_STARTED"] is False
    assert gate["ROUND1_HOLDOUT_TRAINING_ROWS"]==0
    assert gate["COMPOSITION_HOLDOUT_MODEL_ACCESSED"] is False
    assert gate["META_HOLDOUT_MODEL_ACCESSED"] is False
    assert gate["EVAL60_ACCESSED"] is gate["KAGGLE_GOLD_ACCESSED"] is False


def test_audits_pass_and_fingerprint_portable():
    for name in ("PROGRAM_DISTINGUISHABILITY_AUDIT.json","DEMONSTRATION_IDENTIFIABILITY_AUDIT.json","R2_V2_SEMANTIC_NONDEGENERACY_AUDIT.json","R2_V2_OVERLAP_AUDIT.json","V2_TOKENIZER_PARITY_AUDIT.json","V2_CONTEXT_LENGTH_AUDIT.json","V2_WORKER_DETERMINISM_AUDIT.json","V2_CPU_DATALOADER_DRY_RUN.json"):
        assert read(name)["status"]=="PASS"
    fp=read("CAPABILITY_BANK_V2_DATASET_FINGERPRINT.json")
    assert fp["status"]=="PASS" and fp["relocation_invariant"] is True
    assert fp["logical_manifest"]["absolute_paths"] is False
