from pathlib import Path

from foundation_round2_data_v1.pipeline import (
    ADAPTER_SHA256, V11_FP, V2_FP, apply_oracle, digest, family_specs,
    generate_episode, split_families, validate_episode,
)
from training_data_v2.pipeline import signatures

ROOT=Path(__file__).parents[1]
ART=ROOT/"artifacts/foundation_round2_data_v1"


def test_source_and_identity_pins():
    assert len(ADAPTER_SHA256) == 64
    assert V2_FP == "2a0e2df7df77bf43b461f20b10c9090049f4165699fab4ec8ca0b532fe24e7e7"
    assert V11_FP == "3b040b3fcd45d84361aa5286c3b762e268a5f7c98dae5b3c87ddeb16c5105c82"


def test_family_identity_and_split_isolation():
    specs=family_specs(); assert len(specs)==28
    assert len({s.signature for s in specs})==28
    assert len({s.family_id for s in specs})==28
    splits=split_families(specs)
    assert {k:len(v) for k,v in splits.items()}=={"train":16,"validation":4,"holdout":4,"meta_holdout":4}
    sets=[{s.family_id for s in v} for v in splits.values()]
    assert all(not sets[i]&sets[j] for i in range(4) for j in range(i+1,4))
    assert split_families(specs)==splits


def test_oracles_are_deterministic_and_nondegenerate():
    for spec in family_specs():
        a=generate_episode(spec,7); b=generate_episode(spec,7)
        assert digest(a)==digest(b)
        assert validate_episode(a)==[]
        assert all(pair["output"]==apply_oracle(pair["input"],spec.operator) for pair in a["train"]+a["test"])


def test_equivalence_signatures_detect_d4_and_color():
    task=generate_episode(family_specs()[0],3)
    sig=signatures(task)
    assert len(sig["d4_signature_sha256"])==64
    assert len(sig["color_signature_sha256"])==64
    assert len(sig["d4_color_signature_sha256"])==64


def test_forbidden_names_and_babyarc_cache_absent():
    text=Path(__file__).parents[1].joinpath("src/foundation_round2_data_v1/pipeline.py").read_text(encoding="utf-8")
    assert "arc_objs.pt" not in text
    assert "skip_load_pretrain_obj=False" not in text
    assert "Eval60" not in " ".join(s.signature for s in family_specs())


def _read(name):
    import json
    return json.loads((ART/name).read_text(encoding="utf-8"))


def test_source_license_lineage_and_babyarc_fail_closed():
    registry=_read("SOURCE_ACCEPTANCE_REGISTRY_R2.json")
    baby=next(r for r in registry["records"] if r["source"]=="frankaging/BabyARC")
    assert baby["revision"]=="7357681a35c18dd2f9bfe7c76404e7ceed41ac6d"
    assert baby["license"]=="MIT" and baby["decision"]=="QUARANTINE_LINEAGE"
    assert registry["accepted_novel_sources"]==["repository_owned_round2_procedural"]


def test_tokenizer_context_and_worker_determinism_gates():
    assert _read("R2_TOKENIZER_PARITY_AUDIT.json")["status"]=="PASS"
    assert _read("R2_TOKENIZER_PARITY_AUDIT.json")["assistant_only_masking"] is True
    assert _read("R2_CONTEXT_LENGTH_AUDIT.json")["sequence_length"]["max"]<=8704
    assert _read("R2_WORKER_DETERMINISM_AUDIT.json")["status"]=="PASS"
    assert _read("R2_WORKER_DETERMINISM_AUDIT.json")["workers"]==[12,20]


def test_overlap_holdout_and_dataloader_gates():
    assert _read("R2_OVERLAP_AUDIT.json")["status"]=="PASS"
    loader=_read("R2_CPU_DATALOADER_DRY_RUN.json")
    assert loader["status"]=="PASS"
    assert loader["holdout_rows_sampled"]==loader["meta_holdout_rows_sampled"]==0
    gate=_read("FOUNDATION_ROUND2_GPU_GATE.json")
    assert gate["round1_holdout_training_rows"]==0
    assert gate["EVAL60_ACCESSED"] is gate["KAGGLE_GOLD_ACCESSED"] is False


def test_portable_fingerprint_and_token_aware_mix():
    fp=_read("R2_DATASET_FINGERPRINT.json")
    assert fp["status"]=="PASS" and fp["relocation_invariant"] is True
    assert fp["logical_manifest"]["absolute_paths"] is False
    dry=_read("R2_CURRICULUM_DRY_RUN.json")
    assert dry["status"]=="PASS" and dry["raw_row_count_domination"] is False
    assert dry["family_exposure_min"]>0 and dry["family_exposure_max"]>=dry["family_exposure_min"]
