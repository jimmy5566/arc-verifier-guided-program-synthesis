from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from novel_training_data_v1.pipeline import (
    ROLE_HOLDOUT,
    SIGNATURE_FIELDS,
    WORKERS,
    _init_tokenizer,
    api_tokenize_text,
    candidate_rejection,
    digest,
    portable_fingerprint,
    process_1d_item,
    split_families,
    validate_curriculum_row,
    validate_grid,
    bounded_pool_map,
)


def _row(prefix: str = "x") -> dict:
    return {field: f"{prefix}-{field}" for field in SIGNATURE_FIELDS}


def _indices() -> dict:
    return {bucket: {field: set() for field in SIGNATURE_FIELDS} for bucket in ("replay", "hard_exclude")}


@pytest.mark.parametrize("field,marker", [
    ("full_content_sha256", "EXACT_FULL_CONTENT"),
    ("d4_signature_sha256", "D4_EQUIVALENT"),
    ("color_signature_sha256", "COLOR_EQUIVALENT"),
    ("d4_color_signature_sha256", "D4_COLOR_EQUIVALENT"),
])
def test_v2_overlap_rejection_modes(field: str, marker: str) -> None:
    row = _row(); indices = _indices(); indices["replay"][field].add(row[field])
    reason, _ = candidate_rejection(row, indices, {name: {} for name in SIGNATURE_FIELDS})
    assert reason is not None and reason.startswith("V2_1_REPLAY_") and marker in reason


def test_project_blacklist_precedes_replay() -> None:
    row = _row(); indices = _indices()
    indices["hard_exclude"]["full_content_sha256"].add(row["full_content_sha256"])
    indices["replay"]["full_content_sha256"].add(row["full_content_sha256"])
    reason, _ = candidate_rejection(row, indices, {name: {} for name in SIGNATURE_FIELDS})
    assert reason.startswith("PROJECT_BLACKLIST_OR_HARD_EXCLUDE_")


def test_exact_new_duplicate_rejected() -> None:
    row = _row(); seen = {name: {} for name in SIGNATURE_FIELDS}
    seen["full_content_sha256"][row["full_content_sha256"]] = "prior"
    assert candidate_rejection(row, _indices(), seen) == ("NEW_CORPUS_EXACT_FULL_CONTENT_OVERLAP", "prior")


def test_source_family_seen_is_catalog_excluded() -> None:
    from novel_training_data_v1.pipeline import source_catalog
    catalog = {row["source"]: row for row in source_catalog()}
    assert catalog["arc_agi_1"]["status"] == "REJECTED"
    assert catalog["larc"]["status"] == "REJECTED"


def test_family_split_isolated_and_deterministic() -> None:
    rows = []
    for source in ("a", "b"):
        for index in range(20):
            rows.append({"source":source,"generator_family":f"{source}:{index}"})
    first, mapping_a = split_families(rows); second, mapping_b = split_families(list(reversed(rows)))
    assert first == second and mapping_a == mapping_b
    assert not (set(first["train"]) & set(first["validation"]))
    assert not (set(first["train"]) & set(first["holdout"]))


def test_holdout_and_quarantine_cannot_be_sampled() -> None:
    assert not validate_curriculum_row({"final_training_role": ROLE_HOLDOUT})
    assert not validate_curriculum_row({"final_training_role": "QUARANTINE_PROVENANCE"})
    assert validate_curriculum_row({"final_training_role": "DEFENSIBLE_NOVEL_TRAINABLE"})


def test_source_specific_1d_width_supports_long_sequences() -> None:
    assert validate_grid([list(range(10)) * 20], field="fixture") == [list(range(10)) * 20]


def test_bounded_pool_map_preserves_order() -> None:
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        values = list(bounded_pool_map(pool, digest, range(1000), batch_size=37, chunksize=3))
    assert values == [digest(value) for value in range(1000)]


def test_twenty_worker_processing_is_deterministic(tmp_path: Path) -> None:
    task = {"train":[{"input":[[1]],"output":[[2]]}],"test":[{"input":[[3]],"output":[[4]]}]}
    paths=[]
    for index in range(20):
        path=tmp_path/f"family_{index}.json"; path.write_text(json.dumps(task),encoding="utf-8"); paths.append(path)
    items=[(str(path),"v1",str(tmp_path)) for path in paths]
    doubled=items+list(reversed(items))
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        cooked=list(pool.map(process_1d_item,doubled,chunksize=1))
    first,second=cooked[:len(items)],cooked[len(items):]
    assert sorted((r["base_puzzle_id"],r["full_content_sha256"]) for r in first) == sorted((r["base_puzzle_id"],r["full_content_sha256"]) for r in second)


def test_actual_tokenizer_api(tmp_path: Path) -> None:
    root=Path(__file__).resolve().parents[1]
    model=root/"data/raw/models/sorokin_qwen3_4b_grids15_sft139_transformers_bfloat16_1"
    if not model.exists(): pytest.skip("untracked tokenizer unavailable")
    _init_tokenizer(str(model))
    text="<|im_start|>user\n1<|im_end|><|im_start|>assistant\n2<|im_end|>"
    ids,labels,details=api_tokenize_text(text)
    assert len(ids)==len(labels) and ids[-1]==details["eos_token_id"]
    assert any(value==-100 for value in labels) and any(value!=-100 for value in labels)


def test_portable_fingerprint_has_no_paths() -> None:
    manifests={"train":[{"logical_name":"train/novel/a.parquet","rows":1,"bytes":10,"sha256":"x","tokens":2,"supervised_tokens":1}],"validation":[]}
    splits={"train":["a"],"validation":["b"],"holdout":["c"]}
    catalog=[{"source":"s","version":"v","status":"ACCEPTED_FOR_TRAINING"}]
    first=portable_fingerprint(manifests,splits,catalog)
    manifests["train"][0]["bytes"]=999
    second=portable_fingerprint(manifests,splits,catalog)
    assert first["fingerprint_sha256"]==second["fingerprint_sha256"]
    assert not first["absolute_paths"]
