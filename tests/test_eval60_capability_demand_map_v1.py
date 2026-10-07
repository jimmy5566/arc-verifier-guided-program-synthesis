from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from eval60_capability_demand_map_v1 import pipeline


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "analysis/eval60_capability_demand_map_v1"


def _json(name: str):
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_annotation_cohort_and_v3_ontology_are_exact() -> None:
    manifest = json.loads((ROOT / "governance/registry/cohorts/eval60.json").read_text(encoding="utf-8"))
    ontology, identity = pipeline.load_ontology(ROOT)
    concepts = {concept for values in ontology.values() for concept in values}
    assert identity["concept_count"] == 83
    assert set(pipeline.ANNOTATIONS) == set(manifest["task_ids"])
    assert len(pipeline.ANNOTATIONS) == 60
    assert all(set(row["required_capabilities"]) <= concepts for row in pipeline.ANNOTATIONS.values())


def test_identity_and_target_blind_freeze() -> None:
    identity = _json("EVAL60_IDENTITY_AUDIT.json")
    freeze = _json("EVAL60_FAMILY_MAP_FREEZE.json")
    demand = _json("EVAL60_CAPABILITY_DEMAND_MAP_V1.json")
    assert identity["status"] == "PASS"
    assert (identity["task_count"], identity["test_output_count"]) == (60, 89)
    assert identity["selected_task_hash_parity"] == "60_OF_60_PASS"
    assert identity["test_input_hash_parity"] == "89_OF_89_PASS"
    assert freeze["taxonomy_frozen_before_historical_outcome_join"] is True
    assert freeze["TEST_GOLD_ACCESSED_DURING_CLASSIFICATION"] is False
    assert freeze["HISTORICAL_OUTCOME_ACCESSED_DURING_CLASSIFICATION"] is False
    assert freeze["CAPABILITY_PROFILE_ACCESSED_DURING_CLASSIFICATION"] is False
    assert demand["classification_contract"]["test_output_fields_discarded_before_classification"] is True
    assert len(demand["tasks"]) == 60


def test_taxonomy_hashes_and_output_index() -> None:
    freeze = _json("EVAL60_FAMILY_MAP_FREEZE.json")
    for name, expected in freeze["taxonomy_artifact_sha256"].items():
        assert _sha(ARTIFACT / name) == expected
    with (ARTIFACT / "EVAL60_OUTPUT_CAPABILITY_INDEX_V1.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 89
    assert len({row["output_id"] for row in rows}) == 89


def test_post_freeze_oracle_and_future_alignment_contract() -> None:
    oracle = _json("EVAL60_HISTORICAL_ORACLE_BY_FAMILY.json")
    alignment = _json("EVAL60_CAPABILITY_ALIGNMENT_PROTOCOL.json")
    assert oracle["total_outputs"] == 89
    assert oracle["ORC_UNION_solved"] == 35
    assert "candidate-pool oracle" in oracle["claim_limit"]
    assert alignment["status"].startswith("NOT_RUN")
    assert alignment["taxonomy_labels_are_immutable"] is True
    assert alignment["Base_profile_loaded"] is False
    assert alignment["Foundation_V2_profile_loaded"] is False
    assert alignment["alignment_categories_calculated"] is False
