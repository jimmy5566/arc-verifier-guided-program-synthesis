from __future__ import annotations
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e05_freezer", ROOT / "scripts/freeze_e05_structurally_held_out_marker_conditioned_rotation_v1.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_plan_has_48_bases_and_four_turns_each():
    rows = MODULE.plan()
    assert len(rows) == 192
    assert {r["turn"] for r in rows} == {0, 1, 2, 3}
    assert len({r["canonical_base_id"] for r in rows}) == 48


def test_e05_shape_is_not_e04_shape_up_to_d4():
    assert MODULE.d4_shape_signature(MODULE.POLY) != MODULE.d4_shape_signature(((0, 0), (1, 0), (2, 0), (2, 1)))


def test_cpu_freeze_writes_768_prompts_and_separate_targets():
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "e05"
        manifest = MODULE.freeze(output)
        prompts = (output / "INPUT_PROMPTS.jsonl").read_text(encoding="utf-8").splitlines()
        targets = (output / "TARGET_SCORER_SIDECAR.jsonl").read_text(encoding="utf-8").splitlines()
        assert manifest["episodes"] == 768
        assert len(prompts) == len(targets) == 768
        assert all("target" not in json.loads(line) for line in prompts)
        assert all("query_input" not in json.loads(line) for line in targets)
        audit = json.loads((output / "STRUCTURAL_DISJOINTNESS_AUDIT.json").read_text(encoding="utf-8"))
        assert audit["status"] == "PASS"
        assert audit["source"]["sealed_target_sidecar_opened"] is False
