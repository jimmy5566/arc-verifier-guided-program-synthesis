from __future__ import annotations
import json, unittest
from pathlib import Path
from scripts.freeze_e4f2_distance_overlap_affinity_v1 import freeze, OUT
from scripts.e4f2_component_affinity import affinity_loss, pair_layout, scalar_mass
from scripts.freeze_e4f2_counterfactual_ood_v1 import freeze as freeze_ood
from scripts.run_e4f2_component_affinity_cpu_preflight import cpu_integration, promotion

class E4F2Tests(unittest.TestCase):
    def test_counterfactual_pair_construct_is_deterministic_balanced_and_nonseparable(self):
        one=freeze(); two=freeze(); self.assertEqual(one, two)
        self.assertEqual(one["status"], "PASS_E4F2_COUNTERFACTUAL_TRAIN_FREEZE_AND_PAIR_SHORTCUT_AUDIT_NO_MODEL")
        spec=json.loads((OUT/"PAIR_AFFINITY_SPEC.json").read_text())
        self.assertEqual((spec["treated_slots"], len(spec["untreated_count1_slots"])), (32,16))
        self.assertEqual(spec["audit"]["selected_count"],32)
        self.assertEqual(spec["audit"]["shortcut_audit"],"PASS_EXACT_SELECTED_MARGINAL_MATCH_AND_NO_LISTED_SINGLE_OR_COMBINED_FEATURE_PERFECT")

    def test_real_tokenization_pair_layout_loss_mass_and_count_one_zero_auxiliary(self):
        freeze()
        spec=json.loads((OUT/"PAIR_AFFINITY_SPEC.json").read_text())
        schedule=json.loads((OUT/"SUCCESSOR_COUNTERFACTUAL_SCHEDULE.json").read_text())["episodes"]
        task_by_slot={x["slot"]:x["task"] for x in schedule}
        treated=0
        for row in spec["records"]:
            if row["count"] == 1:
                control=scalar_mass(2.0, 3.0, False, 1)
                treatment=scalar_mass(2.0, 3.0, True, 1)
                self.assertEqual(control, treatment)
                continue
            layout=pair_layout(task_by_slot[row["slot"]], row["selected_pairs"])
            self.assertEqual(layout.component_count, row["count"])
            self.assertIn(layout.labels[0], (0,1))
            self.assertEqual(len(layout.positions), sum(v != 0 for r in task_by_slot[row["slot"]]["test"][0]["input"] for v in r))
            treated += 1
        self.assertEqual(treated, 32)
        self.assertEqual(scalar_mass(2.0, 4.0, True, 2)["step_term"], 0.75)

    def test_non_count_slots_are_byte_identical_to_the_frozen_source_schedule(self):
        freeze()
        source=json.loads(Path("experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1/TREATMENT_SCHEDULE.json").read_text())["episodes"]
        successor=json.loads((OUT/"SUCCESSOR_COUNTERFACTUAL_SCHEDULE.json").read_text())["episodes"]
        import hashlib
        canonical=lambda value: json.dumps(value,sort_keys=True,separators=(",",":")).encode()
        for before,after in zip(source,successor,strict=True):
            if before["role"] != "PRIMARY_CONNECTED_COMPONENT_COUNT":
                self.assertEqual(hashlib.sha256(canonical(before)).hexdigest(), hashlib.sha256(canonical(after)).hexdigest())

    def test_exact_global_and_per_stratum_pair_contract_has_no_serialized_generator_metadata(self):
        freeze()
        spec=json.loads((OUT/"PAIR_AFFINITY_SPEC.json").read_text())
        selected=spec["audit"]["selected_pairs"]
        self.assertEqual(len(selected),32)
        self.assertEqual({label:sum(x["label"] == label for x in selected) for label in ("SAME_COMPONENT","DIFFERENT_COMPONENT")}, {"SAME_COMPONENT":16,"DIFFERENT_COMPONENT":16})
        for count in (2,3):
            rows=[x for x in selected if x["count"] == count]
            self.assertEqual({label:sum(x["label"] == label for x in rows) for label in ("SAME_COMPONENT","DIFFERENT_COMPONENT")}, {"SAME_COMPONENT":8,"DIFFERENT_COMPONENT":8})
        features=("colour_equal","manhattan","chebyshev","relative_displacement","absolute_coordinates","component_sizes","topology","count","pair_order")
        for feature in features:
            same=sorted(json.dumps(x[feature],sort_keys=True) for x in selected if x["label"] == "SAME_COMPONENT")
            different=sorted(json.dumps(x[feature],sort_keys=True) for x in selected if x["label"] == "DIFFERENT_COMPONENT")
            self.assertEqual(same,different)
        schedule=json.loads((OUT/"SUCCESSOR_COUNTERFACTUAL_SCHEDULE.json").read_text())["episodes"]
        self.assertTrue(all("variant" not in row["task"] and "generator" not in row["task"] for row in schedule))

    def test_affinity_is_pair_order_invariant_and_gradients_are_finite(self):
        import torch
        freeze()
        spec=json.loads((OUT/"PAIR_AFFINITY_SPEC.json").read_text())
        schedule=json.loads((OUT/"SUCCESSOR_COUNTERFACTUAL_SCHEDULE.json").read_text())["episodes"]
        first=next(r for r in spec["records"] if r["count"] == 2)
        layout=pair_layout({x["slot"]:x["task"] for x in schedule}[first["slot"]], first["selected_pairs"])
        hidden=torch.randn((1,max(layout.positions)+1,8),requires_grad=True)
        value,_=affinity_loss(hidden,layout); value.backward()
        self.assertTrue(bool(torch.isfinite(value)))
        self.assertGreater(float(hidden.grad.abs().sum()),0.0)
        swapped=type(layout)(layout.positions,layout.cells,tuple((b,a) for a,b in layout.pairs),layout.labels,layout.component_count)
        reversed_value,_=affinity_loss(hidden.detach(),swapped)
        self.assertEqual(float(value.detach()),float(reversed_value.detach()))

    def test_new_ood_is_target_blind_balanced_and_two_order(self):
        freeze()
        receipt=freeze_ood()
        self.assertEqual(receipt["status"], "PASS_E4F2_NEW_OOD_PROSPECTIVE_FREEZE_NO_MODEL")
        panel=json.loads((OUT/"NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json").read_text())
        self.assertEqual((panel["canonical_units"], panel["prompt_records"]), (48,96))
        self.assertTrue(panel["target_blind"])
        by_unit={}
        for row in panel["records"]: by_unit.setdefault(row["canonical_unit_id"],set()).add(row["demo_order"])
        self.assertTrue(all(v == {"A","B"} for v in by_unit.values()))

    def test_production_path_freezes_raw_before_mock_sidecar_and_blocks_protected_regression(self):
        freeze(); freeze_ood()
        result=cpu_integration()
        self.assertEqual(result["preflight"], "PASS_E4F2_PRODUCTION_SHAPED_PREFLIGHT_NO_MODEL")
        self.assertTrue(result["raw_freeze_before_sidecar"])
        rejected=promotion({"native_delta":8,"bootstrap_lower":1,"marker":{"endpoint_0":0},"retention":{"f":0},"retention_aggregate":0,"composition":-1,"order_invariance":True})
        self.assertEqual(rejected["decision"], "REJECT")

if __name__ == "__main__": unittest.main()
