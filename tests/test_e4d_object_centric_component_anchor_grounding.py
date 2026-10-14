from __future__ import annotations
import json
from pathlib import Path
from scripts.e4d_object_centric_component_anchor_grounding import anchor_cells, connected_components, query_grid_anchor_layout, scalar_mass_contract
from scripts.run_e4d_object_centric_component_anchor_grounding_v1 import schedule, anchor_audit, noncount_identity, panels, cpu_fixture


def test_anchor_labels_are_color_separated_and_lexicographic():
 grid=[[1,1,0,2],[1,0,0,2],[0,0,3,0]]
 assert len(connected_components(grid)) == 3
 assert anchor_cells(grid) == {(0,0),(0,3),(2,2)}


def test_real_tokenized_count_slots_and_loss_contract():
 rows=schedule(); report,mass=anchor_audit(rows)
 assert len(report)==48
 assert {row['query_grid_tokens'] for row in report}=={81}
 assert [sum(row['positive_anchor_tokens']==n for row in report) for n in (1,2,3)]==[16,16,16]
 assert mass['control']['slot_coefficient']==mass['treatment']['slot_coefficient']==.25
 assert mass['treatment']['answer_coefficient']==.5 and mass['treatment']['anchor_coefficient']==.5


def test_noncount_paths_panels_and_no_model_production_trace():
 noncount=noncount_identity(schedule())
 assert len(noncount)==336 and {x['loss_path'] for x in noncount}=={'ORIGINAL_EQUAL_SLOT_CE'}
 assert panels()=={'ood_canonical_units':48,'ood_records':96,'marker_canonical_units':8,'marker_records':16,'target_sidecar_opened':False}
 trace=cpu_fixture()
 assert trace[1]['model_loaded'] is False and trace[2]['targets_opened'] is False and trace[-1]['scored'] is False