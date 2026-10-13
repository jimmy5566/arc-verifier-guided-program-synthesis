from scripts.run_connected_component_fit_generation_diagnostic_v1 import _raw_checkpoint_name

def test_preserves_full_checkpoint_prefix():
 assert _raw_checkpoint_name("CONTROL_NARROW_STRUCTURE_RAW.jsonl") == "CONTROL_NARROW_STRUCTURE"
 assert _raw_checkpoint_name("TREATMENT_STRUCTURE_DIVERSE_RAW.jsonl") == "TREATMENT_STRUCTURE_DIVERSE"
 assert _raw_checkpoint_name("V7_RAW.jsonl") == "V7"

def test_rejects_non_raw_filename():
 try:
  _raw_checkpoint_name("V7.jsonl")
 except RuntimeError as exc:
  assert str(exc) == "RAW_FILENAME_LAYOUT_INVALID"
 else:
  raise AssertionError("bad raw filename accepted")
