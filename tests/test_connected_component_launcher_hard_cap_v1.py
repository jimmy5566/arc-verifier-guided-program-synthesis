from scripts.launch_connected_component_fit_generation_diagnostic_v1 import frozen_hard_cap_seconds

def test_accepts_only_frozen_caps_without_retry():
 assert frozen_hard_cap_seconds({"hard_cap_seconds":7200,"retry":False}) == 7200
 assert frozen_hard_cap_seconds({"hard_cap_seconds":10800,"retry":False}) == 10800

def test_rejects_unfrozen_cap_and_retry():
 for execution in ({"hard_cap_seconds":7199,"retry":False},{"hard_cap_seconds":7200,"retry":True}):
  try:
   frozen_hard_cap_seconds(execution)
  except RuntimeError as error:
   assert str(error) == "EXECUTION_SHAPE_MISMATCH"
  else:
   raise AssertionError("unfrozen execution shape was accepted")
