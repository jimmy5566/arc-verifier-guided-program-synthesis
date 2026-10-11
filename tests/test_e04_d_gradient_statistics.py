from scripts import e04_d_gradient_statistics as s

def test_chord_frequency_classification_uses_base_blocks():
 rows=[]
 for base in range(12):
  for turn in range(4): rows.append({'canonical_base_id':str(base),'rotation_marker_cosine':-.2})
 r=s.classify(rows,{'FIXED_TURN_ROTATION_CONTROL':0.999999,'MARKER_BINDING_CONTROL':0.999999,'NO_TRANSFORM_RETENTION_CONTROL':1.0})
 assert r['decision']=='LOCAL_CONFLICT_SUPPORTED'
 assert r['negative_frequency_lower_ci']['lower']>.5

def test_ambiguous_sign_cannot_support_conflict():
 rows=[]
 for base in range(12):
  for turn in range(4): rows.append({'canonical_base_id':str(base),'rotation_marker_cosine':-.005})
 r=s.classify(rows,{'FIXED_TURN_ROTATION_CONTROL':.99995,'MARKER_BINDING_CONTROL':.99995,'NO_TRANSFORM_RETENTION_CONTROL':1.0})
 assert r['decision']=='MIXED_OR_INCONCLUSIVE'
