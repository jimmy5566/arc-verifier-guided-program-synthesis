from scripts.e04_orientation_generator_preflight_v2 import *
def bad(rows,mut):
 x=[dict(r) for r in rows];mut(x);return x

def rejects(rows,code):
 try: validate_plan(rows,enumerate_plan('VALIDATION'))
 except PreflightFailure as e: assert str(e)==code
 else: raise AssertionError(code)
def test_positive_and_frozen_binding():
 bind_frozen_inputs();assert validate_plan(enumerate_plan('TRAIN'),enumerate_plan('VALIDATION'))['validation_base_tuples']==48;assert fixture_seed(1)==910000001
def test_negative_fail_closed():
 t=enumerate_plan('TRAIN');v=enumerate_plan('VALIDATION')
 x=[dict(r) for r in t];x[0]['split']='VALIDATION';
 try: validate_plan(x,v)
 except PreflightFailure as e: assert str(e)=='WRONG_SPLIT'
 else: raise AssertionError
 x=[dict(r) for r in t];x[1]['seed']=x[0]['seed'];rejects(x,'DUPLICATE_SEED')
 x=[dict(r) for r in t];x[1]['semantic_signature_sha256']=x[0]['semantic_signature_sha256'];rejects(x,'SEMANTIC_OVERLAP')
 x=[dict(r) for r in t];x[0]['color_role_tuple']=(1,1,2,3);rejects(x,'INVALID_COLOR_ROLE')
 try: bind_frozen_inputs(Path('missing'),PREFREEZE)
 except FileNotFoundError: pass
 else: raise AssertionError
