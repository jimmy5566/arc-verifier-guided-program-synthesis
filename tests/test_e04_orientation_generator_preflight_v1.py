from scripts.e04_orientation_generator_preflight_v1 import CONDITIONS, enumerate_plan, fixture_seed, validate_plan

def test_plan_counts_and_disjoint_latent_identities():
    train,valid=enumerate_plan('TRAIN'),enumerate_plan('VALIDATION')
    summary=validate_plan(train,valid)
    assert summary['train_base_tuples']==288 and summary['validation_base_tuples']==48
    assert summary['paired_condition_views_per_base_tuple']==4
    assert len(CONDITIONS)==4

def test_non_scientific_fixture_namespace_and_invalid_split():
    train=enumerate_plan('TRAIN')
    assert fixture_seed(0)==910000000
    assert fixture_seed(3)==910000003
    assert min(x['seed'] for x in train)>=810000
    try: enumerate_plan('FINAL_AUDIT')
    except ValueError: pass
    else: raise AssertionError('invalid split accepted')
