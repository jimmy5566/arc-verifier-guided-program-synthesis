from scripts import amend_e04_d_chord_sensitivity_v2 as m

def test_chord_radius_dominates_old_one_minus_cosine_near_one():
    old = 1.0 - 0.99995
    assert m.radius(0.99995, 0.99995) > 100.0 * old

def test_sign_ambiguity_and_frequency_bounds_are_conservative():
    u = m.radius(0.99995, 0.99995)
    assert m.pair_status(-0.005, u) == 'NUMERICALLY_SIGN_AMBIGUOUS'
    assert m.frequency_counts([-0.5, -0.001, 0.4], u) == (1, 1, 1)

def test_bootstrap_resamples_complete_turn_quartets():
    rows = m.base_resample_rows(12, 4, 20261011)
    assert len(rows) == 48
    for base in range(12):
        multiplicity = rows.count(base * 4)
        assert all(rows.count(base * 4 + turn) == multiplicity for turn in range(4))
