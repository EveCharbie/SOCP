"""
Fast, no-solve regression test: builds the NLP (via prepare_ocp, symbolic construction only,
no IPOPT) for a broad set of (transcription, discretization) combinations and asserts the
number of decision variables / constraints match reference counts captured from the current
code. This is what catches an accidental change to variable/constraint bookkeeping in any
declare_variables/set_dynamics_constraints method, across every transcription and discretization
method at once, without paying for a full solve.

VertebrateArm is used (not the paper's own ArmReaching benchmark) because ArmReaching's
get_bounds_and_init currently returns 7 values while prepare_ocp expects 10 (ref bounds missing) -
a pre-existing issue unrelated to this test suite.

DirectMultipleShooting and Variational x MeanAndCovariance are excluded: DMS's
initialize_dynamics_integrator currently requires an `is_brownian` argument that
prepare_ocp's call site does not pass (in-progress work, not something this suite should paper
over), and Variational x MeanAndCovariance fails during construction with a CasADi "free
variables" error rather than the intended clean RuntimeError guard.
"""
import pytest

from socp import (
    VertebrateArm,
    prepare_ocp,
    DirectCollocationTrapezoidal,
    DirectCollocationPolynomial,
    Variational,
    VariationalPolynomial,
    Deterministic,
    NoiseDiscretization,
    MeanAndCovariance,
)

N_SHOOTING = 3

# (transcription, discretization, nb_random) -> (nb_variables, nb_constraints)
COMBOS = {
    ("DirectCollocationTrapezoidal", "Deterministic"): (DirectCollocationTrapezoidal, Deterministic, 1, 25, 14),
    ("DirectCollocationTrapezoidal", "NoiseDiscretization"): (DirectCollocationTrapezoidal, NoiseDiscretization, 10, 193, 130),
    ("DirectCollocationTrapezoidal", "MeanAndCovariance"): (DirectCollocationTrapezoidal, MeanAndCovariance, 10, 209, 166),
    ("DirectCollocationPolynomial", "Deterministic"): (DirectCollocationPolynomial, Deterministic, 1, 97, 86),
    ("DirectCollocationPolynomial", "NoiseDiscretization"): (DirectCollocationPolynomial, NoiseDiscretization, 10, 913, 850),
    ("DirectCollocationPolynomial", "MeanAndCovariance"): (DirectCollocationPolynomial, MeanAndCovariance, 10, 473, 430),
    ("Variational", "Deterministic"): (Variational, Deterministic, 1, 21, 10),
    ("Variational", "NoiseDiscretization"): (Variational, NoiseDiscretization, 10, 153, 90),
    ("VariationalPolynomial", "Deterministic"): (VariationalPolynomial, Deterministic, 1, 57, 46),
    ("VariationalPolynomial", "NoiseDiscretization"): (VariationalPolynomial, NoiseDiscretization, 10, 513, 450),
    ("VariationalPolynomial", "MeanAndCovariance"): (VariationalPolynomial, MeanAndCovariance, 10, 169, 138),
}


@pytest.mark.parametrize("combo_name", list(COMBOS.keys()), ids=lambda c: f"{c[0]}-{c[1]}")
def test_problem_construction_sizes(combo_name):
    transcription_cls, discretization_cls, nb_random, expected_nb_variables, expected_nb_constraints = COMBOS[combo_name]

    ocp_example = VertebrateArm(nb_random=nb_random)
    ocp_example.n_shooting = N_SHOOTING
    dynamics_transcription = transcription_cls()
    discretization_method = discretization_cls(dynamics_transcription)

    ocp = prepare_ocp(ocp_example, dynamics_transcription, discretization_method)

    assert int(ocp["w"].shape[0]) == expected_nb_variables
    assert int(ocp["g"].shape[0]) == expected_nb_constraints
