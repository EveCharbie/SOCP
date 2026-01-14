import numpy as np
import numpy.testing as npt
import casadi as cas

from socp import (
    MeanAndCovariance,
    NoiseDiscretization,
    DirectCollocationPolynomial,
    DirectCollocationTrapezoidal,
    DirectMultipleShooting,
    ArmReaching,
    ObstacleAvoidance
)
from socp.analysis.save_results import get_opt_arrays


def test_mean_and_covariance_polynomial_variable_creation():
    """Test variable creation for MeanAndCovariance with DirectCollocationPolynomial."""
    polynomial_order = 3

    ocp_example = ObstacleAvoidance()
    dynamics_transcription = DirectCollocationPolynomial(order=polynomial_order)
    discretization_method = MeanAndCovariance(
        dynamics_transcription=dynamics_transcription,
        with_cholesky=False,
        with_helper_matrix=True,
    )

    # Get bounds and initial guesses
    (
        states_lower_bounds,
        states_upper_bounds,
        states_initial_guesses,
        controls_lower_bounds,
        controls_upper_bounds,
        controls_initial_guesses,
    ) = ocp_example.get_bounds_and_init(n_shooting=ocp_example.n_shooting)

    # Declare variables
    T, x, z, u, w, w_lb, w_ub, w_init = discretization_method.declare_variables(
        ocp_example=ocp_example,
        states_lower_bounds=states_lower_bounds,
        states_upper_bounds=states_upper_bounds,
        states_initial_guesses=states_initial_guesses,
        controls_lower_bounds=controls_lower_bounds,
        controls_upper_bounds=controls_upper_bounds,
        controls_initial_guesses=controls_initial_guesses,
    )

    # Check that variables were created
    assert T is not None
    assert len(x) == ocp_example.n_shooting + 1
    assert len(z) == ocp_example.n_shooting + 1
    assert len(u) == ocp_example.n_shooting + 1
    assert len(w) > 0
    assert len(w_lb) == len(w_ub) == len(w_init)

    # Create a test optimization vector
    w_test = cas.DM(w_init)

    # Re-access variables
    T_opt, states_opt, collocation_points_opt, controls_opt, x_opt, z_opt, u_opt = (
        discretization_method.get_variables_from_vector(
            model=ocp_example.model,
            states_lower_bounds=states_lower_bounds,
            controls_lower_bounds=controls_lower_bounds,
            vector=w_test,
        )
    )

    # Check dimensions
    for key in states_lower_bounds.keys():
        assert states_opt[key].shape == states_lower_bounds[key].shape

    for key in controls_lower_bounds.keys():
        assert controls_opt[key].shape == controls_lower_bounds[key].shape


    # Test the values against the initial guess
    # T
    npt.assert_array_almost_equal(T_opt, w_test[0], decimal=6)
    # States
    state_names = [name for name in states_initial_guesses.keys() if name not in ["covariance", "m"]]
    for key in state_names:
        npt.assert_almost_equal(
            np.array(states_opt[key]),
            np.array(states_initial_guesses[key]),
            decimal=6,
        )
    # Controls
    for key in controls_initial_guesses.keys():
        npt.assert_almost_equal(
            np.array(controls_opt[key]),
            np.array(controls_initial_guesses[key]),
            decimal=6,
        )

    # Test the accession of opt_arrays
    nb_collocation_points = polynomial_order + 2
    states_opt_array, controls_opt_array = get_opt_arrays(
        ocp_example,
        discretization_method,
        states_opt,
        controls_opt,
        nb_collocation_points,
        ocp_example.n_shooting,
    )
    states_offset = 0
    controls_offset = 0
    nb_total_states = states_opt_array.shape[0]
    nb_total_controls = controls_opt_array.shape[0]
    for i_node in range(ocp_example.n_shooting + 1):
        if i_node < ocp_example.n_shooting:
            npt.assert_array_almost_equal(
                np.array(x_opt[states_offset: states_offset + nb_total_states]).reshape(-1, ),
                states_opt_array[:, i_node],
                decimal=6,
            )
            states_offset += nb_total_states
            npt.assert_array_almost_equal(
                np.array(u_opt[controls_offset: controls_offset + nb_total_controls]).reshape(-1, ),
                controls_opt_array[:, i_node],
                decimal=6,
            )
            controls_offset += nb_total_controls
        else:
            # The last node has normal states, and cov, but not m
            npt.assert_array_almost_equal(
                np.array(x_opt[states_offset: states_offset + ocp_example.model.nb_states]).reshape(-1, ),
                states_opt_array[: ocp_example.model.nb_states, i_node],
                decimal=6,
            )
            states_offset += ocp_example.model.nb_states

            if "covariance" in states_opt.keys():
                if discretization_method.with_cholesky:
                    nb_cov_variables = ocp_example.model.nb_cholesky_components(ocp_example.model.nb_states)
                else:
                    nb_cov_variables = ocp_example.model.nb_states ** 2
                npt.assert_array_almost_equal(
                    np.array(x_opt[states_offset: states_offset + nb_cov_variables]).reshape(-1, ),
                    states_opt_array[ocp_example.model.nb_states: ocp_example.model.nb_states + nb_cov_variables, i_node],
                    decimal=6,
                )
                states_offset += nb_cov_variables

            if "m" in states_opt.keys():
                # No m at the last node
                nb_m_variables = ocp_example.model.nb_states ** 2
                npt.assert_array_almost_equal(
                    np.array(x_opt[states_offset: states_offset + nb_m_variables]).reshape(-1, ),
                    states_opt_array[ocp_example.model.nb_states + nb_cov_variables : ocp_example.model.nb_states + nb_cov_variables + nb_m_variables, i_node],
                    decimal=6,
                )
                npt.assert_array_almost_equal(
                    np.array(x_opt[states_offset: states_offset + nb_m_variables]).reshape(-1, ),
                    np.zeros(nb_m_variables),
                    decimal=6,
                )



def test_mean_and_covariance_trapezoidal_variable_creation(self, arm_reaching_example):
    """Test variable creation for MeanAndCovariance with DirectCollocationTrapezoidal."""
    dynamics_transcription = DirectCollocationTrapezoidal()
    discretization = MeanAndCovariance(
        dynamics_transcription=dynamics_transcription,
        with_cholesky=True,
        with_helper_matrix=False,
    )

    # Get bounds and initial guesses
    (
        states_lower_bounds,
        states_upper_bounds,
        states_initial_guesses,
        controls_lower_bounds,
        controls_upper_bounds,
        controls_initial_guesses,
    ) = arm_reaching_example.get_bounds_and_init(n_shooting=10)

    # Declare variables
    T, x, z, u, w, w_lb, w_ub, w_init = discretization.declare_variables(
        ocp_example=arm_reaching_example,
        states_lower_bounds=states_lower_bounds,
        states_upper_bounds=states_upper_bounds,
        states_initial_guesses=states_initial_guesses,
        controls_lower_bounds=controls_lower_bounds,
        controls_upper_bounds=controls_upper_bounds,
        controls_initial_guesses=controls_initial_guesses,
    )

    # Create a test optimization vector
    w_test = cas.DM(w_init)

    # Re-access variables
    T_opt, states_opt, collocation_points_opt, controls_opt, x_opt, z_opt, u_opt = (
        discretization.get_variables_from_vector(
            model=arm_reaching_example.model,
            states_lower_bounds=states_lower_bounds,
            controls_lower_bounds=controls_lower_bounds,
            vector=w_test,
        )
    )

    # Verify Cholesky decomposition was applied correctly
    assert "covariance" in states_opt
    nb_states = arm_reaching_example.model.nb_states
    for i_node in range(11):
        cov_matrix = states_opt["covariance"][:, :, i_node]
        # Check symmetry
        assert np.allclose(cov_matrix, cov_matrix.T)
        # Check positive semi-definite (eigenvalues >= 0)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvalues >= -1e-10)

def test_noise_discretization_variable_creation(self, obstacle_avoidance_example):
    """Test variable creation for NoiseDiscretization."""
    dynamics_transcription = DirectMultipleShooting()
    discretization = NoiseDiscretization(
        dynamics_transcription=dynamics_transcription,
        with_cholesky=False,
        with_helper_matrix=False,
    )

    # Get bounds and initial guesses
    (
        states_lower_bounds,
        states_upper_bounds,
        states_initial_guesses,
        controls_lower_bounds,
        controls_upper_bounds,
        controls_initial_guesses,
    ) = obstacle_avoidance_example.get_bounds_and_init(n_shooting=10)

    # Declare variables
    T, x, z, u, w, w_lb, w_ub, w_init = discretization.declare_variables(
        ocp_example=obstacle_avoidance_example,
        states_lower_bounds=states_lower_bounds,
        states_upper_bounds=states_upper_bounds,
        states_initial_guesses=states_initial_guesses,
        controls_lower_bounds=controls_lower_bounds,
        controls_upper_bounds=controls_upper_bounds,
        controls_initial_guesses=controls_initial_guesses,
    )

    # Check that variables were created
    assert T is not None
    assert len(x) == 11  # n_shooting + 1
    assert len(u) == 11
    assert len(w) > 0

    # Create a test optimization vector
    w_test = cas.DM(w_init)

    # Re-access variables
    T_opt, states_opt, collocation_points_opt, controls_opt, x_opt, z_opt, u_opt = (
        discretization.get_variables_from_vector(
            model=obstacle_avoidance_example.model,
            states_lower_bounds=states_lower_bounds,
            controls_lower_bounds=controls_lower_bounds,
            vector=w_test,
        )
    )

    # Verify re-accessed variables
    assert T_opt is not None
    assert isinstance(states_opt, dict)
    assert isinstance(controls_opt, dict)

    # Check dimensions for noise discretization (should have nb_random copies)
    nb_random = obstacle_avoidance_example.nb_random
    for key in states_lower_bounds.keys():
        expected_shape = (states_lower_bounds[key].shape[0] * nb_random, states_lower_bounds[key].shape[1])
        assert states_opt[key].shape == expected_shape

def test_variable_bounds_consistency(self, arm_reaching_example):
    """Test that bounds and initial guesses are consistent."""
    dynamics_transcription = DirectCollocationPolynomial(order=3)
    discretization = MeanAndCovariance(
        dynamics_transcription=dynamics_transcription,
        with_cholesky=False,
        with_helper_matrix=True,
    )

    # Get bounds and initial guesses
    (
        states_lower_bounds,
        states_upper_bounds,
        states_initial_guesses,
        controls_lower_bounds,
        controls_upper_bounds,
        controls_initial_guesses,
    ) = arm_reaching_example.get_bounds_and_init(n_shooting=10)

    # Declare variables
    T, x, z, u, w, w_lb, w_ub, w_init = discretization.declare_variables(
        ocp_example=arm_reaching_example,
        states_lower_bounds=states_lower_bounds,
        states_upper_bounds=states_upper_bounds,
        states_initial_guesses=states_initial_guesses,
        controls_lower_bounds=controls_lower_bounds,
        controls_upper_bounds=controls_upper_bounds,
        controls_initial_guesses=controls_initial_guesses,
    )

    # Check that initial guess is within bounds
    w_lb_array = np.array(w_lb)
    w_ub_array = np.array(w_ub)
    w_init_array = np.array(w_init)

    assert np.all(w_init_array >= w_lb_array - 1e-10)
    assert np.all(w_init_array <= w_ub_array + 1e-10)

def test_roundtrip_variable_conversion(self, arm_reaching_example):
    """Test that variables can be converted to vector and back without loss."""
    dynamics_transcription = DirectCollocationPolynomial(order=3)
    discretization = MeanAndCovariance(
        dynamics_transcription=dynamics_transcription,
        with_cholesky=False,
        with_helper_matrix=True,
    )

    # Get bounds and initial guesses
    (
        states_lower_bounds,
        states_upper_bounds,
        states_initial_guesses,
        controls_lower_bounds,
        controls_upper_bounds,
        controls_initial_guesses,
    ) = arm_reaching_example.get_bounds_and_init(n_shooting=10)

    # Declare variables
    T, x, z, u, w, w_lb, w_ub, w_init = discretization.declare_variables(
        ocp_example=arm_reaching_example,
        states_lower_bounds=states_lower_bounds,
        states_upper_bounds=states_upper_bounds,
        states_initial_guesses=states_initial_guesses,
        controls_lower_bounds=controls_lower_bounds,
        controls_upper_bounds=controls_upper_bounds,
        controls_initial_guesses=controls_initial_guesses,
    )

    # Create a test optimization vector
    w_test = cas.DM(w_init)

    # Re-access variables
    T_opt, states_opt, collocation_points_opt, controls_opt, x_opt, z_opt, u_opt = (
        discretization.get_variables_from_vector(
            model=arm_reaching_example.model,
            states_lower_bounds=states_lower_bounds,
            controls_lower_bounds=controls_lower_bounds,
            vector=w_test,
        )
    )

    # Reconstruct the vector from the extracted variables
    w_reconstructed = cas.vertcat(T_opt, x_opt, u_opt)

    # Check that the reconstructed vector matches the original
    assert w_reconstructed.shape == w_test.shape
    assert np.allclose(np.array(w_reconstructed).flatten(), np.array(w_test).flatten())
