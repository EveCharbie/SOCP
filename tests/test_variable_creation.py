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
    ObstacleAvoidance,
)


def test_mean_and_covariance_polynomial_variable_creation():
    """Test variable creation for MeanAndCovariance with DirectCollocationPolynomial."""
    from socp.transcriptions.mean_and_covariance import Variables

    polynomial_order = 3
    with_cholesky = False
    with_helper_matrix = True

    ocp_example = ObstacleAvoidance()
    dynamics_transcription = DirectCollocationPolynomial(order=polynomial_order)
    discretization_method = MeanAndCovariance(
        dynamics_transcription=dynamics_transcription,
        with_cholesky=with_cholesky,
        with_helper_matrix=with_helper_matrix,
    )

    # Get bounds and initial guesses
    (
        states_lower_bounds,
        states_upper_bounds,
        states_initial_guesses,
        controls_lower_bounds,
        controls_upper_bounds,
        controls_initial_guesses,
        collocation_points_initial_guesses,
    ) = ocp_example.get_bounds_and_init(
        n_shooting=ocp_example.n_shooting, nb_collocation_points=dynamics_transcription.nb_collocation_points
    )
    motor_noise_magnitude, sensory_noise_magnitude = ocp_example.get_noises_magnitude()

    # Declare variables
    variables_vector = discretization_method.declare_variables(
        ocp_example=ocp_example,
        states_lower_bounds=states_lower_bounds,
        controls_lower_bounds=controls_lower_bounds,
    )
    noises_numerical, noises_single = discretization_method.declare_noises(
        ocp_example.model,
        ocp_example.n_shooting,
        ocp_example.nb_random,
        motor_noise_magnitude,
        sensory_noise_magnitude,
        seed=ocp_example.seed,
    )

    dynamics_transcription.initialize_dynamics_integrator(
        ocp_example=ocp_example,
        discretization_method=discretization_method,
        variables_vector=variables_vector,
        noises_single=noises_single,
    )

    # Declare bounds
    w_lb, w_ub, w_init = discretization_method.declare_bounds_and_init(
        ocp_example=ocp_example,
        states_lower_bounds=states_lower_bounds,
        states_upper_bounds=states_upper_bounds,
        states_initial_guesses=states_initial_guesses,
        controls_lower_bounds=controls_lower_bounds,
        controls_upper_bounds=controls_upper_bounds,
        controls_initial_guesses=controls_initial_guesses,
        collocation_points_initial_guesses=collocation_points_initial_guesses,
    )
    states_initial_guesses, collocation_points_initial_guesses, controls_initial_guesses = (
        discretization_method.modify_init(
            ocp_example,
            w_init,
        )
    )

    # Try to access variables in both ways - Only with symbolic variables
    w0_vector_only_sym = w_init.get_full_vector(keep_only_symbolic=True)
    w0_variables_only_sym = Variables(
        ocp_example.n_shooting,
        dynamics_transcription.nb_collocation_points,
        ocp_example.model.state_indices,
        ocp_example.model.control_indices,
        with_cholesky,
        with_helper_matrix,
    )
    w0_variables_only_sym.set_from_vector(w0_vector_only_sym, only_has_symbolics=True)
    re_w0_variables_only_sym = w0_variables_only_sym.get_full_vector(keep_only_symbolic=True)
    npt.assert_array_almost_equal(w0_vector_only_sym, re_w0_variables_only_sym, decimal=12)

    # Try to access variables in both ways - Only with all variables
    w0_vector_not_only_sym = w_init.get_full_vector(keep_only_symbolic=False)
    w0_variables_only_sym = Variables(
        ocp_example.n_shooting,
        dynamics_transcription.nb_collocation_points,
        ocp_example.model.state_indices,
        ocp_example.model.control_indices,
        with_cholesky,
        with_helper_matrix,
    )
    w0_variables_only_sym.set_from_vector(w0_vector_not_only_sym, only_has_symbolics=False)
    re_w0_variables_not_only_sym = w0_variables_only_sym.get_full_vector(keep_only_symbolic=False)
    npt.assert_array_almost_equal(w0_vector_not_only_sym, re_w0_variables_not_only_sym, decimal=12)

    # # Check that the initialization of M is correct (the dFdz.T - dGdz.T @ M.T constraint should be = 0)
    # for i_node in range(ocp_example.n_shooting):
    #     lbg, ubg, g, g_names = dynamics_transcription.other_internal_constraints(
    #         ocp_example,
    #         discretization_method,
    #         T_opt,  # T
    #         x_opt[i_node * nb_total_states : (i_node + 1) * nb_total_states],  # x
    #         z_opt[i_node * nb_total_collocation_points: (i_node + 1) * nb_total_collocation_points],  # z
    #         u_opt[i_node * nb_total_controls: (i_node + 1) * nb_total_controls],  # u
    #         noises_numerical,
    #     )
    #     nb_defects = 4*(polynomial_order + 1)
    #     nb_m_constraints = 4 * 4 * (polynomial_order + 1)
    #     for i_g in range(nb_defects, nb_defects + nb_m_constraints):
    #         npt.assert_array_almost_equal(g, 0)


def test_mean_and_covariance_polynomial():
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
        collocation_points_initial_guesses,
    ) = ocp_example.get_bounds_and_init(
        n_shooting=ocp_example.n_shooting, nb_collocation_points=dynamics_transcription.nb_collocation_points
    )
    motor_noise_magnitude, sensory_noise_magnitude = ocp_example.get_noises_magnitude()

    # Declare variables
    variables_vector = discretization_method.declare_variables(
        ocp_example=ocp_example,
        states_lower_bounds=states_lower_bounds,
        controls_lower_bounds=controls_lower_bounds,
    )
    noises_numerical, noises_single = discretization_method.declare_noises(
        ocp_example.model,
        ocp_example.n_shooting,
        ocp_example.nb_random,
        motor_noise_magnitude,
        sensory_noise_magnitude,
        seed=ocp_example.seed,
    )

    dynamics_transcription.initialize_dynamics_integrator(
        ocp_example=ocp_example,
        discretization_method=discretization_method,
        variables_vector=variables_vector,
        noises_single=noises_single,
    )

    # Declare bounds
    w_lb, w_ub, w_init = discretization_method.declare_bounds_and_init(
        ocp_example=ocp_example,
        states_lower_bounds=states_lower_bounds,
        states_upper_bounds=states_upper_bounds,
        states_initial_guesses=states_initial_guesses,
        controls_lower_bounds=controls_lower_bounds,
        controls_upper_bounds=controls_upper_bounds,
        controls_initial_guesses=controls_initial_guesses,
        collocation_points_initial_guesses=collocation_points_initial_guesses,
    )

    discretization_method.modify_init(ocp_example, w_init)

    # Check that variables were created
    assert T is not None
    assert len(x) == ocp_example.n_shooting + 1
    assert len(z) == ocp_example.n_shooting + 1
    assert len(u) == ocp_example.n_shooting + 1
    assert len(w) > 0
    assert len(w_lb) == len(w_ub) == len(w_init)

    # Re-access variables
    T_opt, states_opt, collocation_points_opt, controls_opt, x_opt, z_opt, u_opt = (
        discretization_method.get_variables_from_vector(
            model=ocp_example.model,
            states_lower_bounds=states_lower_bounds,
            controls_lower_bounds=controls_lower_bounds,
            vector=w_init,
        )
    )

    # Check dimensions
    for key in states_lower_bounds.keys():
        assert states_opt[key].shape == states_lower_bounds[key].shape

    for key in controls_lower_bounds.keys():
        assert controls_opt[key].shape == controls_lower_bounds[key].shape

    # Test the values against the initial guess
    # T
    npt.assert_array_almost_equal(T_opt, w_init[0], decimal=6)

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
    states_opt_array, collocation_points_opt_array, controls_opt_array = discretization_method.get_var_arrays(
        ocp_example,
        discretization_method,
        states_opt,
        collocation_points_opt,
        controls_opt,
    )
    states_offset = 0
    collocation_points_offset = 0
    controls_offset = 0
    nb_total_states = states_opt_array.shape[0]
    nb_total_collocation_points = collocation_points_opt_array.shape[0]
    nb_total_controls = controls_opt_array.shape[0]
    for i_node in range(ocp_example.n_shooting + 1):
        if i_node < ocp_example.n_shooting:
            npt.assert_array_almost_equal(
                np.array(x_opt[states_offset : states_offset + nb_total_states]).reshape(
                    -1,
                ),
                states_opt_array[:, i_node],
                decimal=6,
            )
            states_offset += nb_total_states
            npt.assert_array_almost_equal(
                np.array(
                    z_opt[collocation_points_offset : collocation_points_offset + nb_total_collocation_points]
                ).reshape(
                    -1,
                ),
                collocation_points_opt_array[:, i_node],
                decimal=6,
            )
            collocation_points_offset += nb_total_collocation_points
            npt.assert_array_almost_equal(
                np.array(u_opt[controls_offset : controls_offset + nb_total_controls]).reshape(
                    -1,
                ),
                controls_opt_array[:, i_node],
                decimal=6,
            )
            controls_offset += nb_total_controls
        else:
            # The last node has normal states, and cov, but not m
            npt.assert_array_almost_equal(
                np.array(x_opt[states_offset : states_offset + ocp_example.model.nb_states]).reshape(
                    -1,
                ),
                states_opt_array[: ocp_example.model.nb_states, i_node],
                decimal=6,
            )
            states_offset += ocp_example.model.nb_states

            if "covariance" in states_opt.keys():
                if discretization_method.with_cholesky:
                    nb_cov_variables = ocp_example.model.nb_cholesky_components(ocp_example.model.nb_states)
                else:
                    nb_cov_variables = ocp_example.model.nb_states**2
                npt.assert_array_almost_equal(
                    np.array(x_opt[states_offset : states_offset + nb_cov_variables]).reshape(
                        -1,
                    ),
                    states_opt_array[
                        ocp_example.model.nb_states : ocp_example.model.nb_states + nb_cov_variables, i_node
                    ],
                    decimal=6,
                )
                states_offset += nb_cov_variables

            if "m" in states_opt.keys():
                # No m at the last node
                nb_m_variables = ocp_example.model.nb_states**2
                npt.assert_array_almost_equal(
                    np.array(x_opt[states_offset : states_offset + nb_m_variables]).reshape(
                        -1,
                    ),
                    states_opt_array[
                        ocp_example.model.nb_states
                        + nb_cov_variables : ocp_example.model.nb_states
                        + nb_cov_variables
                        + nb_m_variables,
                        i_node,
                    ],
                    decimal=6,
                )
                npt.assert_array_almost_equal(
                    np.array(x_opt[states_offset : states_offset + nb_m_variables]).reshape(
                        -1,
                    ),
                    np.zeros(nb_m_variables),
                    decimal=6,
                )

    # Check that the initialization of M is correct (the dFdz.T - dGdz.T @ M.T constraint should be = 0)
    for i_node in range(ocp_example.n_shooting):
        lbg, ubg, g, g_names = dynamics_transcription.other_internal_constraints(
            ocp_example,
            discretization_method,
            T_opt,  # T
            x_opt[i_node * nb_total_states : (i_node + 1) * nb_total_states],  # x
            z_opt[i_node * nb_total_collocation_points : (i_node + 1) * nb_total_collocation_points],  # z
            u_opt[i_node * nb_total_controls : (i_node + 1) * nb_total_controls],  # u
            noises_numerical,
        )
        nb_defects = 4 * (polynomial_order + 1)
        nb_m_constraints = 4 * 4 * (polynomial_order + 1)
        for i_g in range(nb_defects, nb_defects + nb_m_constraints):
            npt.assert_array_almost_equal(g, 0)
