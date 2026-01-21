import numpy as np
import numpy.testing as npt
import casadi as cas

from socp import (
    ObstacleAvoidance,
    DirectCollocationPolynomial,
    MeanAndCovariance,
    prepare_ocp,
    solve_ocp,
    save_results,
    get_the_save_path,
)


def is_semi_definite_positive(matrix: np.array) -> bool:
    try:
        np.linalg.cholesky(matrix + 1e-12 * np.eye(matrix.shape[0]))
        return True
    except np.linalg.LinAlgError:
        return False


def test_solve_DCP_MAC():

    tol = 1e-6
    n_simulations = 100

    dynamics_transcription = DirectCollocationPolynomial()
    discretization_method = MeanAndCovariance(dynamics_transcription, with_cholesky=False, with_helper_matrix=True)

    # --- Run the problem a first time without robustification of the constraint --- #
    ocp_example = ObstacleAvoidance(is_robustified=False)

    # Prepare the problem
    ocp = prepare_ocp(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        discretization_method=discretization_method,
    )

    # Solve the problem
    w_opt, solver, grad_f_func, grad_g_func = solve_ocp(
        ocp,
        ocp_example=ocp_example,
        hessian_approximation="exact",  # or "limited-memory",
        linear_solver="ma57",
        pre_optim_plot=False,
        show_online_optim=False,
    )

    # --- Run the problem a second time robustified --- #
    ocp_example = ObstacleAvoidance(is_robustified=True)

    # Prepare the problem
    ocp = prepare_ocp(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        discretization_method=discretization_method,
    )
    # Warm-start
    ocp["w0"] = cas.DM(w_opt)

    # Solve the problem
    w_opt, solver, grad_f_func, grad_g_func = solve_ocp(
        ocp,
        ocp_example=ocp_example,
        hessian_approximation="exact",  # or "limited-memory",
        linear_solver="ma57",
        pre_optim_plot=False,
        show_online_optim=False,
    )

    # --- Test the results --- #
    nb_variables = ocp["w"].shape[0]
    assert nb_variables == 5701
    nb_constraints = ocp["g"].shape[0]
    assert nb_constraints == 5705
    nb_non_zero_elem_in_grad_f = grad_f_func(ocp["w"]).nnz()
    assert nb_non_zero_elem_in_grad_f == 5701
    nb_non_zero_elem_in_grad_g = grad_g_func(ocp["w"]).nnz()
    assert nb_non_zero_elem_in_grad_g == 67431
    nb_iterations = solver.stats()["iter_count"]
    npt.assert_equal(nb_iterations, 270)

    cost_function = cas.Function("cost_function", [ocp["w"]], [ocp["j"]])
    optimal_cost = float(cost_function(w_opt))
    npt.assert_almost_equal(optimal_cost, 8.33022993497717, decimal=5)

    npt.assert_almost_equal(np.sum(w_opt), 386.97682860321925, decimal=5)

    variable_opt = ocp["discretization_method"].Variables(
        ocp["ocp_example"].n_shooting,
        ocp["dynamics_transcription"].nb_collocation_points,
        ocp["ocp_example"].model.state_indices,
        ocp["ocp_example"].model.control_indices,
        ocp["discretization_method"].with_cholesky,
        ocp["discretization_method"].with_helper_matrix,
    )
    variable_opt.set_from_vector(w_opt, only_has_symbolics=True)

    for i_node in range(ocp["n_shooting"]):
        cov_matrix_0 = variable_opt.get_cov_matrix(i_node)
        cov_matrix_1 = variable_opt.get_cov_matrix(i_node + 1)
        m_matrix = variable_opt.get_m_matrix(i_node)
        noises_numerical = np.diag(np.array([1, 1]))
        dGdx, dGdz, dGdw, dFdz = ocp["dynamics_transcription"].jacobian_funcs(
            variable_opt.get_time(),
            variable_opt.get_states(i_node),
            variable_opt.get_collocation_points(i_node),
            variable_opt.get_controls(i_node),
            np.array([1, 1]),
        )

        assert is_semi_definite_positive(cov_matrix_0)
        assert is_semi_definite_positive(cov_matrix_1)

        cov_integrated = m_matrix @ (dGdx @ cov_matrix_0 @ dGdx.T + dGdw @ noises_numerical @ dGdw.T) @ m_matrix.T
        npt.assert_array_less(np.max(np.abs(cov_matrix_1 - cov_integrated)), 1e-5)

        m_constraint = dFdz.T - dGdz.T @ m_matrix.T
        npt.assert_array_less(np.max(np.abs(m_constraint)), 1e-5)



