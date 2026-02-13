import numpy as np
import numpy.testing as npt
import casadi as cas

from socp import (
    ObstacleAvoidance,
    DirectMultipleShooting,
    MeanAndCovariance,
    prepare_ocp,
    solve_ocp,
)
from socp.utils import is_semi_definite_positive


def test_solve_DMS_MAC():

    # DirectMultipleShooting - NoiseDiscretization -> OK :D
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = MeanAndCovariance(dynamics_transcription)

    # --- Run the problem a first time without robustification of the constraint --- #
    ocp_example = ObstacleAvoidance(is_robustified=False)

    # Prepare the problem
    ocp = prepare_ocp(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        discretization_method=discretization_method,
    )

    # Solve the problem
    w_opt, solver, grad_f_func, grad_g_func, save_path = solve_ocp(
        ocp,
        ocp_example=ocp_example,
        hessian_approximation="exact",  # or "limited-memory",
        linear_solver="ma57",
        pre_optim_plot=False,
        show_online_optim=False,
        save_path_suffix="not_robust",
        plot_solution=False,
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
    w_opt, solver, grad_f_func, grad_g_func, save_path = solve_ocp(
        ocp,
        ocp_example=ocp_example,
        hessian_approximation="exact",  # or "limited-memory",
        linear_solver="ma57",
        pre_optim_plot=False,
        show_online_optim=False,
        save_path_suffix="robustified",
        plot_solution=False,
    )

    # --- Test the results --- #
    nb_variables = ocp["w"].shape[0]
    assert nb_variables == 901
    nb_constraints = ocp["g"].shape[0]
    assert nb_constraints == 905
    nb_non_zero_elem_in_grad_f = grad_f_func(ocp["w"]).nnz()
    assert nb_non_zero_elem_in_grad_f == 901
    nb_non_zero_elem_in_grad_g = grad_g_func(ocp["w"]).nnz()
    assert nb_non_zero_elem_in_grad_g == 17191
    nb_iterations = solver.stats()["iter_count"]
    npt.assert_equal(nb_iterations, 68)

    cost_function = cas.Function("cost_function", [ocp["w"]], [ocp["j"]])
    optimal_cost = float(cost_function(w_opt))
    npt.assert_almost_equal(optimal_cost, 1.6540606466485324, decimal=5)

    npt.assert_almost_equal(np.sum(w_opt), 76.81560747713468, decimal=5)

    variable_opt = ocp["discretization_method"].Variables(
        ocp["ocp_example"].n_shooting,
        ocp["dynamics_transcription"].nb_collocation_points,
        ocp["dynamics_transcription"].nb_m_points,
        ocp["ocp_example"].model.state_indices,
        ocp["ocp_example"].model.control_indices,
        ocp["ocp_example"].model.nb_random,
    )
    variable_opt.set_from_vector(w_opt, only_has_symbolics=True, qdot_variables_skipped=False)

    for i_node in range(ocp["n_shooting"]):
        cov_matrix_0 = variable_opt.get_cov_matrix(i_node)
        cov_matrix_1 = variable_opt.get_cov_matrix(i_node + 1)
        noises_numerical = np.diag(np.array([1, 1]))
        dFdx, dFdw = ocp["dynamics_transcription"].jacobian_funcs(
            variable_opt.get_time(),
            variable_opt.get_states(i_node),
            variable_opt.get_controls(i_node),
            np.array([1, 1]),
        )

        assert is_semi_definite_positive(cov_matrix_0)
        assert is_semi_definite_positive(cov_matrix_1)

        cov_integrated = dFdx @ cov_matrix_0 @ dFdx.T + dFdw @ noises_numerical @ dFdw.T
        npt.assert_array_less(np.max(np.abs(cov_matrix_1 - cov_integrated)), 1e-5)
