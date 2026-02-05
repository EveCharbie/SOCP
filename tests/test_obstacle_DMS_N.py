import numpy as np
import numpy.testing as npt
import casadi as cas

from socp import (
    ObstacleAvoidance,
    DirectMultipleShooting,
    NoiseDiscretization,
    prepare_ocp,
    solve_ocp,
)
from socp.utils import is_semi_definite_positive


def test_solve_DMS_N():

    # DirectMultipleShooting - NoiseDiscretization -> OK :D
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = NoiseDiscretization(dynamics_transcription)

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
    assert nb_variables == 1721
    nb_constraints = ocp["g"].shape[0]
    assert nb_constraints == 2437
    nb_non_zero_elem_in_grad_f = grad_f_func(ocp["w"]).nnz()
    assert nb_non_zero_elem_in_grad_f == 1721
    nb_non_zero_elem_in_grad_g = grad_g_func(ocp["w"]).nnz()
    assert nb_non_zero_elem_in_grad_g == 15050
    nb_iterations = solver.stats()["iter_count"]
    npt.assert_equal(nb_iterations, 918)

    cost_function = cas.Function("cost_function", [ocp["w"]], [ocp["j"]])
    optimal_cost = float(cost_function(w_opt))
    npt.assert_almost_equal(optimal_cost, 1.6897254628504117, decimal=5)

    npt.assert_almost_equal(np.sum(w_opt), 496.45680267939565, decimal=5)

    variable_opt = ocp["discretization_method"].Variables(
        ocp["ocp_example"].n_shooting,
        ocp["dynamics_transcription"].nb_collocation_points,
        ocp["ocp_example"].model.state_indices,
        ocp["ocp_example"].model.control_indices,
        ocp["ocp_example"].model.nb_random,
        ocp["discretization_method"].with_helper_matrix,
    )
    variable_opt.set_from_vector(w_opt, only_has_symbolics=True, qdot_variables_skipped=False)
