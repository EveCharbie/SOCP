import numpy as np
import numpy.testing as npt
import casadi as cas

from socp import (
    ObstacleAvoidance,
    Variational,
    NoiseDiscretization,
    prepare_ocp,
    solve_ocp,
)


def test_solve_DC_N():

    dynamics_transcription = Variational()
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
        plot_solution=False,
    )

    # --- Test the results --- #
    assert solver.stats()["success"] == True
    nb_variables = ocp["w"].shape[0]
    assert nb_variables == 941
    nb_constraints = ocp["g"].shape[0]
    assert nb_constraints == 1657
    nb_non_zero_elem_in_grad_f = grad_f_func(ocp["w"]).nnz()
    assert nb_non_zero_elem_in_grad_f == 941
    nb_non_zero_elem_in_grad_g = grad_g_func(ocp["w"]).nnz()
    assert nb_non_zero_elem_in_grad_g == 9550
    nb_iterations = solver.stats()["iter_count"]
    npt.assert_equal(nb_iterations, 254)

    cost_function = cas.Function("cost_function", [ocp["w"]], [ocp["j"]])
    optimal_cost = float(cost_function(w_opt))
    npt.assert_almost_equal(optimal_cost, 1.625622545258195, decimal=5)

    npt.assert_almost_equal(np.sum(w_opt), 692.3888275785566, decimal=5)
