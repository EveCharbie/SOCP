"""
This script aims to reproduce the results from Gillis & Deihl 2013.
So we use a similar warm-starting procedure.
"""

import casadi as cas

from socp import (
    ObstacleAvoidance,
    DirectMultipleShooting,
    DirectCollocationTrapezoidal,
    DirectCollocationPolynomial,
    NoiseDiscretization,
    MeanAndCovariance,
    prepare_ocp,
    solve_ocp,
    save_results,
    get_the_save_path,
)


def run_obstacle_avoidance():
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
        pre_optim_plot=True,
        show_online_optim=False,
    )
    save_path = get_the_save_path(solver, tol, ocp_example, dynamics_transcription, discretization_method).replace(
        ".pkl", "_not_robust.pkl"
    )
    data_saved = save_results(w_opt, ocp, save_path, n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))

    # # ---
    # import numpy as np
    # variable_opt = ocp["discretization_method"].Variables(
    #     ocp["ocp_example"].n_shooting,
    #     ocp["dynamics_transcription"].nb_collocation_points,
    #     ocp["ocp_example"].model.state_indices,
    #     ocp["ocp_example"].model.control_indices,
    #     ocp["discretization_method"].with_cholesky,
    #     ocp["discretization_method"].with_helper_matrix,
    # )
    # variable_opt.set_from_vector(w_opt, only_has_symbolics=True)
    #
    # def is_semi_definite_positive(matrix: np.array) -> bool:
    #     try:
    #         np.linalg.cholesky(matrix + 1e-12 * np.eye(matrix.shape[0]))
    #         return True
    #     except np.linalg.LinAlgError:
    #         return False
    #
    # for i_node in range(ocp["n_shooting"]):
    #     cov_matrix_0 = variable_opt.get_cov_matrix(i_node)
    #     cov_matrix_1 = variable_opt.get_cov_matrix(i_node + 1)
    #     m_matrix = variable_opt.get_m_matrix(i_node)
    #     noises_numerical = np.diag(np.array([1, 1]))
    #     dGdx, dGdz, dGdw, dFdz = ocp["dynamics_transcription"].jacobian_funcs(
    #         variable_opt.get_time(),
    #         variable_opt.get_states(i_node),
    #         variable_opt.get_collocation_points(i_node),
    #         variable_opt.get_controls(i_node),
    #         np.array([1, 1]),
    #     )
    #
    #     cov_integrated = m_matrix @ (dGdx @ cov_matrix_0 @ dGdx.T + dGdw @ noises_numerical @ dGdw.T) @ m_matrix.T
    #     print(f"\n COV {i_node} --- ", np.max(np.abs(cov_matrix_1 - cov_integrated)), f"----------- {is_semi_definite_positive(cov_matrix_0)} / {is_semi_definite_positive(cov_matrix_1)}")
    #
    #     m_constraint = dFdz.T - dGdz.T @ m_matrix.T
    #     print(f"\n M --- {i_node} ", np.max(np.abs(m_constraint)))
    # ---


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
        linear_solver="ma57",  # TODO change to "ma97" if available
        pre_optim_plot=False,
        show_online_optim=False,
    )
    save_path = get_the_save_path(solver, tol, ocp_example, dynamics_transcription, discretization_method).replace(
        ".pkl", "_robustified.pkl"
    )
    data_saved = save_results(w_opt, ocp, save_path, n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))


if __name__ == "__main__":
    run_obstacle_avoidance()
