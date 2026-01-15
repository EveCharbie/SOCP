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
    ocp_example = ObstacleAvoidance(is_robustified=True)

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
        linear_solver="ma57",  # TODO change to "ma97" if available
        pre_optim_plot=False,
        show_online_optim=True,
    )
    save_path = get_the_save_path(solver, tol, ocp_example, dynamics_transcription, discretization_method).replace(".pkl", "_not_robust.pkl")
    data_saved = save_results(w_opt, ocp, save_path, n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))


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
        show_online_optim=True,
    )
    save_path = get_the_save_path(solver, tol, ocp_example, dynamics_transcription, discretization_method).replace(
        ".pkl", "_robustified.pkl")
    data_saved = save_results(w_opt, ocp, save_path, n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))

if __name__ == "__main__":
    run_obstacle_avoidance()
