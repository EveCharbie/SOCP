"""
This script aims to reproduce the results from Gillis & Deihl 2013.
So we use a similar warm-starting procedure.
"""

import casadi as cas

from socp import (
    Parabola,
    Variational,
    VariationalPolynomial,
    NoiseDiscretization,
    MeanAndCovariance,
    prepare_ocp,
    solve_ocp,
    save_results,
)


def run_parabola(
    dynamics_transcription,
    discretization_method,
):

    ocp_example = Parabola()

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
        linear_solver="mumps",  # TODO: change back to ma57
        pre_optim_plot=True,
        show_online_optim=False,
        save_path_suffix="",
    )

    data_saved = save_results(w_opt, ocp, save_path, ocp_example.n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))


if __name__ == "__main__":

    # # Variational - NoiseDiscretization ->
    # dynamics_transcription = Variational()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_parabola(
    #     dynamics_transcription,
    #     discretization_method,
    #     with_lbq_bound=True
    # )

    # # Variational - MeanAndCovariance
    # dynamics_transcription = Variational()
    # discretization_method = MeanAndCovariance(dynamics_transcription, with_helper_matrix=True)
    # run_parabola(
    #     dynamics_transcription,
    #     discretization_method,
    #     with_lbq_bound=True
    # )

    # VariationalPolynomial - NoiseDiscretization ->
    dynamics_transcription = VariationalPolynomial(order=3)
    discretization_method = NoiseDiscretization(dynamics_transcription)
    run_parabola(dynamics_transcription, discretization_method)
