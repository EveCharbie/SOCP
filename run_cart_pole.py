"""
This script aims to invert a pole mounted on a cart.
The cart can move sideways and the pole can rotate around its base without actuation. 
"""

import casadi as cas

from socp import (
    CartPole,
    DirectMultipleShooting,
    DirectCollocationTrapezoidal,
    DirectCollocationPolynomial,
    Variational,
    VariationalPolynomial,
    NoiseDiscretization,
    MeanAndCovariance,
    prepare_ocp,
    solve_ocp,
    save_results,
    get_the_save_path,
)


def run_cart_pole(
    dynamics_transcription,
    discretization_method,
):

    ocp_example = CartPole()

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
        linear_solver="ma57",  # TODO: change back to ma57
        pre_optim_plot=True,
        show_online_optim=False,
        save_path_suffix="",
    )

    data_saved = save_results(w_opt, ocp, save_path, ocp_example.n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))

if __name__ == "__main__":

    # # DirectCollocationPolynomial - NoiseDiscretization ->
    # dynamics_transcription = DirectCollocationPolynomial()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_cart_pole(dynamics_transcription, discretization_method)

    # # DirectCollocationPolynomial - MeanAndCovariance ->
    # dynamics_transcription = DirectCollocationPolynomial()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_cart_pole(dynamics_transcription, discretization_method)

    # # DirectMultipleShooting - NoiseDiscretization -> OK :D
    # dynamics_transcription = DirectMultipleShooting()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_cart_pole(dynamics_transcription, discretization_method)

    # # DirectMultipleShooting - MeanAndCovariance ->
    # dynamics_transcription = DirectMultipleShooting()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_cart_pole(dynamics_transcription, discretization_method)

    # # DirectCollocationTrapezoidal - NoiseDiscretization -> OK :D
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_cart_pole(dynamics_transcription, discretization_method)

    # # DirectCollocationTrapezoidal - MeanAndCovariance ->
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_cart_pole(dynamics_transcription, discretization_method, with_lbq_bound=True)

    # # Variational - NoiseDiscretization ->
    # dynamics_transcription = Variational()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_cart_pole(dynamics_transcription, discretization_method)

    # # Variational - MeanAndCovariance ->  ?? To be verified the Cov = 0 [1, n_shooting+1]
    # dynamics_transcription = Variational()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_cart_pole(dynamics_transcription, discretization_method)

    # # VariationalPolynomial - NoiseDiscretization ->
    # dynamics_transcription = VariationalPolynomial(order=5)
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_cart_pole(dynamics_transcription, discretization_method)

    # VariationalPolynomial - MeanAndCovariance ->
    dynamics_transcription = VariationalPolynomial(order=5)
    discretization_method = MeanAndCovariance(dynamics_transcription)
    run_cart_pole(dynamics_transcription, discretization_method)
