"""
This script aims to generate a planar backward somersault.
The model is torque driven and has 7 degrees of freedom: 3 unactuated (2 root translations and one rotation) and
4 actuated (knee, hip, shoulder, neck).
There is a sensory feedback on the joint angles and velocity and head angle and velocity (vestibular).
The feedback is directly added to the torque actuation, without delay.
This example reproduced the one from Charbonneau & al. 2026.
"""

import casadi as cas

from socp import (
    Somersault,
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


def run_somersault(
    dynamics_transcription,
    discretization_method,
):

    ocp_example = Somersault()

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
        pre_optim_plot=False,
        show_online_optim=True,
        save_path_suffix="",
    )

    data_saved = save_results(w_opt, ocp, save_path, ocp_example.n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))


if __name__ == "__main__":

    # # DirectCollocationPolynomial - NoiseDiscretization ->
    # dynamics_transcription = DirectCollocationPolynomial()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
    #
    # # DirectCollocationPolynomial - MeanAndCovariance ->
    # dynamics_transcription = DirectCollocationPolynomial()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)

    # DirectMultipleShooting - NoiseDiscretization ->
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = NoiseDiscretization(dynamics_transcription)
    run_somersault(dynamics_transcription, discretization_method)

    # # DirectMultipleShooting - MeanAndCovariance ->
    # dynamics_transcription = DirectMultipleShooting()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
    #
    # # DirectCollocationTrapezoidal - NoiseDiscretization ->
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
    #
    # # DirectCollocationTrapezoidal - MeanAndCovariance ->
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method, with_lbq_bound=True)
    #
    # # Variational - NoiseDiscretization ->
    # dynamics_transcription = Variational()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
    #
    # # Variational - MeanAndCovariance ->
    # dynamics_transcription = Variational()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
    #
    # # VariationalPolynomial - NoiseDiscretization ->
    # dynamics_transcription = VariationalPolynomial(order=5)
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
    #
    # # VariationalPolynomial - MeanAndCovariance ->
    # dynamics_transcription = VariationalPolynomial(order=5)
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_somersault(dynamics_transcription, discretization_method)
