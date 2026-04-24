"""
This script aims to generate an planar arm reaching movement from a start target to an end target.
The model is muscle driven and has 2 degrees of freedom (shoulder and elbow) and 6 muscles (4 monoarticular and 2 biarticular).
There is a sensory feedback on the hand position and velocity.
The feedback is directly added to the muscle excitation, which has an activation delay of 150ms.
This example reproduced the one from Van Wouwe & al. 2022.
"""

import casadi as cas

from socp import (
    ArmReaching,
    DirectMultipleShooting,
    DirectCollocationTrapezoidal,
    DirectCollocationPolynomial,
    Variational,
    VariationalPolynomial,
    NoiseDiscretization,
    MeanAndCovariance,
    Deterministic,
    prepare_ocp,
    solve_ocp,
    save_results,
    get_the_save_path,
)


def run_arm_reaching(
    dynamics_transcription,
    discretization_method,
    nb_random: int = 10,
):

    ocp_example = ArmReaching(nb_random=nb_random)

    # Prepare the problem
    ocp = prepare_ocp(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        discretization_method=discretization_method,
    )

    # Solve the problem
    w_opt, solver, grad_f_func, grad_g_func, save_path, g_without_bounds_at_init = solve_ocp(
        ocp,
        ocp_example=ocp_example,
        hessian_approximation="exact",  # or "limited-memory",
        linear_solver="ma57",  # TODO: change back to ma57
        pre_optim_plot=False,
        show_online_optim=True,
        save_path_suffix="",
    )

    data_saved = save_results(w_opt, ocp, g_without_bounds_at_init, save_path, ocp_example.n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))


if __name__ == "__main__":

    # # Deterministic
    # dynamics_transcription = DirectMultipleShooting()
    # discretization_method = Deterministic(dynamics_transcription)
    # run_arm_reaching(dynamics_transcription, discretization_method, nb_random=1)

    # # DirectCollocationPolynomial - NoiseDiscretization ->
    # dynamics_transcription = DirectCollocationPolynomial()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_arm_reaching(dynamics_transcription, discretization_method)
    #
    # # DirectCollocationPolynomial - MeanAndCovariance ->
    # dynamics_transcription = DirectCollocationPolynomial()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_arm_reaching(dynamics_transcription, discretization_method)

    # # DirectMultipleShooting - NoiseDiscretization -> DVG (mean reach target) + reintegration montre que dyn const pas respectées !?!
    # dynamics_transcription = DirectMultipleShooting()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_arm_reaching(dynamics_transcription, discretization_method, nb_random=2)

    # DirectMultipleShooting - MeanAndCovariance ->
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = MeanAndCovariance(dynamics_transcription)
    run_arm_reaching(dynamics_transcription, discretization_method)
    #
    # # DirectCollocationTrapezoidal - NoiseDiscretization ->
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_arm_reaching(dynamics_transcription, discretization_method)
    #
    # # DirectCollocationTrapezoidal - MeanAndCovariance ->
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_arm_reaching(dynamics_transcription, discretization_method)
    #
    # # Variational - NoiseDiscretization ->
    # dynamics_transcription = Variational()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_arm_reaching(dynamics_transcription, discretization_method)
    #
    # # Variational - MeanAndCovariance ->
    # dynamics_transcription = Variational()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_arm_reaching(dynamics_transcription, discretization_method)
    #
    # # VariationalPolynomial - NoiseDiscretization ->
    # dynamics_transcription = VariationalPolynomial(order=5)
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_arm_reaching(dynamics_transcription, discretization_method)
    #
    # # VariationalPolynomial - MeanAndCovariance ->
    # dynamics_transcription = VariationalPolynomial(order=5)
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_arm_reaching(dynamics_transcription, discretization_method)
