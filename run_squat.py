"""
This script aims to generate a 3D single leg squat (sit-to-stand like) with 3 rigid contacts between the ground and foot.
The model is torque-derivative driven and has 12 degrees of freedom: 6 unactuated (free floating base) and
6 actuated (hip 3, knee 1, ankle 2).
There is a sensory feedback on the joint angles, trunk orientation (vestibular), and foot pressure.
The feedback is directly added to the torque-derivative actuation, without delay ? TODO
"""

import casadi as cas
import matplotlib.pyplot as plt

from socp import (
    Squat,
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


def run_squat(
    dynamics_transcription,
    discretization_method,
    nb_random: int = 10,
):


    # --- First run : Deterministic --- #
    ocp_example = Squat(nb_random=1)
    ocp_example.initial_states_to_impose = ["qdot"]

    # Prepare the problem
    ocp = prepare_ocp(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        discretization_method=Deterministic(dynamics_transcription),
    )

    # Solve the problem
    w_opt, solver, grad_f_func, grad_g_func, save_path, g_without_bounds_at_init = solve_ocp(
        ocp,
        ocp_example=ocp_example,
        hessian_approximation="exact",  # or "limited-memory",
        linear_solver="ma57",  # TODO: change back to ma57
        pre_optim_plot=False,
        show_online_optim=False,  # Cannot plot the deterministic, because I cannot delete the OnlineCallback
        save_path_suffix="",
    )

    data_saved = save_results(w_opt, ocp, g_without_bounds_at_init, save_path, ocp_example.n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    plt.close("all")





    ocp_example = Squat(nb_random=nb_random)

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
        show_online_optim=False,
        save_path_suffix="",
    )

    data_saved = save_results(w_opt, ocp, g_without_bounds_at_init, save_path, ocp_example.n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))


if __name__ == "__main__":

    # Deterministic
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = Deterministic(dynamics_transcription)
    run_squat(dynamics_transcription, discretization_method, nb_random=1)

    # # DirectCollocationPolynomial - NoiseDiscretization ->
    # dynamics_transcription = DirectCollocationPolynomial()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_squat(dynamics_transcription, discretization_method)
    #
    # # DirectCollocationPolynomial - MeanAndCovariance ->
    # dynamics_transcription = DirectCollocationPolynomial()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_squat(dynamics_transcription, discretization_method)

    # DirectMultipleShooting - NoiseDiscretization ->
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = NoiseDiscretization(dynamics_transcription)
    run_squat(dynamics_transcription, discretization_method)

    # # DirectMultipleShooting - MeanAndCovariance ->
    # dynamics_transcription = DirectMultipleShooting()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_squat(dynamics_transcription, discretization_method)
    #
    # # DirectCollocationTrapezoidal - NoiseDiscretization ->
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_squat(dynamics_transcription, discretization_method)
    #
    # # DirectCollocationTrapezoidal - MeanAndCovariance ->
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_squat(dynamics_transcription, discretization_method, with_lbq_bound=True)
    #
    # # Variational - NoiseDiscretization ->
    # dynamics_transcription = Variational()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_squat(dynamics_transcription, discretization_method)
    #
    # # Variational - MeanAndCovariance ->
    # dynamics_transcription = Variational()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_squat(dynamics_transcription, discretization_method)
    #
    # # VariationalPolynomial - NoiseDiscretization ->
    # dynamics_transcription = VariationalPolynomial(order=5)
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_squat(dynamics_transcription, discretization_method)
    #
    # # VariationalPolynomial - MeanAndCovariance ->
    # dynamics_transcription = VariationalPolynomial(order=5)
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_squat(dynamics_transcription, discretization_method)
