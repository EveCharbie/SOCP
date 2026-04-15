"""
This script aims to reposition a torque actuated arm.
"""

from socp import (
    VertebrateArm,
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
)


def run_vertebrate(
    dynamics_transcription,
    discretization_method,
    nb_random: int = 15,
):

    ocp_example = VertebrateArm(nb_random=nb_random)

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
        plot_solution=False,
    )

    data_saved = save_results(w_opt, ocp, g_without_bounds_at_init, save_path, ocp_example.n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))

    # q_mean = data_saved["states_opt_mean"][ocp["ocp_example"].model.q_indices, :]
    # time_vector = data_saved["time_vector"]
    # ocp_example.model.animate(q_mean, time_vector)


if __name__ == "__main__":


    ### --- 1. RUN THE TRANSCRIPTION COMPARISON ANALYSIS --- ###

    dynamics_transcription = DirectCollocationPolynomial(order=5)
    discretization_method = Deterministic(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method, nb_random=1)

    dynamics_transcription = DirectMultipleShooting()
    discretization_method = Deterministic(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method, nb_random=1)

    dynamics_transcription = DirectCollocationTrapezoidal()
    discretization_method = Deterministic(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method, nb_random=1)

    dynamics_transcription = Variational()
    discretization_method = Deterministic(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method, nb_random=1)

    dynamics_transcription = VariationalPolynomial(order=5)
    discretization_method = Deterministic(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method, nb_random=1)

    # DirectCollocationPolynomial - MeanAndCovariance -> OK :D
    dynamics_transcription = DirectCollocationPolynomial(order=5)
    discretization_method = MeanAndCovariance(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)

    # DirectMultipleShooting - MeanAndCovariance -> OK :D
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = MeanAndCovariance(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)

    # DirectCollocationTrapezoidal - MeanAndCovariance -> OK :D
    dynamics_transcription = DirectCollocationTrapezoidal()
    discretization_method = MeanAndCovariance(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)

    # VariationalPolynomial - MeanAndCovariance -> OK :D
    dynamics_transcription = VariationalPolynomial(order=5)
    discretization_method = MeanAndCovariance(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)


    ### --- 2. RUN THE SENSITIVITY ANALYSIS --- ###
    for this_nb_random in [30, 35, 40, 45, 50]: #5, 10, 15, 20, 25, 30, 35, 40, 45, 50

        dynamics_transcription = DirectCollocationPolynomial(order=5)
        discretization_method = NoiseDiscretization(dynamics_transcription)
        run_vertebrate(dynamics_transcription, discretization_method, nb_random=this_nb_random)

        dynamics_transcription = DirectMultipleShooting()
        discretization_method = NoiseDiscretization(dynamics_transcription)
        run_vertebrate(dynamics_transcription, discretization_method, nb_random=this_nb_random)

        dynamics_transcription = DirectCollocationTrapezoidal()
        discretization_method = NoiseDiscretization(dynamics_transcription)
        run_vertebrate(dynamics_transcription, discretization_method, nb_random=this_nb_random)

        dynamics_transcription = Variational()
        discretization_method = NoiseDiscretization(dynamics_transcription)
        run_vertebrate(dynamics_transcription, discretization_method, nb_random=this_nb_random)

        dynamics_transcription = VariationalPolynomial(order=5)
        discretization_method = NoiseDiscretization(dynamics_transcription)
        run_vertebrate(dynamics_transcription, discretization_method, nb_random=this_nb_random)
