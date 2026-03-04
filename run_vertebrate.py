"""
This script aims to reposition a torque actuated vertebrate.
"""

import casadi as cas
import numpy as np

from socp import (
    Vertebrate,
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


def run_vertebrate(
    dynamics_transcription,
    discretization_method,
):

    ocp_example = Vertebrate()

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
        show_online_optim=False,
        save_path_suffix="",
    )

    data_saved = save_results(w_opt, ocp, save_path, ocp_example.n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    q_mean = data_saved["states_opt_mean"][ocp["ocp_example"].model.q_indices, :]
    time_vector = data_saved["time_vector"]
    # ocp_example.model.animate(q_mean, time_vector)


if __name__ == "__main__":

    # DirectCollocationPolynomial - NoiseDiscretization -> OK :D
    dynamics_transcription = DirectCollocationPolynomial()
    discretization_method = NoiseDiscretization(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)

    # DirectCollocationPolynomial - MeanAndCovariance -> OK :D
    dynamics_transcription = DirectCollocationPolynomial()
    discretization_method = MeanAndCovariance(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)

    # DirectMultipleShooting - NoiseDiscretization -> OK :D
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = NoiseDiscretization(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)

    # DirectMultipleShooting - MeanAndCovariance -> OK :D
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = MeanAndCovariance(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)

    # DirectCollocationTrapezoidal - NoiseDiscretization -> OK :D
    dynamics_transcription = DirectCollocationTrapezoidal()
    discretization_method = NoiseDiscretization(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)

    # DirectCollocationTrapezoidal - MeanAndCovariance -> OK :D
    dynamics_transcription = DirectCollocationTrapezoidal()
    discretization_method = MeanAndCovariance(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)

    # Variational - NoiseDiscretization -> Seems OK, but large error :D
    dynamics_transcription = Variational()
    discretization_method = NoiseDiscretization(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)

    # Variational - MeanAndCovariance -> Does not exist

    # VariationalPolynomial - NoiseDiscretization -> OK :D
    dynamics_transcription = VariationalPolynomial(order=5)
    discretization_method = NoiseDiscretization(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)

    # VariationalPolynomial - MeanAndCovariance -> OK :D
    dynamics_transcription = VariationalPolynomial(order=5)
    discretization_method = MeanAndCovariance(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)
