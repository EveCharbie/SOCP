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
        linear_solver="mumps",  # TODO: change back to ma57
        pre_optim_plot=False,
        show_online_optim=False,
        save_path_suffix="",
    )

    data_saved = save_results(w_opt, ocp, save_path, ocp_example.n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    q_mean = data_saved["states_opt_mean"][ocp["ocp_example"].model.q_indices, :]
    time_vector = data_saved["time_vector"]
    ocp_example.model.animate(q_mean, time_vector)

    #
    # cov_matrix_0 = data_saved["variable_opt"].get_cov_matrix(0)
    # m_matrix_0 = data_saved["variable_opt"].get_m_matrix(0)
    #
    # dGdx, dFdz, dGdz, dFdw, dGdw = dynamics_transcription.jacobian_funcs(
    #     data_saved["variable_opt"].get_time(),
    #     data_saved["variable_opt"].get_state("q", 0),
    #     data_saved["variable_opt"].get_state("q", 1),
    #     cas.vertcat(
    #         data_saved["variable_opt"].get_state("q", 0),
    #         (data_saved["variable_opt"].get_state("q", 0) + data_saved["variable_opt"].get_state("q", 1)) / 2,
    #         data_saved["variable_opt"].get_state("q", 1),
    #     ),
    #     data_saved["variable_opt"].get_controls(0),
    #     data_saved["variable_opt"].get_controls(1),
    #     0,
    #     0,
    # )
    #
    # ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))

if __name__ == "__main__":

    # # DirectCollocationPolynomial - NoiseDiscretization ->
    # dynamics_transcription = DirectCollocationPolynomial()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_vertebrate(dynamics_transcription, discretization_method)

    # # DirectCollocationPolynomial - MeanAndCovariance ->
    # dynamics_transcription = DirectCollocationPolynomial()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_vertebrate(dynamics_transcription, discretization_method)

    # DirectMultipleShooting - NoiseDiscretization -> OK :D
    dynamics_transcription = DirectMultipleShooting()
    discretization_method = NoiseDiscretization(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)

    # # DirectMultipleShooting - MeanAndCovariance ->
    # dynamics_transcription = DirectMultipleShooting()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_vertebrate(dynamics_transcription, discretization_method)

    # # DirectCollocationTrapezoidal - NoiseDiscretization -> OK :D
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_vertebrate(dynamics_transcription, discretization_method)

    # # DirectCollocationTrapezoidal - MeanAndCovariance ->
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_vertebrate(dynamics_transcription, discretization_method, with_lbq_bound=True)

    # # Variational - NoiseDiscretization -> Dynamics is really bad !
    # dynamics_transcription = Variational()
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_vertebrate(dynamics_transcription, discretization_method)

    # # Variational - MeanAndCovariance ->  ?? To be verified the Cov = 0 [1, n_shooting+1]
    # dynamics_transcription = Variational()
    # discretization_method = MeanAndCovariance(dynamics_transcription)
    # run_vertebrate(dynamics_transcription, discretization_method)

    # # VariationalPolynomial - NoiseDiscretization -> Did not converge need to see if the L, p, and F are OK ?
    # dynamics_transcription = VariationalPolynomial(order=5)
    # discretization_method = NoiseDiscretization(dynamics_transcription)
    # run_vertebrate(dynamics_transcription, discretization_method)

    # VariationalPolynomial - MeanAndCovariance ->
    dynamics_transcription = VariationalPolynomial(order=5)
    discretization_method = MeanAndCovariance(dynamics_transcription)
    run_vertebrate(dynamics_transcription, discretization_method)
