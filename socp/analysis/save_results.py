from typing import Any
import casadi as cas
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .reintegrate_solution import reintegrate
from .estimate_covariance import estimate_covariance
from ..transcriptions.variational import Variational
from ..transcriptions.variational_polynomial import VariationalPolynomial


def save_results(
    w_opt: cas.DM,
    ocp: dict[str, Any],
    save_path: str,
    n_simulations: int,
    solver: Any,
    grad_f_func: cas.Function,
    grad_g_func: cas.Function,
) -> dict[str, Any]:

    # Solving info
    computational_time = solver.stats()["t_proc_total"]
    nb_iterations = solver.stats()["iter_count"]
    nb_variables = ocp["w"].shape[0]
    nb_constraints = ocp["g"].shape[0]
    nb_non_zero_elem_in_grad_f = grad_f_func(ocp["w"]).nnz()
    nb_non_zero_elem_in_grad_g = grad_g_func(ocp["w"]).nnz()
    cost_function = cas.Function("cost_function", [ocp["w"]], [ocp["j"]])
    optimal_cost = float(cost_function(w_opt))

    if isinstance(ocp["dynamics_transcription"], (Variational, VariationalPolynomial)):
        qdot_variables_skipped = True
    else:
        qdot_variables_skipped = False

    # Get optimization variables
    variable_opt = ocp["discretization_method"].Variables(
        ocp["ocp_example"].n_shooting,
        ocp["dynamics_transcription"].nb_collocation_points,
        ocp["dynamics_transcription"].nb_m_points,
        ocp["ocp_example"].model.state_indices,
        ocp["ocp_example"].model.control_indices,
        ocp["ocp_example"].model.nb_random,
    )
    variable_opt.set_from_vector(w_opt, only_has_symbolics=True, qdot_variables_skipped=qdot_variables_skipped)

    variable_init = ocp["discretization_method"].Variables(
        ocp["ocp_example"].n_shooting,
        ocp["dynamics_transcription"].nb_collocation_points,
        ocp["dynamics_transcription"].nb_m_points,
        ocp["ocp_example"].model.state_indices,
        ocp["ocp_example"].model.control_indices,
        ocp["ocp_example"].model.nb_random,
    )
    variable_init.set_from_vector(ocp["w0"], only_has_symbolics=True, qdot_variables_skipped=qdot_variables_skipped)
    states_init_array = variable_init.get_states_array()
    controls_init_array = variable_init.get_controls_array()

    variable_lb = ocp["discretization_method"].Variables(
        ocp["ocp_example"].n_shooting,
        ocp["dynamics_transcription"].nb_collocation_points,
        ocp["dynamics_transcription"].nb_m_points,
        ocp["ocp_example"].model.state_indices,
        ocp["ocp_example"].model.control_indices,
        ocp["ocp_example"].model.nb_random,
    )
    variable_lb.set_from_vector(ocp["lbw"], only_has_symbolics=True, qdot_variables_skipped=qdot_variables_skipped)

    variable_ub = ocp["discretization_method"].Variables(
        ocp["ocp_example"].n_shooting,
        ocp["dynamics_transcription"].nb_collocation_points,
        ocp["dynamics_transcription"].nb_m_points,
        ocp["ocp_example"].model.state_indices,
        ocp["ocp_example"].model.control_indices,
        ocp["ocp_example"].model.nb_random,
    )
    variable_ub.set_from_vector(ocp["ubw"], only_has_symbolics=True, qdot_variables_skipped=qdot_variables_skipped)

    time_vector = np.linspace(0, variable_opt.get_time(), ocp["n_shooting"] + 1)

    states_opt_array = variable_opt.get_states_array()
    controls_opt_array = variable_opt.get_controls_array()

    # Mean states
    states_opt_mean = np.zeros((variable_opt.nb_states, variable_opt.n_shooting + 1))
    states_opt_mean[:, :] = np.nan
    for i_node in range(variable_opt.n_shooting + 1):
        states_mean = np.array(
            ocp["discretization_method"].get_mean_states(
                variable_opt,
                i_node,
                squared=False,
            )
        ).reshape(
            -1,
        )
        states_opt_mean[: states_mean.shape[0], i_node] = states_mean

    # Reintegrate the solution
    x_simulated = reintegrate(
        time_vector=time_vector,
        states_opt_mean=states_opt_mean,
        states_opt_array=states_opt_array,
        controls_opt_array=controls_opt_array,
        ocp=ocp,
        n_simulations=n_simulations,
        save_path=save_path,
        plot_flag=True,
    )

    # Compute the simulated covariance
    x_mean_simulated = np.mean(x_simulated, axis=2)
    covariance_simulated = estimate_covariance(
        x_mean_simulated,
        x_simulated,
    )

    # --- Metrics to compare --- #
    difference_between_means = np.mean(
        np.abs(states_opt_mean - x_mean_simulated),
        axis=0,
    )
    cov_det_opt = np.zeros((ocp["n_shooting"] + 1,))
    cov_det_simulated = np.zeros((ocp["n_shooting"] + 1,))
    norm_difference_between_covs = np.zeros((ocp["n_shooting"] + 1,))
    cov_opt_array = np.zeros(
        (ocp["ocp_example"].model.nb_states, ocp["ocp_example"].model.nb_states, ocp["n_shooting"] + 1)
    )
    for i_node in range(ocp["n_shooting"] + 1):
        cov_matrix_this_time = ocp["discretization_method"].get_covariance(variable_opt, i_node, is_matrix=True)
        cov_det_opt[i_node] = np.linalg.det(cov_matrix_this_time)
        cov_opt_array[: cov_matrix_this_time.shape[0], : cov_matrix_this_time.shape[1], i_node] = cov_matrix_this_time
        cov_det_simulated[i_node] = np.linalg.det(covariance_simulated[:, :, i_node])

        norm_difference_between_covs[i_node] = np.abs(
            np.linalg.norm(cov_opt_array[:, :, i_node] - covariance_simulated[:, :, i_node], ord="fro")
        )

    difference_between_covs_det = np.abs(cov_det_opt - cov_det_simulated)


    # Plot the covariance difference
    plt.figure()
    plt.plot(cov_opt_array[0, 0, :], "--", color="tab:red")
    plt.plot(cov_opt_array[0, 1, :], "--", color="tab:green")
    plt.plot(cov_opt_array[1, 1, :], "--", color="tab:blue")
    plt.plot(cov_opt_array[1, 0, :], "--", color="tab:orange")
    plt.plot(cov_det_opt, "--k")

    plt.plot(covariance_simulated[0, 0, :], "-", color="tab:red")
    plt.plot(covariance_simulated[0, 1, :], "-", color="tab:green")
    plt.plot(covariance_simulated[1, 1, :], "-", color="tab:blue")
    plt.plot(covariance_simulated[1, 0, :], "-", color="tab:orange")
    plt.plot(cov_det_simulated, "-k")
    plt.savefig(save_path.replace(".pkl", "_cov.png"))
    plt.show()
    print("max state difference: ", np.nanmax(np.abs(states_opt_mean - x_mean_simulated)))
    print("max cov difference: ", np.nanmax(np.abs(cov_opt_array - cov_det_simulated)))
    print("max state difference: ", np.nanmax(np.abs(states_opt_mean - x_mean_simulated)) / np.max(np.abs(states_opt_mean)) * 100, "%")
    print("max cov difference: ", np.nanmax(np.abs(cov_opt_array - cov_det_simulated)) / np.max(np.abs(cov_opt_array)) * 100, "%")



    # Actually save
    data_to_save = {
        "computational_time": computational_time,
        "controls_init_array": controls_init_array,
        "controls_opt_array": controls_opt_array,
        "covariance_simulated": covariance_simulated,
        "cov_det_opt": cov_det_opt,
        "cov_det_simulated": cov_det_simulated,
        "cov_opt_array": cov_opt_array,
        "difference_between_covs_det": difference_between_covs_det,
        "difference_between_means": difference_between_means,
        "nb_constraints": nb_constraints,
        "nb_iterations": nb_iterations,
        "nb_non_zero_elem_in_grad_f": nb_non_zero_elem_in_grad_f,
        "nb_non_zero_elem_in_grad_g": nb_non_zero_elem_in_grad_g,
        "nb_variables": nb_variables,
        "norm_difference_between_covs": norm_difference_between_covs,
        "optimal_cost": optimal_cost,
        "states_init_array": states_init_array,
        "states_opt_array": states_opt_array,
        "states_opt_mean": states_opt_mean,
        "time_vector": time_vector,
        "variable_init": variable_init,
        "variable_lb": variable_lb,
        "variable_ub": variable_ub,
        "variable_opt": variable_opt,
        "w_opt": w_opt,
        "x_mean_simulated": x_mean_simulated,
        "x_simulated": x_simulated,
    }
    with open(save_path, "wb") as file:
        pickle.dump(data_to_save, file)

    return data_to_save
