from typing import Any
import casadi as cas
import pickle
import numpy as np

from .reintegrate_solution import reintegrate
from .estimate_covariance import estimate_covariance


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

    # Get optimization variables
    variable_opt = ocp["discretization_method"].Variables(
        ocp["ocp_example"].n_shooting,
        ocp["dynamics_transcription"].nb_collocation_points,
        ocp["ocp_example"].model.state_indices,
        ocp["ocp_example"].model.control_indices,
        ocp["discretization_method"].with_cholesky,
        ocp["discretization_method"].with_helper_matrix,
    )
    variable_opt.set_from_vector(w_opt, only_has_symbolics=True)

    variable_init = ocp["discretization_method"].Variables(
        ocp["ocp_example"].n_shooting,
        ocp["dynamics_transcription"].nb_collocation_points,
        ocp["ocp_example"].model.state_indices,
        ocp["ocp_example"].model.control_indices,
        ocp["discretization_method"].with_cholesky,
        ocp["discretization_method"].with_helper_matrix,
    )
    variable_init.set_from_vector(ocp["w0"], only_has_symbolics=True)
    states_init_array = variable_init.get_states_array()
    controls_init_array = variable_init.get_controls_array()

    variable_lb = ocp["discretization_method"].Variables(
        ocp["ocp_example"].n_shooting,
        ocp["dynamics_transcription"].nb_collocation_points,
        ocp["ocp_example"].model.state_indices,
        ocp["ocp_example"].model.control_indices,
        ocp["discretization_method"].with_cholesky,
        ocp["discretization_method"].with_helper_matrix,
    )
    variable_lb.set_from_vector(ocp["lbw"], only_has_symbolics=True)

    variable_ub = ocp["discretization_method"].Variables(
        ocp["ocp_example"].n_shooting,
        ocp["dynamics_transcription"].nb_collocation_points,
        ocp["ocp_example"].model.state_indices,
        ocp["ocp_example"].model.control_indices,
        ocp["discretization_method"].with_cholesky,
        ocp["discretization_method"].with_helper_matrix,
    )
    variable_ub.set_from_vector(ocp["ubw"], only_has_symbolics=True)

    time_vector = np.linspace(0, variable_opt.get_time(), ocp["n_shooting"] + 1)

    states_opt_array = variable_opt.get_states_array()
    controls_opt_array = variable_opt.get_controls_array()

    # Mean states
    states_opt_mean = np.array(
        ocp["discretization_method"].get_mean_states(
            model=ocp["ocp_example"].model,
            x=states_opt_array[: ocp["ocp_example"].model.nb_states],
            squared=False,
        )
    )

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
    cov_opt_array = np.zeros(
        (ocp["ocp_example"].model.nb_states, ocp["ocp_example"].model.nb_states, ocp["n_shooting"] + 1)
    )
    for i_node in range(ocp["n_shooting"] + 1):
        cov_matrix_this_time = variable_opt.get_cov_matrix(i_node)
        cov_det_opt[i_node] = np.linalg.det(cov_matrix_this_time)
        cov_opt_array[:, :, i_node] = cov_matrix_this_time
        cov_det_simulated[i_node] = np.linalg.det(covariance_simulated[:, :, i_node])
    difference_between_covs = cov_det_opt - cov_det_simulated

    # Actually save
    data_to_save = {
        "w_opt": w_opt,
        "computational_time": computational_time,
        "nb_iterations": nb_iterations,
        "nb_variables": nb_variables,
        "nb_constraints": nb_constraints,
        "nb_non_zero_elem_in_grad_f": nb_non_zero_elem_in_grad_f,
        "nb_non_zero_elem_in_grad_g": nb_non_zero_elem_in_grad_g,
        "variable_opt": variable_opt,
        "states_opt_array": states_opt_array,
        "controls_opt_array": controls_opt_array,
        "cov_opt_array": cov_opt_array,
        "variable_init": variable_init,
        "states_init_array": states_init_array,
        "controls_init_array": controls_init_array,
        "variable_lb": variable_lb,
        "variable_ub": variable_ub,
        "states_opt_mean": states_opt_mean,
        "x_simulated": x_simulated,
        "x_mean_simulated": x_mean_simulated,
        "covariance_simulated": covariance_simulated,
        "difference_between_means": difference_between_means,
        "difference_between_covs": difference_between_covs,
    }
    with open(save_path, "wb") as file:
        pickle.dump(data_to_save, file)

    return data_to_save
