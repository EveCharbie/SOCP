from typing import Any
import casadi as cas
import pickle
import numpy as np

from .reintegrate_solution import reintegrate
from .estimate_covariance import estimate_covariance
from ..transcriptions.direct_collocation_polynomial import DirectCollocationPolynomial
from ..transcriptions.direct_collocation_trapezoidal import DirectCollocationTrapezoidal
from ..transcriptions.transcription_abstract import TranscriptionAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract
from ..examples.example_abstract import ExampleAbstract


def get_opt_arrays(
        ocp_example: ExampleAbstract,
        discretization_method: DiscretizationAbstract,
        states_opt: dict[str, cas.DM],
        controls_opt: dict[str, cas.DM],
        nb_collocation_points: int,
        n_shooting: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the optimized states and controls into array format to use the same functions as during the optimization with x and u.
    """
    # TODO: Add collocation points

    states_opt_array = np.zeros((ocp_example.model.nb_states, n_shooting + 1))
    state_names = [name for name in states_opt.keys() if name not in ["covariance", "m"]]
    for state_name in state_names:
        indices = ocp_example.model.state_indices[state_name]
        states_opt_array[indices, :] = states_opt[state_name]

    if "covariance" in states_opt.keys():
        cov_vector = None
        for i_node in range(n_shooting + 1):
            if discretization_method.with_cholesky:
                vect = ocp_example.model.reshape_matrix_to_cholesky_vector(
                    states_opt["covariance"][:, :, i_node]
                )
            else:
                vect = ocp_example.model.reshape_matrix_to_vector(
                    states_opt["covariance"][:, :, i_node]
                )

            if cov_vector is None:
                cov_vector = vect
            else:
                cov_vector = np.hstack((cov_vector, vect))

        states_opt_array = np.vstack((states_opt_array, cov_vector))

    if "m" in states_opt.keys():
        # The last interval does not have collocation points
        m_vector = None
        for i_node in range(n_shooting):
            tempo = None
            for i_collocation in range(nb_collocation_points - 1):
                vect = ocp_example.model.reshape_matrix_to_vector(states_opt["m"][:, :, i_collocation, i_node])
                tempo = vect if tempo is None else np.vstack((tempo, vect))

            if m_vector is None:
                m_vector = tempo
            else:
                m_vector = np.hstack((m_vector, tempo))

        m_vector = np.hstack((m_vector, np.zeros_like(tempo)))  # Add the final zeros
        states_opt_array = np.vstack((states_opt_array, m_vector))

    controls_opt_array = np.zeros((ocp_example.model.nb_controls, n_shooting))
    control_names = [name for name in controls_opt.keys()]
    for control_name in control_names:
        indices = ocp_example.model.control_indices[control_name]
        controls_opt_array[indices, :] = controls_opt[control_name]

    return states_opt_array, controls_opt_array


def save_results(
        w_opt: cas.DM,
        ocp: dict[str, Any],
        save_path: str,
        n_simulations: int,
        solver: Any,
        grad_f_func: cas.Function,
        grad_g_func: cas.Function,
) -> dict[str, Any]:

    if isinstance(ocp["dynamics_transcription"], DirectCollocationPolynomial):
        nb_collocation_points = ocp["dynamics_transcription"].order + 2
    elif isinstance(ocp["dynamics_transcription"], DirectCollocationTrapezoidal):
        nb_collocation_points = 1
    else:
        nb_collocation_points = 0

    # Solving info
    computational_time = solver.stats()["t_proc_total"]
    nb_iterations = solver.stats()["iter_count"]
    nb_variables = ocp["w"].shape[0]
    nb_constraints = ocp["g"].shape[0]
    nb_non_zero_elem_in_grad_f = grad_f_func(ocp["w"]).nnz()
    nb_non_zero_elem_in_grad_g = grad_g_func(ocp["w"]).nnz()

    # Get optimization variables
    T_opt, states_opt, collocation_points_opt, controls_opt, x_opt, z_opt, u_opt = ocp[
        "discretization_method"
    ].get_variables_from_vector(
        ocp["ocp_example"].model,
        ocp["states_lower_bounds"],
        ocp["controls_lower_bounds"],
        w_opt,
    )

    T_init, states_init, collocation_points_init, controls_init, x_init, z_init, u_init = ocp[
        "discretization_method"
    ].get_variables_from_vector(
        ocp["ocp_example"].model,
        ocp["states_lower_bounds"],
        ocp["controls_lower_bounds"],
        ocp["w0"],
    )
    T_lb, states_lb, collocation_points_lb, controls_lb, x_lb, z_lb, u_lb = ocp[
        "discretization_method"
    ].get_variables_from_vector(
        ocp["ocp_example"].model,
        ocp["states_lower_bounds"],
        ocp["controls_lower_bounds"],
        ocp["lbw"],
    )
    T_ub, states_ub, collocation_points_ub, controls_ub, x_ub, z_ub, u_ub = ocp[
        "discretization_method"
    ].get_variables_from_vector(
        ocp["ocp_example"].model,
        ocp["states_lower_bounds"],
        ocp["controls_lower_bounds"],
        ocp["ubw"],
    )

    time_vector = np.linspace(0, T_opt, ocp["n_shooting"] + 1)

    states_opt_array, controls_opt_array = get_opt_arrays(
        ocp["ocp_example"],
        ocp["discretization_method"],
        states_opt,
        controls_opt,
        nb_collocation_points,
        ocp["n_shooting"],
    )

    # Mean states
    states_opt_mean = np.array(ocp["discretization_method"].get_mean_states(
        model=ocp["ocp_example"].model,
        x=states_opt_array[:ocp["ocp_example"].model.nb_states],
        squared=False,
    ))

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
    cov_det_opt = np.zeros((ocp["n_shooting"] + 1, ))
    cov_det_simulated = np.zeros((ocp["n_shooting"] + 1, ))
    for i_node in range(ocp["n_shooting"] + 1):
        cov_det_opt[i_node] = np.linalg.det(ocp["discretization_method"].get_covariance(
            ocp["ocp_example"].model,
            states_opt_array[:, i_node],
        ))
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
        "states_opt": states_opt,
        "controls_opt": controls_opt,
        "states_opt_array": states_opt_array,
        "controls_opt_array": controls_opt_array,
        "x_opt": x_opt,
        "u_opt": u_opt,
        "states_init": states_init,
        "controls_init": controls_init,
        "x_init": x_init,
        "u_init": u_init,
        "states_lb": states_lb,
        "controls_lb": controls_lb,
        "x_lb": x_lb,
        "u_lb": u_lb,
        "states_ub": states_ub,
        "controls_ub": controls_ub,
        "x_ub": x_ub,
        "u_ub": u_ub,
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
