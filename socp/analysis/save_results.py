from typing import Any
import casadi as cas
import pickle
import numpy as np

from .reintegrate_solution import reintegrate


def save_results(w_opt: cas.DM, ocp: dict[str, Any], save_path: str, n_simulations: int, solver: Any) -> dict[str, Any]:

    # Solving info
    computational_time = solver.stats()["t_proc_total"]
    nb_iterations = solver.stats()["iter_count"]

    # Get optimization variables
    T_opt, states_opt, collocation_points_opt, controls_opt, x_opt, z_opt, u_opt = ocp[
        "discretization_method"
    ].get_variables_from_vector(
        ocp["model"],
        ocp["states_lower_bounds"],
        ocp["controls_lower_bounds"],
        w_opt,
    )

    T_init, states_init, collocation_points_init, controls_init, x_init, z_init, u_init = ocp[
        "discretization_method"
    ].get_variables_from_vector(
        ocp["model"],
        ocp["states_lower_bounds"],
        ocp["controls_lower_bounds"],
        ocp["w0"],
    )
    T_lb, states_lb, collocation_points_lb, controls_lb, x_lb, z_lb, u_lb = ocp[
        "discretization_method"
    ].get_variables_from_vector(
        ocp["model"],
        ocp["states_lower_bounds"],
        ocp["controls_lower_bounds"],
        ocp["lbw"],
    )
    T_ub, states_ub, collocation_points_ub, controls_ub, x_ub, z_ub, u_ub = ocp[
        "discretization_method"
    ].get_variables_from_vector(
        ocp["model"],
        ocp["states_lower_bounds"],
        ocp["controls_lower_bounds"],
        ocp["ubw"],
    )

    time_vector = np.linspace(0, ocp["final_time"], ocp["n_shooting"] + 1)
    states_opt_mean = ocp["discretization_method"].get_mean_states(
        model=ocp["model"],
        x=x_opt,
        squared=False,
    )

    # Reintegrate the solution
    x_simulated = reintegrate(
        time_vector=time_vector,
        states_opt_mean=states_opt_mean,
        controls_opt=controls_opt,
        ocp=ocp,
        n_simulations=n_simulations,
        save_path=save_path,
        plot_flag=True,
    )

    # Actually save
    data_to_save = {
        "w_opt": w_opt,
        "computational_time": computational_time,
        "nb_iterations": nb_iterations,
        "states_opt": states_opt,
        "controls_opt": controls_opt,
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
        "x_simulated": x_simulated,
    }
    with open(save_path, "wb") as file:
        pickle.dump(data_to_save, file)
