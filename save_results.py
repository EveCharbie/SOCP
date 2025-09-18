import numpy as np
import pickle

from bioptim import SolutionMerge

from utils import get_git_version


def save_ocp(sol_ocp, save_path_ocp: str, tol: float):

    data = {}

    # Optimization variables
    data["q_sol"] = sol_ocp.decision_states(to_merge=SolutionMerge.NODES)["q"]
    data["qdot_sol"] = sol_ocp.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
    data["activations_sol"] = sol_ocp.decision_states(to_merge=SolutionMerge.NODES)["muscles"]
    data["excitations_sol"] = sol_ocp.decision_controls(to_merge=SolutionMerge.NODES)["muscles"]
    data["tau_sol"] = sol_ocp.decision_controls(to_merge=SolutionMerge.NODES)["tau"]
    data["time"] = sol_ocp.decision_time(to_merge=SolutionMerge.NODES)  # Not optimized

    # Min anx max bounds
    data["min_bounds_q"] = sol_ocp.ocp.nlp[0].x_bounds["q"].min
    data["max_bounds_q"] = sol_ocp.ocp.nlp[0].x_bounds["q"].max
    data["min_bounds_qdot"] = sol_ocp.ocp.nlp[0].x_bounds["qdot"].min
    data["max_bounds_qdot"] = sol_ocp.ocp.nlp[0].x_bounds["qdot"].max
    data["min_activations"] = sol_ocp.ocp.nlp[0].x_bounds["activations"].min
    data["max_activations"] = sol_ocp.ocp.nlp[0].x_bounds["activations"].max
    data["min_excitations"] = sol_ocp.ocp.nlp[0].u_bounds["excitations"].min
    data["max_excitations"] = sol_ocp.ocp.nlp[0].u_bounds["excitations"].max
    data["min_tau"] = sol_ocp.ocp.nlp[0].u_bounds["tau"].min
    data["max_tau"] = sol_ocp.ocp.nlp[0].u_bounds["tau"].max

    # Additional infos and objectives + constraints
    data["status"] = sol_ocp.status
    data["iterations"] = sol_ocp.iterations
    data["total_cost"] = sol_ocp.cost
    data["detailed_cost"] = sol_ocp.add_detailed_cost
    # # Otherwise redirect the print output
    # from contextlib import redirect_stdout
    # with open('out.txt', 'w') as f:
    #     with redirect_stdout(f):
    #         sol.print_cost()  # TODO: but in any ways, the output of print is ofter buggy
    data["git_version"] = get_git_version()
    data["real_time_to_optimize"] = sol_ocp.real_time_to_optimize
    data["constraints"] = sol_ocp.constraints
    data["lam_g"] = sol_ocp.lam_g
    data["lam_p"] = sol_ocp.lam_p
    data["lam_x"] = sol_ocp.lam_x
    data["phase_time"] = sol_ocp.ocp.phase_time
    data["n_shooting"] = sol_ocp.ocp.n_shooting
    data["dof_names"] = sol_ocp.ocp.nlp[0].dof_names

    # Integrated states
    integrated_sol = sol_ocp.integrate(to_merge=SolutionMerge.NODES)
    data["q_integrated"] = integrated_sol["q"]
    data["qdot_integrated"] = integrated_sol["qdot"]
    data["activations_integrated"] = integrated_sol["activations"]

    # Get the name of the file to save
    ocp_print_tol = "{:1.1e}".format(tol).replace(".", "p")
    if sol_ocp.status != 0:
        save_path_ocp = save_path_ocp.replace(".pkl", f"_DVG_{ocp_print_tol}.pkl")
    else:
        save_path_ocp = save_path_ocp.replace(".pkl", f"_CVG_{ocp_print_tol}.pkl")

    # --- Save --- #
    with open(save_path_ocp, "wb") as file:
        pickle.dump(data, file)

    with open(save_path_ocp.replace(".pkl", f"_sol.pkl"), "wb") as file:
        del sol_ocp.ocp
        pickle.dump(sol_ocp, file)

    print("Saved : ", save_path_ocp)