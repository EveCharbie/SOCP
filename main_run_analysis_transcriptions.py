import numpy as np
import casadi as cas
import pickle
import os
import matplotlib.pyplot as plt

from socp import prepare_ocp, VertebrateArm, DirectMultipleShooting, NoiseDiscretization
from socp.analysis.reintegrate_solution import reintegrate


def get_nb_random_from_filename(filename):
    # Extract the number after "_NoiseDiscretization_" and before the next underscore
    start_str = "_NoiseDiscretization_"
    start_idx = filename.find(start_str)
    if start_idx == -1:
        return None  # Not found
    start_idx += len(start_str)
    end_idx = filename.find("_", start_idx)
    if end_idx == -1:
        return None  # Not found
    nb_random_str = filename[start_idx:end_idx]
    try:
        return int(nb_random_str)
    except ValueError:
        return None  # Not an integer


def optimal_cost_func(variable_opt, x_end_simulated):

    n_shooting = variable_opt.n_shooting
    dt = variable_opt.get_time() / n_shooting

    f = 0
    for i_node in range(n_shooting + 1):

        tau = variable_opt.get_control("tau", node=i_node)
        f += cas.sum1(tau**2) * dt

        if "k" in variable_opt.control_names:
            k = variable_opt.get_control("k", node=i_node)
            f += 0.01 * cas.sum1(k**2) * dt
            if i_node > 0:
                k_previous = variable_opt.get_control("k", i_node - 1)
                f += 0.01 * cas.sum1((k - k_previous) ** 2) * dt

    f += cas.sum1(100 * np.var(x_end_simulated, axis=1))

    return f

# --- Load the results --- #
randoms_considered = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

data_DirectCollocationPolynomial_Deterministic = None
data_DirectCollocationPolynomial_Noise = {f"nb_random_{nb}": None for nb in randoms_considered}
data_DirectCollocationPolynomial_MeanAndCovariance = None
data_Trapezoidal_Deterministic = None
data_Trapezoidal_Noise = {f"nb_random_{nb}": None for nb in randoms_considered}
data_Trapezoidal_MeanAndCovariance = None
data_VariationalPolynomial_Deterministic = None
data_VariationalPolynomial_Noise = {f"nb_random_{nb}": None for nb in randoms_considered}
data_VariationalPolynomial_MeanAndCovariance = None
data_Variational_Deterministic = None
data_Variational_Noise = {f"nb_random_{nb}": None for nb in randoms_considered}
data_DirectMultipleShooting_Deterministic = None
data_DirectMultipleShooting_Noise = {f"nb_random_{nb}": None for nb in randoms_considered}
data_DirectMultipleShooting_MeanAndCovariance = None

empty_data = {
            "nb var": None,
            "nb const": None,
            "time": None,
            "nb iter": None,
            "cost": None,
        }
PDC_Deterministic = None
PDC_NS = {f"nb_random_{nb}": empty_data for nb in randoms_considered}
PDC_MAC = empty_data
TDC_Deterministic = None
TDC_NS = {f"nb_random_{nb}": empty_data for nb in randoms_considered}
TDC_MAC = empty_data
DMS_Deterministic = None
DMS_NS = {f"nb_random_{nb}": empty_data for nb in randoms_considered}
DMS_MAC = empty_data
TDMaOC_Deterministic = None
TDMaOC_NS = {f"nb_random_{nb}": empty_data for nb in randoms_considered}
TDMaOC_MAC = empty_data
PDMaOC_Deterministic = None
PDMaOC_NS = {f"nb_random_{nb}": empty_data for nb in randoms_considered}
PDMaOC_MAC = empty_data



results_path = "results/to_analyze/"
for file in os.listdir(results_path):
    if file.endswith(".pkl"):

        if "DirectCollocationPolynomial" in file:
            if "NoiseDiscretization" in file:
                with open(results_path + file, "rb",) as f:
                    nb_random = get_nb_random_from_filename(file)
                    data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"] = pickle.load(f)
                    if "DVG" in file:
                        data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["computational_time"] = None
                        data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["nb_iterations"] = None
                        data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["optimal_cost"] = None
                        optimal_cost_simulated = None
                        time_per_iter = None
                    else:
                        time_per_iter = data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["computational_time"] / data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["variable_opt"],
                            x_end_simulated=data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["x_simulated"][:, -1, :],
                        )
                    PDC_NS[f"nb_random_{nb_random}"] = {
                        "nb var": data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["nb_variables"],
                        "nb const": data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["nb_constraints"],
                        "time": data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["computational_time"],
                        "nb iter": data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["g_without_bounds_at_init"],
                    }

            elif "MeanAndCovariance" in file:
                with open(results_path + file, "rb",) as f:
                    data_DirectCollocationPolynomial_MeanAndCovariance = pickle.load(f)
                    if "DVG" in file:
                        data_DirectCollocationPolynomial_MeanAndCovariance["computational_time"] = None
                        data_DirectCollocationPolynomial_MeanAndCovariance["nb_iterations"] = None
                        data_DirectCollocationPolynomial_MeanAndCovariance["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_DirectCollocationPolynomial_MeanAndCovariance["computational_time"] / data_DirectCollocationPolynomial_MeanAndCovariance["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_DirectCollocationPolynomial_MeanAndCovariance["variable_opt"],
                            x_end_simulated=data_DirectCollocationPolynomial_MeanAndCovariance["x_simulated"][:, -1, :],
                        )
                    PDC_MAC = {
                        "nb var": data_DirectCollocationPolynomial_MeanAndCovariance["nb_variables"],
                        "nb const": data_DirectCollocationPolynomial_MeanAndCovariance["nb_constraints"],
                        "time": data_DirectCollocationPolynomial_MeanAndCovariance["computational_time"],
                        "nb iter": data_DirectCollocationPolynomial_MeanAndCovariance["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_DirectCollocationPolynomial_MeanAndCovariance[
                            "g_without_bounds_at_init"],
                    }

            elif "Deterministic" in file:
                with open(results_path + file, "rb",) as f:
                    data_DirectCollocationPolynomial_Deterministic = pickle.load(f)
                    if "DVG" in file:
                        data_DirectCollocationPolynomial_Deterministic["computational_time"] = None
                        data_DirectCollocationPolynomial_Deterministic["nb_iterations"] = None
                        data_DirectCollocationPolynomial_Deterministic["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_DirectCollocationPolynomial_Deterministic["computational_time"] / data_DirectCollocationPolynomial_Deterministic["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_DirectCollocationPolynomial_Deterministic["variable_opt"],
                            x_end_simulated=data_DirectCollocationPolynomial_Deterministic["x_simulated"][:, -1, :],
                        )
                    PDC_D = {
                        "nb var": data_DirectCollocationPolynomial_Deterministic["nb_variables"],
                        "nb const": data_DirectCollocationPolynomial_Deterministic["nb_constraints"],
                        "time": data_DirectCollocationPolynomial_Deterministic["computational_time"],
                        "nb iter": data_DirectCollocationPolynomial_Deterministic["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_DirectCollocationPolynomial_Deterministic["g_without_bounds_at_init"],
                    }

        elif "DirectMultipleShooting" in file:
            if "NoiseDiscretization" in file:
                with open(results_path + file, "rb",) as f:
                    nb_random = get_nb_random_from_filename(file)
                    data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"] = pickle.load(f)
                    if "DVG" in file:
                        data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["computational_time"] = None
                        data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["nb_iterations"] = None
                        data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["computational_time"] / data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["variable_opt"],
                            x_end_simulated=data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["x_simulated"][:, -1, :],
                        )
                    DMS_NS[f"nb_random_{nb_random}"] = {
                        "nb var": data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["nb_variables"],
                        "nb const": data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["nb_constraints"],
                        "time": data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["computational_time"],
                        "nb iter": data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["g_without_bounds_at_init"],
                    }

            elif "MeanAndCovariance" in file:
                with open(results_path + file, "rb",) as f:
                    data_DirectMultipleShooting_MeanAndCovariance = pickle.load(f)
                    if "DVG" in file:
                        data_DirectMultipleShooting_MeanAndCovariance["computational_time"] = None
                        data_DirectMultipleShooting_MeanAndCovariance["nb_iterations"] = None
                        data_DirectMultipleShooting_MeanAndCovariance["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_DirectMultipleShooting_MeanAndCovariance["computational_time"] / data_DirectMultipleShooting_MeanAndCovariance["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_DirectMultipleShooting_MeanAndCovariance["variable_opt"],
                            x_end_simulated=data_DirectMultipleShooting_MeanAndCovariance["x_simulated"][:, -1, :],
                        )
                    DMS_MAC = {
                        "nb var": data_DirectMultipleShooting_MeanAndCovariance["nb_variables"],
                        "nb const": data_DirectMultipleShooting_MeanAndCovariance["nb_constraints"],
                        "time": data_DirectMultipleShooting_MeanAndCovariance["computational_time"],
                        "nb iter": data_DirectMultipleShooting_MeanAndCovariance["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_DirectMultipleShooting_MeanAndCovariance["g_without_bounds_at_init"],
                    }

            elif "Deterministic" in file:
                with open(results_path + file, "rb",) as f:
                    data_DirectMultipleShooting_Deterministic = pickle.load(f)
                    if "DVG" in file:
                        data_DirectMultipleShooting_Deterministic["computational_time"] = None
                        data_DirectMultipleShooting_Deterministic["nb_iterations"] = None
                        data_DirectMultipleShooting_Deterministic["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_DirectMultipleShooting_Deterministic["computational_time"] / data_DirectMultipleShooting_Deterministic["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_DirectMultipleShooting_Deterministic["variable_opt"],
                            x_end_simulated=data_DirectMultipleShooting_Deterministic["x_simulated"][:, -1, :],
                        )
                    DMS_D = {
                        "nb var": data_DirectMultipleShooting_Deterministic["nb_variables"],
                        "nb const": data_DirectMultipleShooting_Deterministic["nb_constraints"],
                        "time": data_DirectMultipleShooting_Deterministic["computational_time"],
                        "nb iter": data_DirectMultipleShooting_Deterministic["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_DirectMultipleShooting_Deterministic["g_without_bounds_at_init"],
                    }

        elif "DirectCollocationTrapezoidal" in file:
            if "NoiseDiscretization" in file:
                with open(results_path + file, "rb",) as f:
                    nb_random = get_nb_random_from_filename(file)
                    data_Trapezoidal_Noise[f"nb_random_{nb_random}"] = pickle.load(f)
                    if "DVG" in file:
                        data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["computational_time"] = None
                        data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["nb_iterations"] = None
                        data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["computational_time"] / data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["variable_opt"],
                            x_end_simulated=data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["x_simulated"][:, -1, :],
                        )
                    TDC_NS[f"nb_random_{nb_random}"] = {
                        "nb var": data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["nb_variables"],
                        "nb const": data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["nb_constraints"],
                        "time": data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["computational_time"],
                        "nb iter": data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["g_without_bounds_at_init"],
                    }

            elif "MeanAndCovariance" in file:
                with open(results_path + file, "rb",) as f:
                    data_Trapezoidal_MeanAndCovariance = pickle.load(f)
                    if "DVG" in file:
                        data_Trapezoidal_MeanAndCovariance["computational_time"] = None
                        data_Trapezoidal_MeanAndCovariance["nb_iterations"] = None
                        data_Trapezoidal_MeanAndCovariance["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_Trapezoidal_MeanAndCovariance["computational_time"] / data_Trapezoidal_MeanAndCovariance["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_Trapezoidal_MeanAndCovariance["variable_opt"],
                            x_end_simulated=data_Trapezoidal_MeanAndCovariance["x_simulated"][:, -1, :],
                        )
                    TDC_MAC = {
                        "nb var": data_Trapezoidal_MeanAndCovariance["nb_variables"],
                        "nb const": data_Trapezoidal_MeanAndCovariance["nb_constraints"],
                        "time": data_Trapezoidal_MeanAndCovariance["computational_time"],
                        "nb iter": data_Trapezoidal_MeanAndCovariance["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_Trapezoidal_MeanAndCovariance["g_without_bounds_at_init"],
                    }

            elif "Deterministic" in file:
                with open(results_path + file, "rb",) as f:
                    data_Trapezoidal_Deterministic = pickle.load(f)
                    if "DVG" in file:
                        data_Trapezoidal_Deterministic["computational_time"] = None
                        data_Trapezoidal_Deterministic["nb_iterations"] = None
                        data_Trapezoidal_Deterministic["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_Trapezoidal_Deterministic["computational_time"] / data_Trapezoidal_Deterministic["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_Trapezoidal_Deterministic["variable_opt"],
                            x_end_simulated=data_Trapezoidal_Deterministic["x_simulated"][:, -1, :],
                        )
                    TDC_D = {
                        "nb var": data_Trapezoidal_Deterministic["nb_variables"],
                        "nb const": data_Trapezoidal_Deterministic["nb_constraints"],
                        "time": data_Trapezoidal_Deterministic["computational_time"],
                        "nb iter": data_Trapezoidal_Deterministic["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_Trapezoidal_Deterministic["g_without_bounds_at_init"],
                    }

        elif "VariationalPolynomial" in file:
            if "NoiseDiscretization" in file:
                with open(results_path + file, "rb",) as f:
                    nb_random = get_nb_random_from_filename(file)
                    data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"] = pickle.load(f)
                    n_shooting = data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["norm_difference_between_means"].shape[
                                     0] - 1
                    if "DVG" in file:
                        data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["computational_time"] = None
                        data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["nb_iterations"] = None
                        data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["computational_time"] / data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["variable_opt"],
                            x_end_simulated=data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["x_simulated"][:, -1, :],
                        )
                    PDMaOC_NS[f"nb_random_{nb_random}"] = {
                        "nb var": data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["nb_variables"],
                        "nb const": data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["nb_constraints"],
                        "time": data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["computational_time"],
                        "nb iter": data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["g_without_bounds_at_init"],
                    }

            elif "MeanAndCovariance" in file:
                with open(results_path + file, "rb",) as f:
                    data_VariationalPolynomial_MeanAndCovariance = pickle.load(f)
                    if "DVG" in file:
                        data_VariationalPolynomial_MeanAndCovariance["computational_time"] = None
                        data_VariationalPolynomial_MeanAndCovariance["nb_iterations"] = None
                        data_VariationalPolynomial_MeanAndCovariance["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_VariationalPolynomial_MeanAndCovariance["computational_time"] / data_VariationalPolynomial_MeanAndCovariance["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_VariationalPolynomial_MeanAndCovariance["variable_opt"],
                            x_end_simulated=data_VariationalPolynomial_MeanAndCovariance["x_simulated"][:, -1, :],
                        )
                    PDMaOC_MAC = {
                        "nb var": data_VariationalPolynomial_MeanAndCovariance["nb_variables"],
                        "nb const": data_VariationalPolynomial_MeanAndCovariance["nb_constraints"],
                        "time": data_VariationalPolynomial_MeanAndCovariance["computational_time"],
                        "nb iter": data_VariationalPolynomial_MeanAndCovariance["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_VariationalPolynomial_MeanAndCovariance["g_without_bounds_at_init"],
                    }

            elif "Deterministic" in file:
                with open(results_path + file, "rb",) as f:
                    data_VariationalPolynomial_Deterministic = pickle.load(f)
                    if "DVG" in file:
                        data_VariationalPolynomial_Deterministic["computational_time"] = None
                        data_VariationalPolynomial_Deterministic["nb_iterations"] = None
                        data_VariationalPolynomial_Deterministic["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_VariationalPolynomial_Deterministic["computational_time"] / data_VariationalPolynomial_Deterministic["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_VariationalPolynomial_Deterministic["variable_opt"],
                            x_end_simulated=data_VariationalPolynomial_Deterministic["x_simulated"][:, -1, :],
                        )
                    PDMaOC_D = {
                        "nb var": data_VariationalPolynomial_Deterministic["nb_variables"],
                        "nb const": data_VariationalPolynomial_Deterministic["nb_constraints"],
                        "time": data_VariationalPolynomial_Deterministic["computational_time"],
                        "nb iter": data_VariationalPolynomial_Deterministic["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_VariationalPolynomial_Deterministic["g_without_bounds_at_init"],
                    }

        elif "Variational" in file:
            if "NoiseDiscretization" in file:
                with open(results_path + file, "rb",) as f:
                    nb_random = get_nb_random_from_filename(file)
                    data_Variational_Noise[f"nb_random_{nb_random}"] = pickle.load(f)
                    if "DVG" in file:
                        data_Variational_Noise[f"nb_random_{nb_random}"]["computational_time"] = None
                        data_Variational_Noise[f"nb_random_{nb_random}"]["nb_iterations"] = None
                        data_Variational_Noise[f"nb_random_{nb_random}"]["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_Variational_Noise[f"nb_random_{nb_random}"]["computational_time"] / data_Variational_Noise[f"nb_random_{nb_random}"]["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_Variational_Noise[f"nb_random_{nb_random}"]["variable_opt"],
                            x_end_simulated=data_Variational_Noise[f"nb_random_{nb_random}"]["x_simulated"][:, -1, :],
                        )
                    TDMaOC_NS[f"nb_random_{nb_random}"] = {
                        "nb var": data_Variational_Noise[f"nb_random_{nb_random}"]["nb_variables"],
                        "nb const": data_Variational_Noise[f"nb_random_{nb_random}"]["nb_constraints"],
                        "time": data_Variational_Noise[f"nb_random_{nb_random}"]["computational_time"],
                        "nb iter": data_Variational_Noise[f"nb_random_{nb_random}"]["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_Variational_Noise[f"nb_random_{nb_random}"]["g_without_bounds_at_init"],
                    }

            elif "Deterministic" in file:
                with open(results_path + file, "rb",) as f:
                    data_Variational_Deterministic = pickle.load(f)
                    if "DVG" in file:
                        data_Variational_Deterministic["computational_time"] = None
                        data_Variational_Deterministic["nb_iterations"] = None
                        data_Variational_Deterministic["optimal_cost"] = None
                        time_per_iter = None
                        optimal_cost_simulated = None
                    else:
                        time_per_iter = data_Variational_Deterministic["computational_time"] / data_Variational_Deterministic["nb_iterations"]
                        optimal_cost_simulated = optimal_cost_func(
                            variable_opt=data_Variational_Deterministic["variable_opt"],
                            x_end_simulated=data_Variational_Deterministic["x_simulated"][:, -1, :],
                        )
                    TDMaOC_D = {
                        "nb var": data_Variational_Deterministic["nb_variables"],
                        "nb const": data_Variational_Deterministic["nb_constraints"],
                        "time": data_Variational_Deterministic["computational_time"],
                        "nb iter": data_Variational_Deterministic["nb_iterations"],
                        "time per iter": time_per_iter,
                        "cost": optimal_cost_simulated,
                        "g_without_bounds_at_init": data_Variational_Deterministic["g_without_bounds_at_init"],
                    }

# --- Plot the sensitivity analysis --- #

DirectCollocationPolynomial_nb_randoms = []
DirectCollocationPolynomial_optimal_costs = []
DirectCollocationPolynomial_state_errors = []
DirectCollocationPolynomial_cov_errors = []
DirectCollocationPolynomial_computational_time = []

DirectMultipleShooting_nb_randoms = []
DirectMultipleShooting_optimal_costs = []
DirectMultipleShooting_state_errors = []
DirectMultipleShooting_cov_errors = []
DirectMultipleShooting_computational_time = []

Trapezoidal_nb_randoms = []
Trapezoidal_optimal_costs = []
Trapezoidal_state_errors = []
Trapezoidal_cov_errors = []
Trapezoidal_computational_time = []

Variational_nb_randoms = []
Variational_optimal_costs = []
Variational_state_errors = []
Variational_cov_errors = []
Variational_computational_time = []

VariationalPolynomial_nb_randoms = []
VariationalPolynomial_optimal_costs = []
VariationalPolynomial_state_errors = []
VariationalPolynomial_cov_errors = []
VariationalPolynomial_computational_time = []

for nb_random in randoms_considered:
    if data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["computational_time"] is not None:
        DirectCollocationPolynomial_nb_randoms += [nb_random]
        DirectCollocationPolynomial_optimal_costs += [data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["optimal_cost"]]
        DirectCollocationPolynomial_state_errors += [data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["norm_difference_between_means"][-1]]
        DirectCollocationPolynomial_computational_time += [data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["computational_time"]]
        DirectCollocationPolynomial_cov_errors += [data_DirectCollocationPolynomial_Noise[f"nb_random_{nb_random}"]["norm_difference_between_covs"][-1]]
    if data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["computational_time"] is not None:
        DirectMultipleShooting_nb_randoms += [nb_random]
        DirectMultipleShooting_optimal_costs += [data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["optimal_cost"]]
        DirectMultipleShooting_state_errors += [data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["norm_difference_between_means"][-1]]
        DirectMultipleShooting_computational_time += [data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["computational_time"]]
        DirectMultipleShooting_cov_errors += [data_DirectMultipleShooting_Noise[f"nb_random_{nb_random}"]["norm_difference_between_covs"][-1]]
    if data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["computational_time"] is not None:
        Trapezoidal_nb_randoms += [nb_random]
        Trapezoidal_optimal_costs += [data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["optimal_cost"]]
        Trapezoidal_state_errors += [data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["norm_difference_between_means"][-1]]
        Trapezoidal_computational_time += [data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["computational_time"]]
        Trapezoidal_cov_errors += [data_Trapezoidal_Noise[f"nb_random_{nb_random}"]["norm_difference_between_covs"][-1]]
    if data_Variational_Noise[f"nb_random_{nb_random}"]["computational_time"] is not None:
        Variational_nb_randoms += [nb_random]
        Variational_optimal_costs += [data_Variational_Noise[f"nb_random_{nb_random}"]["optimal_cost"]]
        Variational_state_errors += [data_Variational_Noise[f"nb_random_{nb_random}"]["norm_difference_between_means"][-1]]
        Variational_computational_time += [data_Variational_Noise[f"nb_random_{nb_random}"]["computational_time"]]
        Variational_cov_errors += [data_Variational_Noise[f"nb_random_{nb_random}"]["norm_difference_between_covs"][-1]]
    if data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["computational_time"] is not None:
        VariationalPolynomial_nb_randoms += [nb_random]
        VariationalPolynomial_optimal_costs += [data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["optimal_cost"]]
        VariationalPolynomial_state_errors += [data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["norm_difference_between_means"][-1]]
        VariationalPolynomial_computational_time += [data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["computational_time"]]
        VariationalPolynomial_cov_errors += [data_VariationalPolynomial_Noise[f"nb_random_{nb_random}"]["norm_difference_between_covs"][-1]]

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(np.array(DirectCollocationPolynomial_nb_randoms), np.array(DirectCollocationPolynomial_state_errors), "--o", color="tab:red", label="Polynomial Direct Collocation")
axs[0, 0].plot(np.array(DirectMultipleShooting_nb_randoms), np.array(DirectMultipleShooting_state_errors), "--o", color="tab:orange", label="Direct Multiple Shooting")
axs[0, 0].plot(np.array(Trapezoidal_nb_randoms), np.array(Trapezoidal_state_errors), "--o", color="tab:green", label="Trapezoidal Direct Collocation")
axs[0, 0].plot(np.array(Variational_nb_randoms), np.array(Variational_state_errors), "--o", color="tab:blue", label="Trapezoidal DMaOC")
axs[0, 0].plot(np.array(VariationalPolynomial_nb_randoms), np.array(VariationalPolynomial_state_errors), "--o", color="tab:purple", label="Polynomial, DMaOC")

axs[0, 0].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_DirectCollocationPolynomial_MeanAndCovariance["norm_difference_between_means"][-1], ":", color="tab:red")
axs[0, 0].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_DirectMultipleShooting_MeanAndCovariance["norm_difference_between_means"][-1], ":", color="tab:orange")
axs[0, 0].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_Trapezoidal_MeanAndCovariance["norm_difference_between_means"][-1], ":", color="tab:green")
axs[0, 0].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_VariationalPolynomial_MeanAndCovariance["norm_difference_between_means"][-1], ":", color="tab:purple")

axs[0, 0].set_title(r"$||\bar{q}_{opt}(T) - \bar{q}_{sim}(T)||_{2}$")

axs[0, 1].plot(np.array(DirectCollocationPolynomial_nb_randoms), np.array(DirectCollocationPolynomial_cov_errors), "--o", color="tab:red", label="Direct Collocation Polynomial")
axs[0, 1].plot(np.array(DirectMultipleShooting_nb_randoms), np.array(DirectMultipleShooting_cov_errors), "--o", color="tab:orange", label="Direct Multiple Shooting")
axs[0, 1].plot(np.array(Trapezoidal_nb_randoms), np.array(Trapezoidal_cov_errors), "--o", color="tab:green", label="Trapezoidal Direct Collocation")
axs[0, 1].plot(np.array(Variational_nb_randoms), np.array(Variational_cov_errors), "--o", color="tab:blue", label="Trapezoidal DMaOC")
axs[0, 1].plot(np.array(VariationalPolynomial_nb_randoms), np.array(VariationalPolynomial_cov_errors), "--o", color="tab:purple", label="Polynomial DMaOC")

axs[0, 1].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_DirectCollocationPolynomial_MeanAndCovariance["norm_difference_between_covs"][-1], ":", color="tab:red")
axs[0, 1].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_DirectMultipleShooting_MeanAndCovariance["norm_difference_between_covs"][-1], ":", color="tab:orange")
axs[0, 1].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_Trapezoidal_MeanAndCovariance["norm_difference_between_covs"][-1], ":", color="tab:green")
axs[0, 1].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_VariationalPolynomial_MeanAndCovariance["norm_difference_between_covs"][-1], ":", color="tab:purple")

axs[0, 1].set_title(r"$||P_{opt}(T) - P_{sim}(T)||_{Frobenius}$")

axs[1, 0].plot(np.array(DirectCollocationPolynomial_nb_randoms), np.array(DirectCollocationPolynomial_optimal_costs), "--o", color="tab:red", label="Polynomial Direct Collocation")
axs[1, 0].plot(np.array(DirectMultipleShooting_nb_randoms), np.array(DirectMultipleShooting_optimal_costs), "--o", color="tab:orange", label="Direct Multiple Shooting")
axs[1, 0].plot(np.array(Trapezoidal_nb_randoms), np.array(Trapezoidal_optimal_costs), "--o", color="tab:green", label="Trapezoidal Direct Collocation")
axs[1, 0].plot(np.array(Variational_nb_randoms), np.array(Variational_optimal_costs), "--o", color="tab:blue", label="Trapezoidal DMaOC")
axs[1, 0].plot(np.array(VariationalPolynomial_nb_randoms), np.array(VariationalPolynomial_optimal_costs), "--o", color="tab:purple", label="Polynomial DMaOC")

axs[1, 0].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_DirectCollocationPolynomial_MeanAndCovariance["optimal_cost"], ":", color="tab:red")
axs[1, 0].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_DirectMultipleShooting_MeanAndCovariance["optimal_cost"], ":", color="tab:orange")
axs[1, 0].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_Trapezoidal_MeanAndCovariance["optimal_cost"], ":", color="tab:green")
axs[1, 0].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_VariationalPolynomial_MeanAndCovariance["optimal_cost"], ":", color="tab:purple")

axs[1, 0].set_title("Optimal cost")

axs[1, 1].plot(np.array(DirectCollocationPolynomial_nb_randoms), np.array(DirectCollocationPolynomial_computational_time), "--o", color="tab:red", label="Polynomial Direct Collocation")
axs[1, 1].plot(np.array(DirectMultipleShooting_nb_randoms), np.array(DirectMultipleShooting_computational_time), "--o", color="tab:orange", label="Direct Multiple Shooting")
axs[1, 1].plot(np.array(Trapezoidal_nb_randoms), np.array(Trapezoidal_computational_time), "--o", color="tab:green", label="Trapezoidal Direct Collocation")
axs[1, 1].plot(np.array(Variational_nb_randoms), np.array(Variational_computational_time), "--o", color="tab:blue", label="Trapezoidal DMaOC")
axs[1, 1].plot(np.array(VariationalPolynomial_nb_randoms), np.array(VariationalPolynomial_computational_time), "--o", color="tab:purple", label="Polynomial DMaOC")

axs[1, 1].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_DirectCollocationPolynomial_MeanAndCovariance["computational_time"], ":", color="tab:red")
axs[1, 1].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_DirectMultipleShooting_MeanAndCovariance["computational_time"], ":", color="tab:orange")
axs[1, 1].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_Trapezoidal_MeanAndCovariance["computational_time"], ":", color="tab:green")
axs[1, 1].plot(np.array(randoms_considered), np.ones((len(randoms_considered), )) * data_VariationalPolynomial_MeanAndCovariance["computational_time"], ":", color="tab:purple")

axs[1, 1].set_title("Computational time [s]")

axs[0, 0].set_xticks(randoms_considered)
axs[0, 1].set_xticks(randoms_considered)
axs[1, 0].set_xticks(randoms_considered)
axs[1, 1].set_xticks(randoms_considered)

axs[1, 0].set_xlabel("Number of noise samples")
axs[1, 1].set_xlabel("Number of noise samples")

axs[0, 0].set_yscale("log")
axs[0, 1].set_yscale("log")
axs[1, 0].set_yscale("log")
axs[1, 1].set_yscale("log")

axs[1, 0].legend(bbox_to_anchor=(0.75, -0.2), loc="upper left", ncol=1)

plt.subplots_adjust(bottom=0.3, left=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.25)
plt.savefig("results/vertebrate_arm_sensitivity_analysis.png", dpi=300)
plt.show()


# --- Plot the comparison metrics --- #
nb_random_chosen = f"nb_random_30"
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

axs[0].plot(np.linspace(0, n_shooting, n_shooting + 1), np.zeros((n_shooting + 1,)), "-k")
axs[1].plot(np.linspace(0, n_shooting, n_shooting + 1), np.zeros((n_shooting + 1)), "-k")
if data_DirectCollocationPolynomial_Noise[nb_random_chosen]["computational_time"] is not None:
    axs[0].plot(data_DirectCollocationPolynomial_Noise[nb_random_chosen]["norm_difference_between_means"], "--", color="tab:red",
                label="Polynomial Direct Collocation x Noise Sampling")
    axs[1].plot(data_DirectCollocationPolynomial_Noise[nb_random_chosen]["norm_difference_between_covs"], "--", color="tab:red",
                label="Polynomial Direct Collocation x Noise Sampling")

if data_Trapezoidal_Noise[nb_random_chosen]["computational_time"] is not None:
    axs[0].plot(data_Trapezoidal_Noise[nb_random_chosen]["norm_difference_between_means"], "--", color="tab:green",
                label="Trapezoidal Direct Collocation x Noise Sampling")
    axs[1].plot(data_Trapezoidal_Noise[nb_random_chosen]["norm_difference_between_covs"], "--", color="tab:green",
                label="Trapezoidal Direct Collocation x Noise Sampling")

if data_DirectMultipleShooting_Noise[nb_random_chosen]["computational_time"] is not None:
    axs[0].plot(data_DirectMultipleShooting_Noise[nb_random_chosen]["norm_difference_between_means"], "--", color="tab:orange",
                label="Direct Multiple Shooting x Noise Sampling")
    axs[1].plot(data_DirectMultipleShooting_Noise[nb_random_chosen]["norm_difference_between_covs"], "--", color="tab:orange",
                label="Direct Multiple Shooting x Noise Sampling")

if data_Variational_Noise[nb_random_chosen]["computational_time"] is not None:
    axs[0].plot(data_Variational_Noise[nb_random_chosen]["norm_difference_between_means"], "--", color="tab:blue",
                label="Trapezoidal DMaOC x Noise Sampling")
    axs[1].plot(data_Variational_Noise[nb_random_chosen]["norm_difference_between_covs"], "--", color="tab:blue",
                label="Trapezoidal DMaOC x Noise Sampling")

if data_VariationalPolynomial_Noise[nb_random_chosen]["computational_time"] is not None:
    axs[0].plot(data_VariationalPolynomial_Noise[nb_random_chosen]["norm_difference_between_means"], "--", color="tab:purple",
                label="Polynomial DMaOC x Noise Sampling")
    axs[1].plot(data_VariationalPolynomial_Noise[nb_random_chosen]["norm_difference_between_covs"], "--", color="tab:purple",
                label="Polynomial DMaOC x Noise Sampling")

if data_DirectCollocationPolynomial_MeanAndCovariance is not None:
    axs[0].plot(data_DirectCollocationPolynomial_MeanAndCovariance["norm_difference_between_means"], ":",
                color="tab:red", label="Polynomial Direct Collocation x Noise distribution approx.")
    axs[1].plot(data_DirectCollocationPolynomial_MeanAndCovariance["norm_difference_between_covs"], ":",
                color="tab:red", label="Polynomial Direct Collocation x Noise distribution approx.")

if data_Trapezoidal_MeanAndCovariance is not None:
    axs[0].plot(data_Trapezoidal_MeanAndCovariance["norm_difference_between_means"], ":", color="tab:green",
                label="Trapezoidal Direct Collocation x Noise distribution approx.")
    axs[1].plot(data_Trapezoidal_MeanAndCovariance["norm_difference_between_covs"], ":", color="tab:green",
                label="Trapezoidal Direct Collocation x Noise distribution approx.")

if data_DirectMultipleShooting_MeanAndCovariance is not None:
    axs[0].plot(data_DirectMultipleShooting_MeanAndCovariance["norm_difference_between_means"], ":", color="tab:orange",
                label="Direct Multiple Shooting x Noise distribution approx.")
    axs[1].plot(data_DirectMultipleShooting_MeanAndCovariance["norm_difference_between_covs"], ":", color="tab:orange",
                label="Direct Multiple Shooting x Noise distribution approx.")

if data_VariationalPolynomial_MeanAndCovariance is not None:
    axs[0].plot(data_VariationalPolynomial_MeanAndCovariance["norm_difference_between_means"], ":", color="tab:purple",
                label="Polynomial DMaOC x Noise distribution approx.")
    axs[1].plot(data_VariationalPolynomial_MeanAndCovariance["norm_difference_between_covs"], ":", color="tab:purple",
                label="Polynomial DMaOC x Noise distribution approx.")

axs[0].set_title(r"$||\bar{q}_{opt} - \bar{q}_{sim}||_{2}$")
axs[0].set_xlabel("Shooting node")
axs[0].set_ylabel("Difference")
axs[0].set_yscale("log")
axs[0].legend(bbox_to_anchor=(1.1, -0.15), loc="upper center", ncol=2)

axs[1].set_title(r"$||P_{opt} - P_{sim}||_{Frobenius}$")
axs[1].set_xlabel("Shooting node")
axs[1].set_yscale("log")

plt.subplots_adjust(bottom=0.3, left=0.1, right=0.95, top=0.95)
# plt.tight_layout()
plt.savefig("results/vertebrate_arm_analysis.png", dpi=300)
plt.show()


# --- Create the LaTeX result table --- #
import colorsys
DATA = {
    "PDC": {
        "D": PDC_D,
        "NS": PDC_NS[nb_random_chosen],
        "NDA": PDC_MAC,
    },
    "TDC": {
        "D": TDC_D,
        "NS": TDC_NS[nb_random_chosen],
        "NDA": TDC_MAC,
    },
    "DMS": {
        "D": DMS_D,
        "NS": DMS_NS[nb_random_chosen],
        "NDA": DMS_MAC,
    },
    "TDMaOC": {
        "D": TDMaOC_D,
        "NS": TDMaOC_NS[nb_random_chosen],
        "NDA": TDMaOC_MAC,
    },
    "PDMaOC": {
        "D": PDMaOC_D,
        "NS": PDMaOC_NS[nb_random_chosen],
        "NDA": PDMaOC_MAC,
    },
}

# Column order for the numeric metrics
METRIC_COLS = ["nb var", "nb const", "time", "nb iter", "time per iter", "cost"]

# Column headers
METRIC_HEADERS = [r"\# var.", r"\# const.", "Time [s]", r"\# iter.", "Time/iter [s]", "Cost"]


# ── Color helpers ─────────────────────────────────────────────────────────────


def value_to_rgb(value: float, vmin: float, vmax: float):
    """Map value in [vmin, vmax] → RGB (green = low, red = high), log scale."""
    if vmax != vmin and value > 0 and vmin > 0:
        t = (np.log(vmax) - np.log(value)) / (np.log(vmax) - np.log(vmin))
    else:
        t = 0.5
    hue = t * 120 / 360
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
    return int(r * 255), int(g * 255), int(b * 255)


# ── Flatten data & compute per-metric min/max ─────────────────────────────────

flat_rows = []  # (trans, title, metrics)
for trans, titles in DATA.items():
    for title, metrics in titles.items():
        flat_rows.append((trans, title, metrics))

col_values = {m: [] for m in METRIC_COLS}
col_for_min_max = {m: [] for m in METRIC_COLS}
for _, titles, metrics in flat_rows:
    for m in METRIC_COLS:
        if metrics.get(m) is not None:
            col_values[m].append(float(metrics[m]))
            if titles == "D":
                col_for_min_max[m].append(np.nan)
            else:
                col_for_min_max[m].append(float(metrics[m]))
        else:
            col_values[m].append(np.nan)
            col_for_min_max[m].append(np.nan)

col_min = {m: np.nanmin(v) for m, v in col_for_min_max.items()}
col_max = {m: np.nanmax(v) for m, v in col_for_min_max.items()}


# ── Build color definitions and table rows ────────────────────────────────────

color_defs = []
table_rows = []

trans_seen = {}

for row_idx, (trans, title, metrics) in enumerate(flat_rows):
    span = sum(1 for t, _, _ in flat_rows if t == trans)
    cells = []

    # Transcription cell (multirow on first occurrence)
    if trans not in trans_seen:
        cells.append(rf"\multirow{{{span}}}{{*}}{{{trans}}}")
        trans_seen[trans] = True
    else:
        cells.append("")

    # Title cell
    cells.append(title)

    # Numeric / colored cells
    for col_idx, metric in enumerate(METRIC_COLS):
        val = metrics.get(metric)
        if val is not None:
            # Get the value
            fval = float(val)
            display = f"{fval:.2f}" if isinstance(val, float) else str(int(val))

            if title == "D":
                # No background color for deterministic
                cells.append(f"{display}")
            else:
                r, g, b = value_to_rgb(fval, col_min[metric], col_max[metric])
                cname = f"cell{row_idx}m{col_idx}"
                color_defs.append(rf"\definecolor{{{cname}}}{{RGB}}{{{r},{g},{b}}}")
                cells.append(rf"\cellcolor{{{cname}}}{display}")
        else:
            cells.append("")

    # Draw \hline only after last row of each transcription group
    is_last_in_group = row_idx == len(flat_rows) - 1 or flat_rows[row_idx + 1][0] != trans
    hline = r" \hline" if is_last_in_group else ""
    table_rows.append("    " + " & ".join(cells) + rf" \\{hline}")


# ── Assemble full LaTeX document ──────────────────────────────────────────────

col_spec = "|c|c|" + "c|" * len(METRIC_COLS)
color_block = "\n".join(color_defs)

header_cells = [
    r"\textbf{Trans.}",
    r"\textbf{Noise}",
] + [rf"\textbf{{{h}}}" for h in METRIC_HEADERS]

header_row = "    " + " & ".join(header_cells) + r" \\ \hline"

latex = (
    r"\documentclass{article}" + "\n"
    r"\usepackage[table]{xcolor}" + "\n"
    r"\usepackage{multirow}" + "\n"
    r"\usepackage{array}" + "\n"
    r"\usepackage{booktabs}" + "\n"
    "\n"
    "% Auto-generated cell colours\n" + color_block + "\n"
    "\n"
    r"\begin{document}" + "\n"
    "\n"
    r"\begin{table}[ht]" + "\n"
    r"  \centering" + "\n"
    r"  \caption{Comparison of the efficiency of all implementations. The cells are color coded from red (undesirable) to green (desirable).}"
    + "\n"
    r"  \renewcommand{\arraystretch}{1.4}" + "\n"
    r"  {\footnotesize" + "\n"
    rf"  \begin{{tabular}}{{{col_spec}}}" + "\n"
    r"    \hline" + "\n" + header_row + "\n"
    r"    \hline" + "\n" + "\n".join(table_rows) + "\n"
    r"  \end{tabular}}" + "\n"
    r"\end{table}" + "\n"
    "\n"
    r"\end{document}" + "\n"
)

OUTPUT_FILE = "results/table.tex"
with open(OUTPUT_FILE, "w") as fh:
    fh.write(latex)

print(f"LaTeX file written to: {OUTPUT_FILE}")
print()
print("Customise the DATA dict at the top of the script and re-run.")
print("Compile with:  pdflatex table.tex")


# --- Plot the initial constraints distribution --- #
# nb_random_chosen = f"nb_random_30"
#
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# for i_percent in range(100):
#     if i_percent == 0:
#         axs[0].bar(i_percent + 0, np.percentile(PDC_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent),
#                    color="tab:red", alpha=0.5, width=0.2, label="Polynomial Direct Collocation x Noise Sampling")
#         axs[0].bar(i_percent + 0.2, np.percentile(TDC_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent),
#                    color="tab:green", alpha=0.5, width=0.2, label="Trapezoidal Direct Collocation x Noise Sampling")
#         axs[0].bar(i_percent + 0.4, np.percentile(DMS_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent),
#                    color="tab:orange", alpha=0.5, width=0.2, label="Direct Multiple Shooting x Noise Sampling")
#         axs[0].bar(i_percent + 0.6, np.percentile(TDMaOC_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent),
#                    color="tab:blue", alpha=0.5, width=0.2, label="Trapezoidal DMaOC x Noise Sampling")
#         axs[0].bar(i_percent + 0.8, np.percentile(PDMaOC_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent),
#                    color="tab:purple", alpha=0.5, width=0.2, label="Polynomial DMaOC x Noise Sampling")
#
#         axs[1].bar(i_percent + 0, np.percentile(PDC_MAC["g_without_bounds_at_init"], i_percent), color="tab:red", alpha=0.5, width=0.2, label="Polynomial Direct Collocation x Noise distribution approx.")
#         axs[1].bar(i_percent + 0.2, np.percentile(TDC_MAC["g_without_bounds_at_init"], i_percent), color="tab:green", alpha=0.5, width=0.2, label="Trapezoidal Direct Collocation x Noise distribution approx.")
#         axs[1].bar(i_percent + 0.4, np.percentile(DMS_MAC["g_without_bounds_at_init"], i_percent), color="tab:orange", alpha=0.5, width=0.2, label="Direct Multiple Shooting x Noise distribution approx.")
#         axs[1].bar(i_percent + 0.8, np.percentile(PDMaOC_MAC["g_without_bounds_at_init"], i_percent), color="tab:purple", alpha=0.5, width=0.2, label="Polynomial DMaOC x Noise distribution approx.")
#     else:
#         axs[0].bar(i_percent + 0, np.percentile(PDC_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent), color="tab:red", alpha=0.5, width=0.2)
#         axs[0].bar(i_percent + 0.2, np.percentile(TDC_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent), color="tab:green", alpha=0.5, width=0.2)
#         axs[0].bar(i_percent + 0.4, np.percentile(DMS_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent), color="tab:orange", alpha=0.5, width=0.2)
#         axs[0].bar(i_percent + 0.6, np.percentile(TDMaOC_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent), color="tab:blue", alpha=0.5, width=0.2)
#         axs[0].bar(i_percent + 0.8, np.percentile(PDMaOC_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent), color="tab:purple", alpha=0.5, width=0.2)
#
#         axs[1].bar(i_percent + 0, np.percentile(PDC_MAC["g_without_bounds_at_init"], i_percent), color="tab:red", alpha=0.5, width=0.2)
#         axs[1].bar(i_percent + 0.2, np.percentile(TDC_MAC["g_without_bounds_at_init"], i_percent), color="tab:green", alpha=0.5, width=0.2)
#         axs[1].bar(i_percent + 0.4, np.percentile(DMS_MAC["g_without_bounds_at_init"], i_percent), color="tab:orange", alpha=0.5, width=0.2)
#         axs[1].bar(i_percent + 0.8, np.percentile(PDMaOC_MAC["g_without_bounds_at_init"], i_percent), color="tab:purple", alpha=0.5, width=0.2)
#
# axs[0].legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=1)
# axs[0].set_xlabel("Percentile")
# axs[0].set_ylabel("Constraint violation at initialization")
# axs[0].set_title("Noise sampling")
# axs[0].set_yscale("log")
#
# axs[1].legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=1)
# axs[1].set_xlabel("Percentile")
# axs[1].set_title("Noise distribution approximation")
# axs[1].set_yscale("log")
#
# plt.subplots_adjust(bottom=0.32, left=0.1, right=0.95, top=0.95)
# plt.savefig("results/vertebrate_arm_initial_constraints.png", dpi=300)
# plt.show()



nb_random_chosen = f"nb_random_30"

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(np.arange(101), np.array([np.percentile(PDC_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent) for i_percent in range(101)]),
           "--", color="tab:red", label="Polynomial Direct Collocation x Noise Sampling")
ax.plot(np.arange(101), np.array([np.percentile(TDC_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent) for i_percent in range(101)]),
           "--", color="tab:green", label="Trapezoidal Direct Collocation x Noise Sampling")
ax.plot(np.arange(101), np.array([np.percentile(DMS_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent) for i_percent in range(101)]),
           "--", color="tab:orange", label="Direct Multiple Shooting x Noise Sampling")
ax.plot(np.arange(101), np.array([np.percentile(TDMaOC_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent) for i_percent in range(101)]),
           "--", color="tab:blue", label="Trapezoidal DMaOC x Noise Sampling")
ax.plot(np.arange(101), np.array([np.percentile(PDMaOC_NS[nb_random_chosen]["g_without_bounds_at_init"], i_percent) for i_percent in range(101)]),
           "--", color="tab:purple", label="Polynomial DMaOC x Noise Sampling")

ax.plot(np.arange(101), np.array([np.percentile(PDC_MAC["g_without_bounds_at_init"], i_percent) for i_percent in range(101)]), ":", color="tab:red", label="Polynomial Direct Collocation x Noise distribution approx.")
ax.plot(np.arange(101), np.array([np.percentile(TDC_MAC["g_without_bounds_at_init"], i_percent) for i_percent in range(101)]), ":", color="tab:green", label="Trapezoidal Direct Collocation x Noise distribution approx.")
ax.plot(np.arange(101), np.array([np.percentile(DMS_MAC["g_without_bounds_at_init"], i_percent) for i_percent in range(101)]), ":", color="tab:orange", label="Direct Multiple Shooting x Noise distribution approx.")
ax.plot(np.arange(101), np.array([np.percentile(PDMaOC_MAC["g_without_bounds_at_init"], i_percent) for i_percent in range(101)]), ":", color="tab:purple", label="Polynomial DMaOC x Noise distribution approx.")

ax.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)
ax.set_xlabel("Percentile")
ax.set_ylabel("Constraint violation at initialization")
ax.set_yscale("log")
ax.set_ylim(1e-8, 1e1)
ax.set_xlim(0, 100)

plt.subplots_adjust(bottom=0.33, left=0.1, right=0.95, top=0.95)
plt.savefig("results/vertebrate_arm_initial_constraints.png", dpi=300)
plt.show()



# --- Plot the final simulated states distribution --- #
time_vector = data_DirectMultipleShooting_Noise[nb_random_chosen]["time_vector"]
states_opt_mean = data_DirectMultipleShooting_Noise[nb_random_chosen]["states_opt_mean"]
states_opt_array = data_DirectMultipleShooting_Noise[nb_random_chosen]["states_opt_array"]
controls_opt_array = data_DirectMultipleShooting_Noise[nb_random_chosen]["controls_opt_array"]
ref_opt_array = data_DirectMultipleShooting_Noise[nb_random_chosen]["ref_opt_array"]

ocp_example = VertebrateArm(nb_random=30, seed=0)
ocp = prepare_ocp(
    ocp_example=ocp_example,
    dynamics_transcription=DirectMultipleShooting(),
    discretization_method=NoiseDiscretization(dynamics_transcription=DirectMultipleShooting()),
)

x_simulated = reintegrate(
    time_vector=time_vector,
    states_opt_mean=states_opt_mean,
    states_opt_array=states_opt_array,
    controls_opt_array=controls_opt_array,
    ref_opt_array=ref_opt_array,
    ocp=ocp,
    n_simulations=5000,
    save_path="results/vertebrate_arm_final_states_distribution_reintegration.pkl",
    plot_flag=False,
)

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s=np.ones_like(x), alpha=0.5)

    # now determine nice limits by hand:
    binwidth = 0.05
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

fig = plt.figure(layout='constrained')
ax = fig.add_subplot()

ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
scatter_hist(
    x_simulated[0, -1, :],
    x_simulated[1, -1, :],
    ax,
    ax_histx,
    ax_histy,
)

ax.set_xlim(
    np.min(x_simulated[:2, -1, :]) - 0.05,
    np.max(x_simulated[:2, -1, :]) + 0.05,
)
ax.set_ylim(
    np.min(x_simulated[:2, -1, :]) - 0.05,
    np.max(x_simulated[:2, -1, :]) + 0.05,
)
ax.set_xlabel("Shounder angle at final time")
ax.set_ylabel("Elbow angle at final time")
ax.set_aspect('equal')
plt.savefig("results/vertebrate_arm_final_states_distribution.png", dpi=300, bbox_inches='tight')
plt.show()