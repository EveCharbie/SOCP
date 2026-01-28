import numpy as np


def estimate_covariance(x_mean_simulated: np.ndarray, x_simulated: np.ndarray) -> np.ndarray:

    n_shooting = x_mean_simulated.shape[1] - 1
    n_simulations = x_simulated.shape[2]
    cov_simulated = np.zeros((x_mean_simulated.shape[0], x_mean_simulated.shape[0], x_mean_simulated.shape[1]))
    for i_node in range(n_shooting + 1):
        diff = x_simulated[:, i_node, :] - x_mean_simulated[:, i_node : i_node + 1]
        cov_simulated[:, :, i_node] = (diff @ diff.T) / (n_simulations - 1)

    return cov_simulated
