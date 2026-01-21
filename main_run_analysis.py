import matplotlib.pyplot as plt
import numpy as np
import casadi as cas
import pickle


# --- Plot the results for the ObstacleAvoidance problem

with open(
        "/home/charbie/Documents/Programmation/SOCP/results/ObstacleAvoidance_DirectCollocationPolynomial_MeanAndCovariance_CVG_1p0e-06_2026-01-21-14-09_robustified.pkl",
        "rb",
) as f:
    data_DCP_MAC = pickle.load(f)
    n_shooting = data_DCP_MAC["difference_between_means"].shape[0] - 1

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(np.linspace(0, n_shooting, n_shooting+1), np.zeros((n_shooting+1, )), "--k")
axs[0].plot(data_DCP_MAC["difference_between_means"], '-r', label="DCP + MAC (Gillis)")
axs[0].set_title(r"$|\bar{x}_{opt} - \bar{x}_{sim}|$")
axs[0].set_xlabel("Shooting node")
axs[0].set_ylabel("Difference")

axs[1].plot(np.linspace(0, n_shooting, n_shooting+1), np.zeros((n_shooting+1)), "--k")
axs[1].plot(data_DCP_MAC["difference_between_covs"], '-r', label="DCP + MAC (Gillis)")
axs[1].set_title(r"$|Det(P_{opt}) - Det(P_{sim})|$")
axs[1].set_xlabel("Shooting node")

axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("obstacle_avoidance_analysis.png", dpi=300)
plt.show()