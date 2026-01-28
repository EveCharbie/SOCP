import matplotlib.pyplot as plt
import numpy as np
import casadi as cas
import pickle

from socp import ObstacleAvoidance, MeanAndCovariance, NoiseDiscretization

# --- Plot the results for the ObstacleAvoidance problem

with open(
    "/home/charbie/Documents/Programmation/SOCP/results/ObstacleAvoidance_DirectCollocationPolynomial_MeanAndCovariance_CVG_1p0e-06_2026-01-21-14-09_robustified.pkl",
    "rb",
) as f:
    data_DC_MAC = pickle.load(f)
    n_shooting = data_DC_MAC["difference_between_means"].shape[0] - 1

with open(
    "/home/charbie/Documents/Programmation/SOCP/results/ObstacleAvoidance_DirectMultipleShooting_NoiseDiscretization_CVG_1p0e-06_2026-01-21-18-28_robustified.pkl",
    "rb",
) as f:
    data_DMS_N = pickle.load(f)


def plot_techniques(ax, data_saved, color_opt, color_sim):
    obstacle_avoidance = ObstacleAvoidance(is_robustified=True)

    q_mean = data_saved["states_opt_mean"][:2, :]
    q_simulated = data_saved["x_simulated"][:2, :, :]
    covariance_simulated = data_saved["covariance_simulated"]
    cov_opt = data_saved["cov_opt_array"]

    for i_ellipse in range(2):
        a = obstacle_avoidance.model.super_ellipse_a[i_ellipse]
        b = obstacle_avoidance.model.super_ellipse_b[i_ellipse]
        n = obstacle_avoidance.model.super_ellipse_n[i_ellipse]
        x_0 = obstacle_avoidance.model.super_ellipse_center_x[i_ellipse]
        y_0 = obstacle_avoidance.model.super_ellipse_center_y[i_ellipse]

        X, Y, Z = obstacle_avoidance.superellipse(a, b, n, x_0, y_0)

        ax.contourf(X, Y, Z, levels=[-1000, 0], colors=["#DA1984"], alpha=0.5)

    q_simulated_mean = np.mean(q_simulated, axis=2)
    for i_node in range(n_shooting):
        obstacle_avoidance.draw_cov_ellipse(cov=cov_opt[:2, :2, i_node], pos=q_mean[:, i_node], ax=ax, color=color_opt)
        obstacle_avoidance.draw_cov_ellipse(
            cov=covariance_simulated[:2, :2, i_node],
            pos=q_simulated_mean[:, i_node],
            ax=ax,
            color=color_sim,
        )

    ax.plot(q_mean[0, :], q_mean[1, :], color=color_opt)
    ax.plot(q_simulated_mean[0, :], q_simulated_mean[1, :], color=color_sim)

    return


# --- Plot the comparison metrics --- #

fig, axs = plt.subplots(1, 4, figsize=(12, 6))

axs[0].plot(np.linspace(0, n_shooting, n_shooting + 1), np.zeros((n_shooting + 1,)), "--k")
axs[0].plot(data_DC_MAC["difference_between_means"], "-", color="tab:red", label="DC & Mean+COV (Gillis)")
axs[0].plot(data_DMS_N["difference_between_means"], "-", color="tab:purple", label="DMS & Noise")
axs[0].set_title(r"$|\bar{x}_{opt} - \bar{x}_{sim}|$")
axs[0].set_xlabel("Shooting node")
axs[0].set_ylabel("Difference")

axs[1].plot(np.linspace(0, n_shooting, n_shooting + 1), np.zeros((n_shooting + 1)), "--k")
axs[1].plot(data_DC_MAC["norm_difference_between_covs"], "--", color="tab:red", label="DC & Mean+COV (Gillis)")
axs[1].plot(data_DMS_N["norm_difference_between_covs"], "--", color="tab:purple", label="DMS & Noise")
axs[1].set_title(r"$|(||P_{opt} - P_{sim}||_{fro})|$")

axs[1].plot(data_DC_MAC["difference_between_covs_det"], "-", color="tab:red", label="DC & Mean+COV (Gillis)")
axs[1].plot(data_DMS_N["difference_between_covs_det"], "-", color="tab:purple", label="DMS & Noise")
axs[1].set_title(r"$|Det(P_{opt}) - Det(P_{sim})|$")
axs[1].set_xlabel("Shooting node")

axs[2].bar(0, data_DC_MAC["computation_time"], width=0.4, color="tab:red", label="DC & Mean+COV (Gillis)")
axs[2].bar(0.5, data_DMS_N["computation_time"], width=0.4, color="tab:purple", label="DMS & Noise")
axs[2].set_xlabel("Computation Time [s]")

axs[3].bar(0, data_DC_MAC["optimal_cost"], width=0.4, color="tab:red", alpha=0.5, label="DC Simulation")
axs[3].bar(0.5, data_DMS_N["optimal_cost"], width=0.4, color="tab:purple", alpha=0.5, label="DMS Simulation")
axs[3].set_xlabel("Optimal Cost")

axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("results/obstacle_avoidance_analysis.png", dpi=300)


# --- Plot the optimal trajectories --- #

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

plot_techniques(ax, data_DC_MAC, color_opt="tab:red", color_sim="pink")
plot_techniques(ax, data_DMS_N, color_opt="tab:purple", color_sim="violet")

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("results/obstacle_avoidance_solutions.png", dpi=300)
plt.show()
