from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
from scipy.integrate import solve_ivp


def dynamics_wrapper(t, x, u, ref, noise, ocp_example):
    return ocp_example.model.dynamics(x, u, ref, noise).flatten()


def reintegrate(
    time_vector: np.ndarray,
    states_opt_mean: np.ndarray,
    controls_opt: np.ndarray,
    ocp: dict[str, Any],
    n_simulations: int,
    save_path: str,
    plot_flag: bool = True,
) -> np.ndarray:

    n_shooting = ocp["n_shooting"]
    nb_states = ocp["model"].nb_states
    dt = time_vector[1] - time_vector[0]

    # Reintegrate the solution with noise
    x_simulated = np.zeros((nb_states, n_shooting + 1, n_simulations))
    for i_simulation in range(n_simulations):

        np.random.seed(i_simulation)
        noise_magnitude = cas.vertcat(ocp["motor_noise_magnitude"], ocp["sensory_noise_magnitude"])

        # Initialize the states with the mean at the first node
        x_simulated[:, 0, i_simulation] = states_opt_mean[:, 0]

        for i_node in range(n_shooting):
            x_prev = x_simulated[:, i_node, i_simulation].flatten()
            u_prev = controls_opt[:, i_node]
            noise_this_time = np.random.normal(0, noise_magnitude, ocp["ocp_example"].nb_noises)

            ref = ocp["ocp_example"].get_reference(
                model=ocp["ocp_example"],
                x=states_opt_mean[:, i_node],
                u=controls_opt[:, i_node],
            )

            sol = solve_ivp(
                fun=lambda t, x: dynamics_wrapper(
                    t, x, u_prev, ref, noise_this_time, ocp["ocp_example"]
                ),
                t_span=(0.0, dt),
                y0=x_prev,
                method="DOP853",
                rtol=1e-6,
                atol=1e-8,
            )

            # Save next state (end of interval)
            x_simulated[:, i_node + 1, i_simulation] = sol.y[:, -1]

    if plot_flag:
        nrows = len(ocp.states_initial_guesses.keys())
        ncols = 0
        for key in ocp.states_initial_guesses.keys():
            if ocp.states_initial_guesses[key].shape[0] > ncols:
                ncols = ocp.states_initial_guesses[key].shape[0]
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        i_state = 0
        for i_row, key in enumerate(ocp.states_initial_guesses.keys()):
            for i_col in range(ocp.states_initial_guesses[key].shape[0]):
                for i_simulation in range(n_simulations):
                    axs[i_row, i_state].plot(
                        time_vector,
                        x_simulated[i_state, :, i_simulation],
                        color="k",
                        linewidth=0.5,
                    )
                axs[i_row, i_state].plot(
                    time_vector,
                    states_opt_mean[i_state, :],
                    color="tab:blue",
                    linewidth=2,
                    label="Mean optimal trajectory",
                )
                axs[i_row, i_state].set_xlabel("Time [s]")
                i_state += 1

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        # plt.close()

    return x_simulated
