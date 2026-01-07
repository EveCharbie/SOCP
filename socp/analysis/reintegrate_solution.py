from typing import Any
import numpy as np
import matplotlib.pyplot as plt


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
    n_random = ocp["model"].n_random
    n_states = ocp["model"].nb_states
    n_motor_noises = ocp["model"].nb_q * n_random
    n_sensory_noises = 4 * n_random
    nb_noises = n_motor_noises + n_sensory_noises
    noise_magnitude = np.array(
        np.array(ocp["model"].motor_noise_magnitude)
        .reshape(
            -1,
        )
        .tolist()
        * n_random
        + np.array(ocp["model"].hand_sensory_noise_magnitude)
        .reshape(
            -1,
        )
        .tolist()
        * n_random
    )

    # Reintegrate the solution with noise
    x_simulated = np.zeros((n_simulations, n_states, n_shooting + 1))
    for i_simulation in range(n_simulations):

        np.random.seed(i_simulation)

        # Initialize the states with the mean at the first node
        x_simulated[i_simulation, :, 0] = states_opt_mean[:, 0]

        for i_node in range(n_shooting):
            x_prev = x_simulated[i_simulation, :, i_node].flatten()
            u_prev = controls_opt[:, i_node]
            noise_this_time = np.random.normal(0, noise_magnitude, nb_noises)

            x_simulated[i_simulation, :, i_node] = ocp["model"].dynamics(x_prev, u_prev, noise_this_time).flatten()

    if plot_flag:
        nrows = len(ocp.states_initial_guesses.keys())
        ncols = 0
        for key in ocp.states_initial_guesses.keys():
            if ocp.states_initial_guesses[key].shape[0] > n_cols:
                ncols = ocp.states_initial_guesses[key].shape[0]
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        i_state = 0
        for i_row, key in enumerate(ocp.states_initial_guesses.keys()):
            for i_col in range(ocp.states_initial_guesses[key].shape[0]):
                for i_simulation in range(n_simulations):
                    axs[i_row, i_state].plot(
                        time_vector,
                        x_simulated[i_simulation, i_state, :],
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
