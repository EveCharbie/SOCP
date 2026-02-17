"""
This example implements a stochastic optimal control problem for mass-point circling around obstacles.
The goal is to find a time-optimal stable periodic trajectory.
The controls are coordinates of a quide point (the mass is attached to this guide point with a sping).
This example was taken from Gillis et al. 2013.
"""

from typing import Any
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

from .example_abstract import ExampleAbstract
from ..constraints import Constraints
from ..models.cart_pole_model import CartPoleModel
from ..models.model_abstract import ModelAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract
from ..transcriptions.mean_and_covariance import MeanAndCovariance
from ..transcriptions.noises_abstract import NoisesAbstract
from ..transcriptions.transcription_abstract import TranscriptionAbstract
from ..transcriptions.variables_abstract import VariablesAbstract
from ..transcriptions.variational import Variational
from ..transcriptions.variational_polynomial import VariationalPolynomial


class CartPole(ExampleAbstract):
    def __init__(self) -> None:
        super().__init__()  # Does nothing

        self.nb_random = 10
        self.n_threads = 7
        self.n_simulations = 100
        self.seed = 0
        self.model = CartPoleModel(self.nb_random)

        self.final_time = 1.0
        self.min_time = 0.5
        self.max_time = 2.0
        self.n_shooting = 40
        self.initial_state_variability = np.array([0.001, 0.001, 0.001, 0.001])
        self.initial_covariance = np.diag((self.initial_state_variability**2).tolist())

        # Solver options
        self.tol = 1e-8
        self.max_iter = 10000

    @property
    def name(self) -> str:
        return "CartPole"

    def get_bounds_and_init(
        self,
        n_shooting: int,
        nb_collocation_points: int,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        """
        Get all the bounds and initial guesses for the states and controls.
        """
        nb_q = self.model.nb_q

        # Q
        lbq = np.zeros((nb_q, n_shooting + 1))
        lbq[0, :] = -1
        lbq[1, :] = -2*np.pi
        lbq[:, 0] = -0.5  # Start with zero translation or rotation
        ubq = np.zeros((nb_q, n_shooting + 1))
        ubq[0, :] = 5
        ubq[1, :] = 2*np.pi
        ubq[:, 0] = 0.5  # Start with zero translation or rotation

        # Zero initial guess
        q0 = np.zeros((nb_q, n_shooting + 1))

        # Qdot
        lbqdot = np.ones((nb_q, n_shooting + 1)) * -10*np.pi
        lbqdot[:, 0] = -0.5  # Start with zero velocity
        ubqdot = np.ones((nb_q, n_shooting + 1)) * 10*np.pi
        ubqdot[:, 0] = 0.5  # Start with zero velocity

        # Zero initial guess
        qdot0 = np.zeros((nb_q, n_shooting + 1))

        states_lower_bounds = {
            "q": lbq,
            "qdot": lbqdot,
        }
        states_upper_bounds = {
            "q": ubq,
            "qdot": ubqdot,
        }
        states_initial_guesses = {
            "q": q0,
            "qdot": qdot0,
        }

        collocation_points_initial_guesses = {}
        if nb_collocation_points != 0:
            qz0 = np.zeros((nb_q, nb_collocation_points, n_shooting + 1))
            qdotz0 = np.zeros((nb_q, nb_collocation_points, n_shooting + 1))

            collocation_points_initial_guesses = {
                "q": qz0,
                "qdot": qdotz0,
            }

        # u
        lbu = np.ones((1, n_shooting + 1)) * -100
        ubu = np.ones((1, n_shooting + 1)) * 100
        u0 = np.zeros((1, n_shooting + 1))

        controls_lower_bounds = {
            "u": lbu,
        }
        controls_upper_bounds = {
            "u": ubu,
        }
        controls_initial_guesses = {
            "u": u0,
        }

        return (
            states_lower_bounds,
            states_upper_bounds,
            states_initial_guesses,
            controls_lower_bounds,
            controls_upper_bounds,
            controls_initial_guesses,
            collocation_points_initial_guesses,
        )

    def get_noises_magnitude(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the motor and sensory noise magnitude.
        """
        motor_noise_magnitude = np.array([0.001])
        return motor_noise_magnitude, None

    def set_specific_constraints(
        self,
        model: ModelAbstract,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        constraints: Constraints,
    ) -> None:

        # Initial covariance is imposed
        if isinstance(dynamics_transcription, (Variational, VariationalPolynomial)):
            nb_states = model.nb_q
        else:
            nb_states = variables_vector.nb_states

        cov_matrix_0 = discretization_method.get_covariance(variables_vector, 0, is_matrix=True)[:nb_states, :nb_states]
        constraints.add(
            g=variables_vector.reshape_matrix_to_vector(cov_matrix_0 - self.initial_covariance[:nb_states, :nb_states]),
            lbg=[0] * (nb_states * nb_states),
            ubg=[0] * (nb_states * nb_states),
            g_names=["initial_covariance"] * (nb_states * nb_states),
            node=0,
        )

        # Initial mean states are imposed
        x_intial = np.array([0, 0, 0, 0])[:nb_states]
        mean_states = discretization_method.get_mean_states(variables_vector, 0)[:nb_states]
        constraints.add(
            g=mean_states - x_intial,
            lbg=[0] * nb_states,
            ubg=[0] * nb_states,
            g_names=["final_mean_states"] * nb_states,
            node=0,
        )

        # Final mean states are imposed
        x_final = np.array([0, np.pi, 0, 0])[:nb_states]
        mean_states = discretization_method.get_mean_states(variables_vector, self.n_shooting)[:nb_states]
        constraints.add(
            g=mean_states - x_final,
            lbg=[0] * nb_states,
            ubg=[0] * nb_states,
            g_names=["final_mean_states"] * nb_states,
            node=self.n_shooting,
        )

        return

    def get_specific_objectives(
        self,
        model: ModelAbstract,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> cas.SX:

        dt = variables_vector.get_time() / self.n_shooting

        # Minimize controls
        j_controls = 0
        for i_node in range(self.n_shooting + 1):
            j_controls += cas.sum1(variables_vector.get_controls(i_node) ** 2) * dt

        # Minimize final variability
        cov_matrix = discretization_method.get_covariance(variables_vector, self.n_shooting, is_matrix=True)
        j_variability = cas.sum1(cas.sum2(cov_matrix.T @ cov_matrix))

        # Regularization on time
        j_time = variables_vector.get_time()

        return 100 * j_variability + j_controls + j_time

    # --- helper functions --- #
    def get_marker_position(self, q: np.ndarray) -> np.ndarray:
        """
        Get the position of the marker (the end of the pendulum) given the state q.
        """
        x_cart = q[0]
        theta = q[1]
        pole_length = self.model.pole_length

        if theta < np.pi/2:
            x_marker = x_cart + pole_length * np.sin(theta)
            y_marker = -pole_length * np.cos(theta)
        elif theta < np.pi:
            x_marker = x_cart + pole_length * np.cos(theta -np.pi/2)
            y_marker = pole_length * np.sin(theta -np.pi/2)
        elif theta < 3*np.pi/2:
            x_marker = x_cart - pole_length * np.sin(theta - np.pi)
            y_marker = pole_length * np.cos(theta - np.pi)
        else:
            x_marker = x_cart - pole_length * np.cos(theta - 3*np.pi/2)
            y_marker = -pole_length * np.sin(theta - 3*np.pi/2)
        return np.array([x_marker, y_marker])

    def specific_plot_results(
        self,
        ocp: dict[str, Any],
        data_saved: dict[str, Any],
        fig_save_path: str,
    ):
        """
        This function plots the reintegration of the optimal solution considering the motor noise.
        The plot compares the covariance obtained numerically by resimulation and the covariance obtained by the optimal
        control problem.
        """
        n_shooting = ocp["n_shooting"]
        states_opt_mean = data_saved["states_opt_mean"]

        q_mean = states_opt_mean[ocp["ocp_example"].model.q_indices, :]
        q_init = data_saved["states_init_array"][ocp["ocp_example"].model.q_indices, :]
        u_opt = data_saved["controls_opt_array"][ocp["ocp_example"].model.u_indices, :]
        q_opt = data_saved["states_opt_array"][ocp["ocp_example"].model.q_indices, :]
        q_simulated = data_saved["x_simulated"][: ocp["ocp_example"].model.nb_q, :, :]
        n_simulations = q_simulated.shape[2]
        covariance_simulated = data_saved["covariance_simulated"]
        cov_opt = data_saved["cov_opt_array"]

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        if isinstance(ocp["discretization_method"], MeanAndCovariance):
            marker_position_opt = np.zeros((2, n_shooting + 1))
            for i_node in range(n_shooting + 1):
                marker_position_opt[:, i_node] = self.get_marker_position(q_opt[:, i_node])

            ax.plot(marker_position_opt[0, :], marker_position_opt[1, :], "--k", label="Initial guess", linewidth=0.5, alpha=0.3)
            ax.plot(marker_position_opt[0, 0], marker_position_opt[1, 0], "og", label="Optimal initial node")
        else:
            marker_position_opt = np.zeros((2, n_shooting + 1, ocp["ocp_example"].nb_random))
            for i_random in range(ocp["ocp_example"].nb_random):
                for i_node in range(n_shooting + 1):
                    marker_position_opt[:, i_node, i_random] = self.get_marker_position(q_opt[:, i_node, i_random])

                if i_random == 0:
                    ax.plot(marker_position_opt[0, 0, i_random], marker_position_opt[1, 0, i_random], "og", label="Optimal initial node")
                else:
                    ax.plot(marker_position_opt[0, 0, i_random], marker_position_opt[1, 0, i_random], "og")
                ax.plot(marker_position_opt[0, :, i_random], marker_position_opt[1, :, i_random], "-", color="g", linewidth=0.5, alpha=0.3)


        marker_position_simulated = np.zeros((2, n_shooting + 1, n_simulations))
        for i_simulation in range(n_simulations):
            for i_node in range(n_shooting + 1):
                marker_position_simulated[:, i_node, i_simulation] = self.get_marker_position(q_simulated[:, i_node, i_simulation])

        q_simulated_mean = np.mean(q_simulated, axis=2)
        for i_node in range(n_shooting):
            if i_node == 0:
                # self.draw_cov_ellipse(
                #     cov=cov_opt[:2, :2, i_node], pos=q_mean[:, i_node], ax=ax[0], color="b", label="Cov optimal"
                # )
                ax.plot(
                    marker_position_simulated[0, i_node, :], marker_position_simulated[1, i_node, :], ".r", markersize=1, label="Noisy integration"
                )
                # self.draw_cov_ellipse(
                #     cov=covariance_simulated[:2, :2, i_node],
                #     pos=q_simulated_mean[:, i_node],
                #     ax=ax[0],
                #     color="r",
                #     label="Cov simulated",
                # )
            else:
                # self.draw_cov_ellipse(cov=cov_opt[:2, :2, i_node], pos=q_mean[:, i_node], ax=ax[0], color="b")
                ax.plot(marker_position_simulated[0, i_node, :], marker_position_simulated[1, i_node, :], ".r", markersize=1)
                # self.draw_cov_ellipse(
                #     cov=covariance_simulated[:2, :2, i_node],
                #     pos=q_simulated_mean[:, i_node],
                #     ax=ax[0],
                #     color="r",
                # )

        marker_position_mean = np.zeros((2, n_shooting + 1))
        for i_node in range(n_shooting + 1):
            marker_position_mean[:, i_node] = self.get_marker_position(q_mean[:, i_node])
            ax.plot(np.array([q_mean[0, i_node], marker_position_mean[0, i_node]]),
                    np.array([0, marker_position_mean[1, i_node]]),
                    "-k",
                    linewidth=1,
                    alpha=0.1 + (i_node/n_shooting) * 0.9
                    )
        ax.plot(np.array([0]), np.array([0]), "o", color="k")
        ax.plot(marker_position_mean[0, :], marker_position_mean[1, :], "-o", color="g", markersize=1, linewidth=2, label="Optimal trajectory")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.16, 0.16)

        ax.legend()
        plt.savefig(fig_save_path)
        plt.show()
