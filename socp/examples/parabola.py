"""
This example is taken from Campos et al.2013.
"""

from typing import Any
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

from .example_abstract import ExampleAbstract
from ..constraints import Constraints
from ..models.parabola_model import ParabolaModel
from ..models.model_abstract import ModelAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract
from ..transcriptions.mean_and_covariance import MeanAndCovariance
from ..transcriptions.noises_abstract import NoisesAbstract
from ..transcriptions.transcription_abstract import TranscriptionAbstract
from ..transcriptions.variables_abstract import VariablesAbstract
from ..transcriptions.variational import Variational
from ..transcriptions.variational_polynomial import VariationalPolynomial


class Parabola(ExampleAbstract):
    def __init__(self) -> None:
        super().__init__()  # Does nothing

        self.nb_random = 1
        self.n_threads = 7
        self.n_simulations = 100
        self.seed = 0
        self.model = ParabolaModel(self.nb_random)

        self.final_time = 10.0
        self.min_time = 10.0
        self.max_time = 10.0
        self.n_shooting = 100
        self.initial_state_variability = np.array([0.01, 0.01])

        # Solver options
        self.tol = 1e-6
        self.max_iter = 10000

    @property
    def name(self) -> str:
        return "Parabola"

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
        lbq = np.ones((nb_q, n_shooting + 1)) * -50
        ubq = np.ones((nb_q, n_shooting + 1)) * 10
        q0 = np.zeros((nb_q, n_shooting + 1))
        # Start with q = 0
        lbq[:, 0] = 0
        ubq[:, 0] = 0
        lbq[:, -1] = 0.4933
        ubq[:, -1] = 0.4933

        # Qdot
        lbqdot = np.ones((nb_q, n_shooting + 1)) * -10
        ubqdot = np.ones((nb_q, n_shooting + 1)) * 10
        qdot0 = np.zeros((nb_q, n_shooting + 1))
        # Start with qdot = 0
        lbqdot[:, 0] = 0
        ubqdot[:, 0] = 0
        # lbqdot[:, -1] = 42.6658
        # ubqdot[:, -1] = 42.6658

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
        lbu = np.ones((nb_q, n_shooting + 1)) * -10
        ubu = np.ones((nb_q, n_shooting + 1)) * 50
        u0 = np.zeros((nb_q, n_shooting + 1))

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
        motor_noise_magnitude = np.array([0])
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
        return

    def get_specific_objectives(
        self,
        model: ModelAbstract,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> cas.SX:

        j = 0
        dt = self.final_time / self.n_shooting
        for i_node in range(self.n_shooting):
            # TODO: The following line is a linear approximation that should be changed for a proper quadrature
            qdot = (variables_vector.get_state("q", i_node+1) - variables_vector.get_state("q", i_node)) / dt
            u = variables_vector.get_controls(i_node)
            j += cas.sum1(qdot ** 2) + cas.sum1(u ** 2) * dt

        return j

    def specific_plot_results(
        self,
        ocp: dict[str, Any],
        data_saved: dict[str, Any],
        fig_save_path: str,
    ):
        """
        This function plots the actual solution against the expected solution.
        """
        n_shooting = ocp["n_shooting"]
        states_opt_mean = data_saved["states_opt_mean"]

        q_mean = states_opt_mean[ocp["ocp_example"].model.q_indices, :]
        u_opt = data_saved["controls_opt_array"][ocp["ocp_example"].model.q_indices, :]
        q_opt = data_saved["states_opt_array"][ocp["ocp_example"].model.q_indices, :]
        q_simulated = data_saved["x_simulated"][: ocp["ocp_example"].model.nb_q, :, :]
        n_simulations = q_simulated.shape[2]
        # covariance_simulated = data_saved["covariance_simulated"]
        # cov_opt = data_saved["cov_opt_array"]
        T = ocp["ocp_example"].final_time
        time_vector = np.linspace(0, T, n_shooting + 1)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(time_vector, (np.cosh(time_vector) - 1) / np.cosh(T), "k", label="Expected states")
        for i_node in range(n_shooting):
            if i_node == 0:
                ax[0].plot(
                    np.ones((n_simulations, )) * time_vector[i_node], q_simulated[0, i_node, :], ".r", markersize=1, label="Noisy integration"
                )
            else:
                ax[0].plot(np.ones((n_simulations, )) * time_vector[i_node], q_simulated[0, i_node, :], ".r", markersize=1)

        ax[0].plot(time_vector, q_mean[0, :], "-o", color="g", markersize=1, linewidth=2, label="Optimal trajectory")

        ax[1].plot(time_vector, (np.cosh(time_vector) / np.cosh(T)) - 1, "k", label="Expected control")
        ax[1].plot(time_vector, u_opt[0, :], "-g", label="Optimal controls")

        ax[0].legend()
        plt.savefig(fig_save_path)
        plt.show()
