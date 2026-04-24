from typing import Any
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

from .example_abstract import ExampleAbstract
from ..constraints import Constraints
from ..models.vertebrate_model import VertebrateModel
from ..models.model_abstract import ModelAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract
from ..transcriptions.mean_and_covariance import MeanAndCovariance
from ..transcriptions.noises_abstract import NoisesAbstract
from ..transcriptions.transcription_abstract import TranscriptionAbstract
from ..transcriptions.variables_abstract import VariablesAbstract
from ..transcriptions.variational import Variational
from ..transcriptions.variational_polynomial import VariationalPolynomial


class Vertebrate(ExampleAbstract):
    def __init__(self, nb_random: int = 10) -> None:
        super().__init__(nb_random=nb_random)

        self.n_threads = 7
        self.n_simulations = 100
        self.seed = 0
        self.model = VertebrateModel(self.nb_random)
        self.initial_states_to_impose = ["q", "qdot"]

        self.final_time = 1.0
        self.min_time = 1.0
        self.max_time = 1.0
        self.n_shooting = 40
        self.initial_state_variability = np.array([0.01] * self.model.nb_q * 2)
        self.initial_covariance = np.diag((self.initial_state_variability**2).tolist())

        # Solver options
        self.tol = 1e-8
        self.max_iter = 1000

    @property
    def name(self) -> str:
        return "Vertebrate"

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
        lbq = np.ones((nb_q, n_shooting + 1)) * -0.1
        lbq[0, :] = -np.pi / 2  # Allow to swing the first dof
        lbq[:, 0] = -0.1  # Start at zero
        lbq[:, -1] = -0.1  # End aligned with a rotation of pi
        lbq[0, -1] = np.pi - 0.1  # End aligned with a rotation of pi
        ubq = np.ones((nb_q, n_shooting + 1)) * np.pi + 0.1
        ubq[:, 0] = 0.1  # Start at zero
        ubq[:, -1] = 0.1  # End aligned with a rotation of pi
        ubq[0, -1] = np.pi + 0.1  # End aligned with a rotation of pi

        # Zero initial guess
        q0 = np.zeros((nb_q, n_shooting + 1))
        q0[0, :] = np.linspace(0, np.pi, n_shooting + 1)  # Initial guess of a linear movement from 0 to pi

        # Qdot
        lbqdot = np.ones((nb_q, n_shooting + 1)) * -10 * np.pi
        lbqdot[:, 0] = -1  # Start with zero velocity
        lbqdot[:, -1] = -1  # End with zero velocity
        ubqdot = np.ones((nb_q, n_shooting + 1)) * 10 * np.pi
        ubqdot[:, 0] = 1  # Start with zero velocity
        ubqdot[:, -1] = 1  # End with zero velocity

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
            qz0[0, :, :] = np.linspace(0, np.pi, n_shooting + 1)  # Initial guess of a linear movement from 0 to pi
            qdotz0 = np.zeros((nb_q, nb_collocation_points, n_shooting + 1))

            collocation_points_initial_guesses = {
                "q": qz0,
                "qdot": qdotz0,
            }

        # u
        lbu = np.ones((nb_q, n_shooting + 1)) * -100
        ubu = np.ones((nb_q, n_shooting + 1)) * 100
        u0 = np.zeros((nb_q, n_shooting + 1))

        controls_lower_bounds = {
            "tau": lbu,
        }
        controls_upper_bounds = {
            "tau": ubu,
        }
        controls_initial_guesses = {
            "tau": u0,
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
        motor_noise_magnitude = np.array([0.01] * self.model.nb_q)
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

        if isinstance(dynamics_transcription, (Variational, VariationalPolynomial)):
            nb_states = model.nb_q
        else:
            nb_states = variables_vector.nb_states

        # Final mean states are imposed
        x_final = np.array([np.pi, 0, 0, 0, 0, 0, 0, 0, 0, 0])[:nb_states]
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

        return 100 * j_variability + j_controls

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
        pass
