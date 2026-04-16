from typing import Any
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt

from .example_abstract import ExampleAbstract
from ..constraints import Constraints
from ..models.vertebrate_arm_model import VertebrateArmModel
from ..models.model_abstract import ModelAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract
from ..transcriptions.noises_abstract import NoisesAbstract
from ..transcriptions.transcription_abstract import TranscriptionAbstract
from ..transcriptions.variables_abstract import VariablesAbstract
from ..transcriptions.variational import Variational
from ..transcriptions.variational_polynomial import VariationalPolynomial


# Taken from Van Wouwe et al. 2022
# HAND_INITIAL_TARGET = np.array([0.0, 0.2])
# HAND_FINAL_TARGET = np.array([0.2, 0.21])
HAND_INITIAL_TARGET = np.array([0.0, 0.2742])
HAND_FINAL_TARGET = np.array([0.0, 0.527332023564034])


class VertebrateArm(ExampleAbstract):
    def __init__(self, nb_random: int = 10, seed: int = 0) -> None:
        super().__init__(nb_random=nb_random)

        self.n_threads = 7
        self.n_simulations = 100
        self.seed = seed
        self.model = VertebrateArmModel(self.nb_random)

        self.final_time = 1.0
        self.min_time = 1.0
        self.max_time = 1.0
        self.n_shooting = 40
        self.initial_state_variability = np.array([0.1] * self.model.nb_q * 2)
        self.initial_covariance = np.diag((self.initial_state_variability**2).tolist())

        # Solver options
        self.tol = 1e-8
        self.max_iter = 1000

    @property
    def name(self) -> str:
        return "VertebrateArm"

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
        # Find a good initial guess based on the start and end targets the hand must reach
        initial_pose = self.model.inverse_kinematics_target(HAND_INITIAL_TARGET)
        shoulder_pos_initial, elbow_pos_initial = initial_pose[0], initial_pose[1]
        final_pose = self.model.inverse_kinematics_target(HAND_FINAL_TARGET)
        shoulder_pos_final, elbow_pos_final = final_pose[0], final_pose[1]

        nb_q = self.model.nb_q

        # Q
        lbq = np.zeros((nb_q, n_shooting + 1))
        ubq = np.zeros((nb_q, n_shooting + 1))
        # ubq[0, :] = np.pi / 2
        # ubq[1, :] = 7 / 8 * np.pi
        ubq[0, :] = np.pi
        ubq[1, :] = np.pi
        q0 = np.zeros((nb_q, n_shooting + 1))
        q0[0, :] = np.linspace(shoulder_pos_initial, shoulder_pos_final, n_shooting + 1)  # Shoulder
        q0[1, :] = np.linspace(elbow_pos_initial, elbow_pos_final, n_shooting + 1)  # Elbow

        # Qdot
        lbqdot = np.zeros((nb_q, n_shooting + 1))
        lbqdot[:, 1:] = -10 * np.pi
        ubqdot = np.zeros((nb_q, n_shooting + 1))
        ubqdot[:, 1:] = 10 * np.pi
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
            # Q
            qz0 = np.zeros((nb_q, nb_collocation_points, n_shooting + 1))
            qz0[
                0,
                :,
            ] = np.linspace(
                shoulder_pos_initial, shoulder_pos_final, n_shooting + 1
            )  # Shoulder
            qz0[
                1,
                :,
            ] = np.linspace(
                elbow_pos_initial, elbow_pos_final, n_shooting + 1
            )  # Elbow
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

        # Initial covariance and mean position are imposed by bounds

        # Terminal constraint
        n_shooting = variables_vector.n_shooting
        g_target, lbg_target, ubg_target = self.mean_reach_target(
            discretization_method,
            dynamics_transcription,
            variables_vector,
        )
        constraints.add(
            g=g_target,
            lbg=lbg_target,
            ubg=ubg_target,
            g_names=[f"mean_reach_target"] * len(lbg_target),
            node=n_shooting,
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
        j_variability: cas.SX | cas.MX = 0
        if discretization_method.name != "Deterministic":
            cov_matrix = discretization_method.get_covariance(variables_vector, self.n_shooting, is_matrix=True)
            j_variability = cas.sum1(cas.sum2(cov_matrix.T @ cov_matrix))

        return 100 * j_variability + j_controls

    # --- helper functions --- #
    def mean_start_on_target(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
    ) -> tuple[cas.MX | cas.SX, list[float], list[float]]:
        """
        Constraint to impose that the mean trajectory reaches the target at the end of the movement
        """
        ee_pos_mean = discretization_method.get_mean_marker(
            ocp_example=self,
            x=variables_vector.get_states(0),
            u=variables_vector.get_controls(0),
        )
        g = ee_pos_mean - HAND_INITIAL_TARGET
        lbg = [-1e-3, -1e-3]
        ubg = [1e-3, 1e-3]
        return g, lbg, ubg

    def mean_reach_target(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
    ) -> tuple[list[cas.MX | cas.SX], list[float], list[float]]:
        """
        Constraint to impose that the mean trajectory reaches the target at the end of the movement
        """
        # The mean end-effector position is on the target
        ee_pos_mean = discretization_method.get_mean_marker(
            ocp_example=self,
            x=variables_vector.get_states(variables_vector.n_shooting),
            u=variables_vector.get_controls(variables_vector.n_shooting),
        )
        g = [ee_pos_mean[0] - HAND_FINAL_TARGET[0], ee_pos_mean[1] - HAND_FINAL_TARGET[1]]
        lbg = [0, 0]
        ubg = [0, 0]

        return g, lbg, ubg

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
        q_opt = data_saved["states_opt_array"][ocp["ocp_example"].model.q_indices, :]
        q_simulated = data_saved["x_simulated"][: ocp["ocp_example"].model.nb_q, :, :]
        n_simulations = q_simulated.shape[2]

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        if ocp["discretization_method"].name == "MeanAndCovariance" or ocp["discretization_method"].name == "Deterministic":
            marker_position_opt = np.zeros((2, n_shooting + 1))
            for i_node in range(n_shooting + 1):
                marker_position_opt[:, i_node] = np.array(self.model.marker_position(q_opt[:, i_node])).reshape(2, )

            ax.plot(
                marker_position_opt[0, :],
                marker_position_opt[1, :],
                "--k",
                label="Initial guess",
                linewidth=0.5,
                alpha=0.3,
            )
            ax.plot(marker_position_opt[0, 0], marker_position_opt[1, 0], "og", label="Optimal initial node")
        elif ocp["discretization_method"].name == "NoiseDiscretization":
            marker_position_opt = np.zeros((2, n_shooting + 1, ocp["ocp_example"].nb_random))
            for i_random in range(ocp["ocp_example"].nb_random):
                for i_node in range(n_shooting + 1):
                    marker_position_opt[:, i_node, i_random] = np.array(self.model.marker_position(
                        q_opt[:, i_node, i_random]
                    )).reshape(2, )

                if i_random == 0:
                    ax.plot(
                        marker_position_opt[0, 0, i_random],
                        marker_position_opt[1, 0, i_random],
                        "og",
                        label="Optimal initial node",
                    )
                else:
                    ax.plot(marker_position_opt[0, 0, i_random], marker_position_opt[1, 0, i_random], "og")
                ax.plot(
                    marker_position_opt[0, :, i_random],
                    marker_position_opt[1, :, i_random],
                    "-",
                    color="g",
                    linewidth=0.5,
                    alpha=0.3,
                )
        else:
            raise RuntimeError(f"Unknown discretization method: {ocp['discretization_method'].name}")

        marker_position_simulated = np.zeros((2, n_shooting + 1, n_simulations))
        for i_simulation in range(n_simulations):
            for i_node in range(n_shooting + 1):
                marker_position_simulated[:, i_node, i_simulation] = np.array(self.model.marker_position(
                    q_simulated[:, i_node, i_simulation]
                )).reshape(2, )

        for i_node in range(n_shooting):
            if i_node == 0:
                ax.plot(
                    marker_position_simulated[0, i_node, :],
                    marker_position_simulated[1, i_node, :],
                    ".r",
                    markersize=1,
                    label="Noisy integration",
                )
            else:
                ax.plot(
                    marker_position_simulated[0, i_node, :], marker_position_simulated[1, i_node, :], ".r", markersize=1
                )

        marker_position_mean = np.zeros((2, n_shooting + 1))
        for i_node in range(n_shooting + 1):
            marker_position_mean[:, i_node] = np.array(self.model.marker_position(q_mean[:, i_node])).reshape(2, )
        ax.plot(
            marker_position_mean[0, :],
            marker_position_mean[1, :],
            "-o",
            color="g",
            markersize=1,
            linewidth=2,
            label="Optimal trajectory",
        )
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-0.16, 0.16)
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")

        ax.legend()
        plt.savefig(fig_save_path)
        # plt.show()
        plt.close()
