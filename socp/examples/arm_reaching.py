"""
This example implements a stochastic optimal control problem for an arm reaching task using muscle-driven dynamics.
The goal is to move the arm from an initial target to a final target while minimizing effort and variability, considering motor and sensory noise.
This example was taken from Van Wouwe et al. 2022.
"""

import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
from typing import Any

from .example_abstract import ExampleAbstract
from ..constraints import Constraints
from ..models.arm_model import ArmModel
from ..models.model_abstract import ModelAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract
from ..transcriptions.mean_and_covariance import MeanAndCovariance
from ..transcriptions.noises_abstract import NoisesAbstract
from ..transcriptions.transcription_abstract import TranscriptionAbstract
from ..transcriptions.variables_abstract import VariablesAbstract
from ..transcriptions.variational import Variational
from ..transcriptions.variational_polynomial import VariationalPolynomial

# Taken from Van Wouwe et al. 2022
HAND_INITIAL_TARGET = np.array([0.0, 0.2742])
HAND_FINAL_TARGET = np.array([0.0, 0.527332023564034])


class ArmReaching(ExampleAbstract):
    def __init__(self, nb_random: int = 10):
        super().__init__(nb_random=nb_random)

        self.n_threads = 7
        self.n_simulations = 30
        self.seed = 0
        self.model = ArmModel(self.nb_random)
        self.initial_states_to_impose = ["q", "qdot", "mus_activation"]

        # Noise parameters (from Van Wouwe et al. 2022)
        self.initial_dt = 0.01  # The real one !!!!!
        # self.initial_dt = 0.05
        self.final_time = 0.8
        self.min_time = 0.8
        self.max_time = 0.8
        self.n_shooting = int(self.final_time / self.initial_dt)
        self.motor_noise_std = 0.05  # Tau noise
        self.wPq_std = 3e-4  # Hand position noise
        self.wPqdot_std = 2.4e-3  # Hand velocity noise
        self.initial_state_variability = np.array([1e-4, 1e-4, 1e-7, 1e-7, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        self.initial_covariance = np.diag((self.initial_state_variability**2).tolist())

        # Solver options
        self.tol = 1e-6
        self.max_iter = 10000

    @property
    def name(self) -> str:
        return "ArmReaching"

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

        n_muscles = self.model.nb_muscles
        nb_q = self.model.nb_q
        nb_k = self.model.nb_k

        # Q
        lbq = np.zeros((nb_q, n_shooting + 1))
        ubq = np.zeros((nb_q, n_shooting + 1))
        # ubq[0, :] = np.pi / 2
        # ubq[1, :] = 7 / 8 * np.pi
        ubq[0, :] = np.pi  # 180  # Bug in Van Wouwe et al. 2022 code?
        ubq[1, :] = np.pi  # 180  # Bug in Van Wouwe et al. 2022 code?
        q0 = np.zeros((nb_q, n_shooting + 1))
        q0[0, :] = np.linspace(shoulder_pos_initial, shoulder_pos_final, n_shooting + 1)  # Shoulder
        q0[1, :] = np.linspace(elbow_pos_initial, elbow_pos_final, n_shooting + 1)  # Elbow

        # Qdot
        lbqdot = np.zeros((nb_q, n_shooting + 1))
        lbqdot[:, 1:] = -10 * np.pi
        ubqdot = np.zeros((nb_q, n_shooting + 1))
        ubqdot[:, 1:] = 10 * np.pi
        qdot0 = np.zeros((nb_q, n_shooting + 1))

        # MuscleActivation
        lbmusa = np.ones((n_muscles, n_shooting + 1)) * 1e-6  # * 0.01  # 1e-6?
        ubmusa = np.ones((n_muscles, n_shooting + 1))
        musa0 = np.ones((n_muscles, n_shooting + 1)) * 0.01  # ?* 1e-6
        # musa0[:, 0] = 0  # Is zero in Van Wouwe et al. 2022, but this is dangerous

        states_lower_bounds = {
            "q": lbq,
            "qdot": lbqdot,
            "mus_activation": lbmusa,
        }
        states_upper_bounds = {
            "q": ubq,
            "qdot": ubqdot,
            "mus_activation": ubmusa,
        }
        states_initial_guesses = {
            "q": q0,
            "qdot": qdot0,
            "mus_activation": musa0,
        }

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

        # MuscleActivation
        musaz0 = np.ones((n_muscles, nb_collocation_points, n_shooting + 1)) * 0.01  # ?* 1e-6

        collocation_points_initial_guesses = {
            "q": qz0,
            "qdot": qdotz0,
            "mus_activation": musaz0,
        }

        # MuscleExcitation
        lbmuse = np.ones((n_muscles, n_shooting + 1)) * 1e-6  # 1e-6?
        ubmuse = np.ones((n_muscles, n_shooting + 1))
        muse0 = np.ones((n_muscles, n_shooting + 1)) * 0.01

        # K
        lbk = np.ones((nb_k, n_shooting + 1)) * -10
        ubk = np.ones((nb_k, n_shooting + 1)) * 10
        k0 = np.ones((nb_k, n_shooting + 1)) * 0.001

        controls_lower_bounds = {
            "mus_excitation": lbmuse,
            "k": lbk,
        }
        controls_upper_bounds = {
            "mus_excitation": ubmuse,
            "k": ubk,
        }
        controls_initial_guesses = {
            "mus_excitation": muse0,
            "k": k0,
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
        motor_noise_magnitude = np.array([self.motor_noise_std**2 / self.initial_dt] * self.model.nb_q)
        sensory_noise_magnitude = np.array(
                [
                    self.wPq_std**2 / self.initial_dt,
                    self.wPq_std**2 / self.initial_dt,
                    self.wPqdot_std**2 / self.initial_dt,
                    self.wPqdot_std**2 / self.initial_dt,
                ]
            )
        return motor_noise_magnitude, sensory_noise_magnitude

    def set_specific_constraints(
        self,
        model: ModelAbstract,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
        constraints: Constraints,
    ) -> None:

        n_shooting = variables_vector.n_shooting

        # Terminal constraint
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

        g_target, lbg_target, ubg_target = self.mean_end_effector_velocity(
            discretization_method,
            dynamics_transcription,
            variables_vector,
        )
        constraints.add(
            g=g_target,
            lbg=lbg_target,
            ubg=ubg_target,
            g_names=[f"mean_end_effector_velocity"] * 2,
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
    ) -> cas.MX | cas.SX:

        dt = variables_vector.get_time() / self.n_shooting

        # Declare useful variables
        q = variables_vector.get_state("q", 0)
        qdot = variables_vector.get_state("qdot", 0)
        ref = discretization_method.get_reference(
            ocp_example=self, x=variables_vector.get_states(0), u=variables_vector.get_controls(0)
        )
        muscle_activations = variables_vector.get_state("mus_activation", 0)
        muscle_excitation = variables_vector.get_control("mus_excitation", 0)

        if discretization_method.name != "Deterministic":
            sensory_noise = noises_vector.get_sensory_noise(0)
            k = variables_vector.get_control("k", 0)
            k_matrix = self.model.reshape_vector_to_matrix(k, self.model.matrix_shape_k)

        # Minimize expected feedbacks
        j: cas.MX | cas.SX = 0
        if discretization_method.name == "Deterministic":
            # Deterministic -> skip
            for i_node in range(self.n_shooting + 1):
                muscle_activations = variables_vector.get_state("mus_activation", i_node)
                muscle_excitation = variables_vector.get_control("mus_excitation", i_node)
                j += (cas.sum1(muscle_activations ** 2) + cas.sum1(muscle_excitation ** 2)) / 2
        else:
            if discretization_method.name == "MeanAndCovariance":
                muscle_fb = k_matrix @ (self.model.sensory_output(q, qdot, sensory_noise) - ref + sensory_noise)
                jacobian_fb_x = cas.jacobian(muscle_fb, variables_vector.get_states(0))
                one_sensory_noise = noises_vector.get_sensory_noise(0)
            else:
                q_this_time = variables_vector.get_specific_state("q", 0, 0)
                qdot_this_time = variables_vector.get_specific_state("qdot", 0, 0)
                a_this_time = variables_vector.get_specific_state("mus_activation", 0, 0)
                sensory_noise_this_time = noises_vector.get_sensory_noise(0)[: self.model.nb_references]
                muscle_fb_this_time = k_matrix @ (
                    self.model.sensory_output(q_this_time, qdot_this_time, sensory_noise_this_time) - ref + sensory_noise_this_time
                )
                jacobian_fb_x = cas.jacobian(muscle_fb_this_time, cas.vertcat(q_this_time, qdot_this_time, a_this_time))
                one_sensory_noise = noises_vector.get_one_sensory_noise(0, 0)

            cov_matrix = discretization_method.get_covariance(variables_vector, 0, is_matrix=True)
            sigma_ww = cas.diag(one_sensory_noise)

            expected_feedback_variability = cas.trace(jacobian_fb_x @ cov_matrix @ jacobian_fb_x.T) + cas.trace(
                k_matrix @ sigma_ww @ k_matrix.T
            )

            # Minimize muscle activation variability
            activations_variations = cas.trace(
                cov_matrix[self.model.mus_activation_indices, self.model.mus_activation_indices]
            )

            j_sym = (
                expected_feedback_variability
                + (activations_variations + cas.sum1(muscle_activations**2) + cas.sum1(muscle_excitation**2)) / 2
            )

            sym_variables = [
                variables_vector.get_states(0),
                variables_vector.get_controls(0),
                one_sensory_noise,
            ]
            if discretization_method.name == "MeanAndCovariance":
                sym_variables += [variables_vector.get_cov(0)]

            j_func = cas.Function("j_func", sym_variables, [j_sym])

            for i_node in range(self.n_shooting + 1):
                _, sensory_noise_magnitude = self.get_noises_magnitude()
                variables_this_time = [
                    variables_vector.get_states(i_node),
                    variables_vector.get_controls(i_node),
                    sensory_noise_magnitude,
                ]
                if discretization_method.name == "MeanAndCovariance":
                    variables_this_time += [variables_vector.get_cov(i_node)]

                j += j_func(*variables_this_time)

        # Encourage to reach the target at each trial
        if discretization_method.name == "Deterministic":
            q_last = variables_vector.get_state("q", self.n_shooting)
            qdot_last = variables_vector.get_state("qdot", self.n_shooting)
            ee_pos_velo = model.sensory_output(q_last, qdot_last, cas.DM.zeros(model.nb_references))
            j += 1e3 * (
                    (ee_pos_velo[0] - HAND_FINAL_TARGET[0]) ** 2 +
                    (ee_pos_velo[1] - HAND_FINAL_TARGET[1]) ** 2 +
                    (ee_pos_velo[2] - 0) ** 2 +
                    (ee_pos_velo[3] - 0) ** 2
            )
        elif discretization_method.name == "NoiseDiscretization":
            for i_random in range(model.nb_random):
                q_this_time = variables_vector.get_specific_state("q", node=self.n_shooting, random=i_random)
                qdot_this_time = variables_vector.get_specific_state("qdot", node=self.n_shooting, random=i_random)
                ee_pos_velo = model.sensory_output(q_this_time, qdot_this_time, cas.DM.zeros(model.nb_references))
                j += 1e3/model.nb_random * (
                        (ee_pos_velo[0] - HAND_FINAL_TARGET[0]) ** 2 +
                        (ee_pos_velo[1] - HAND_FINAL_TARGET[1]) ** 2 +
                        (ee_pos_velo[2] - 0) ** 2 +
                        (ee_pos_velo[3] - 0) ** 2
                )
        elif discretization_method.name == "MeanAndCovariance":
            q_this_time = variables_vector.get_state("q", node=self.n_shooting)
            qdot_this_time = variables_vector.get_state("qdot", node=self.n_shooting)
            ee_pos_velo = model.sensory_output(q_this_time, qdot_this_time, cas.DM.zeros(model.nb_references))
            j += 1e3 / 2 * (
                    (ee_pos_velo[0] - HAND_FINAL_TARGET[0]) ** 2 +
                    (ee_pos_velo[1] - HAND_FINAL_TARGET[1]) ** 2 +
                    (ee_pos_velo[2] - 0) ** 2 +
                    (ee_pos_velo[3] - 0) ** 2
            )
            cov_matrix = discretization_method.get_covariance(variables_vector, 0, is_matrix=True)
            jacobian_fb_x = cas.jacobian(ee_pos_velo, cas.vertcat(q_this_time, qdot_this_time))
            ee_variability = jacobian_fb_x @ cov_matrix[:4, :4] @ jacobian_fb_x.T
            j += 1e3 / 2 * cas.sum2(cas.sum1(ee_variability ** 2))
        else:
            raise RuntimeError(f"Discretization method {discretization_method.name} not implemented for this objective")

        return 1e3 * j * dt

    # --- helper functions --- #
    def null_acceleration(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        noise_single: cas.MX | cas.SX,
        node: int,
    ) -> tuple[list[cas.MX | cas.SX], list[float], list[float]]:

        xdot = dynamics_transcription.dynamics_func(
            variables_vector.get_states(node),
            variables_vector.get_controls(node),
            cas.DM.zeros(noise_single.shape),
        )

        xdot_mean = cas.sum2(xdot) / variables_vector.nb_random

        g = xdot_mean[self.model.qdot_indices]
        lbg = [0, 0]
        ubg = [0, 0]
        return g, lbg, ubg

    def mean_start_on_target(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
    ) -> tuple[cas.MX | cas.SX, list[float], list[float]]:
        """
        Constraint to impose that the mean trajectory reaches the target at the end of the movement
        """
        ee_pos_mean = discretization_method.get_reference(
            ocp_example=self,
            x=variables_vector.get_states(0),
            u=variables_vector.get_controls(0),
        )[:2]
        g = ee_pos_mean - HAND_INITIAL_TARGET
        lbg = [0, 0]
        ubg = [0, 0]
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
        ee_pos_mean = discretization_method.get_reference(
            ocp_example=self,
            x=variables_vector.get_states(variables_vector.n_shooting),
            u=variables_vector.get_controls(variables_vector.n_shooting),
        )[:2]
        g = [ee_pos_mean[0] - HAND_FINAL_TARGET[0], ee_pos_mean[1] - HAND_FINAL_TARGET[1]]
        lbg = [0, 0]
        ubg = [0, 0]

        # All hand positions are inside a circle of radius 4 mm around the target
        if discretization_method.name != "Deterministic":
            ee_pos_variability_x, ee_pos_variability_y = discretization_method.get_ee_variance(
                self.model,
                variables_vector.get_states(variables_vector.n_shooting),
                discretization_method.get_covariance(variables_vector, variables_vector.n_shooting, is_matrix=False),
                variables_vector.get_controls(variables_vector.n_shooting),
                ee_pos_mean,
            )
            radius = 0.004
            g += [ee_pos_variability_x - radius**2, ee_pos_variability_y - radius**2]
            lbg += [-cas.inf, -cas.inf]
            ubg += [0, 0]

        return g, lbg, ubg

    def mean_end_effector_velocity(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
    ) -> tuple[list[cas.SX], list[float], list[float]]:
        """
        Constraint to impose that the mean hand velocity is null at the end of the movement
        """
        ee_velo_mean = discretization_method.get_reference(
            ocp_example=self,
            x=variables_vector.get_states(variables_vector.n_shooting),
            u=variables_vector.get_controls(variables_vector.n_shooting),
        )[2:4]
        g = ee_velo_mean
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
        q_init = data_saved["states_init_array"][ocp["ocp_example"].model.q_indices, :]
        q_opt = data_saved["states_opt_array"][ocp["ocp_example"].model.q_indices, :]
        q_simulated = data_saved["x_simulated"][: ocp["ocp_example"].model.nb_q, :, :]
        n_simulations = q_simulated.shape[2]
        covariance_simulated = data_saved["covariance_simulated"]
        cov_opt = data_saved["cov_opt_array"]

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        if ocp["discretization_method"].name in ["Deterministic", "MeanAndCovariance"]:
            marker_position_opt = np.zeros((2, n_shooting + 1))
            for i_node in range(n_shooting + 1):
                marker_position_opt[:, i_node] = np.array(self.model.end_effector_position(q_opt[:, i_node])).reshape(2, )

            ax.plot(
                marker_position_opt[0, :],
                marker_position_opt[1, :],
                "--k",
                label="Initial guess",
                linewidth=0.5,
                alpha=0.3,
            )
            ax.plot(marker_position_opt[0, 0], marker_position_opt[1, 0], "og", label="Optimal initial node")
        else:
            marker_position_opt = np.zeros((2, n_shooting + 1, ocp["ocp_example"].nb_random))
            for i_random in range(ocp["ocp_example"].nb_random):
                for i_node in range(n_shooting + 1):
                    marker_position_opt[:, i_node, i_random] = np.array(self.model.end_effector_position(
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

        marker_position_simulated = np.zeros((2, n_shooting + 1, n_simulations))
        for i_simulation in range(n_simulations):
            for i_node in range(n_shooting + 1):
                marker_position_simulated[:, i_node, i_simulation] = np.array(self.model.end_effector_position(
                    q_simulated[:, i_node, i_simulation]
                )).reshape(2, )

        q_simulated_mean = np.mean(q_simulated, axis=2)
        for i_node in range(n_shooting):
            if i_node == 0:
                # self.draw_cov_ellipse(
                #     cov=cov_opt[:2, :2, i_node], pos=q_mean[:, i_node], ax=ax[0], color="b", label="Cov optimal"
                # )
                ax.plot(
                    marker_position_simulated[0, i_node, :],
                    marker_position_simulated[1, i_node, :],
                    ".r",
                    markersize=1,
                    label="Noisy integration",
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
                ax.plot(
                    marker_position_simulated[0, i_node, :], marker_position_simulated[1, i_node, :], ".r", markersize=1
                )
                # self.draw_cov_ellipse(
                #     cov=covariance_simulated[:2, :2, i_node],
                #     pos=q_simulated_mean[:, i_node],
                #     ax=ax[0],
                #     color="r",
                # )

        marker_position_mean = np.zeros((2, n_shooting + 1))
        for i_node in range(n_shooting + 1):
            marker_position_mean[:, i_node] = np.array(self.model.end_effector_position(q_mean[:, i_node])).reshape(2, )
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
