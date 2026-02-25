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
    def __init__(self):
        super().__init__()  # Does nothing

        self.nb_random = 15
        self.n_threads = 7
        self.n_simulations = 30
        self.seed = 0
        self.model = ArmModel(self.nb_random)

        # Noise parameters (from Van Wouwe et al. 2022)
        self.initial_dt = 0.05
        self.final_time = 0.8
        self.min_time = 0.8
        self.max_time = 0.8
        self.n_shooting = int(self.final_time / self.initial_dt)
        self.motor_noise_std = 0.05  # Tau noise
        self.wPq_std = 3e-4  # Hand position noise
        self.wPqdot_std = 2.4e-3  # Hand velocity noise
        self.initial_state_variability = np.array([1e-4, 1e-4, 1e-7, 1e-7, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])

        # Solver options
        self.tol = 1e-6
        self.max_iter = 1000

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
        ubq[0, :] = 180  # Bug in Van Wouwe et al. 2022 code?
        ubq[1, :] = 180  # Bug in Van Wouwe et al. 2022 code?
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
        qz0 = np.zeros((nb_q, nb_collocation_points * n_shooting + 1))
        qz0[0, :] = np.linspace(shoulder_pos_initial, shoulder_pos_final, nb_collocation_points * n_shooting + 1)  # Shoulder
        qz0[1, :] = np.linspace(elbow_pos_initial, elbow_pos_final, nb_collocation_points * n_shooting + 1)  # Elbow
        qdotz0 = np.zeros((nb_q, nb_collocation_points * n_shooting + 1))

        # MuscleActivation
        musaz0 = np.ones((n_muscles, nb_collocation_points * n_shooting + 1)) * 0.01  # ?* 1e-6

        collocation_points_initial_guesses = {
            "q": qz0,
            "qdot": qdotz0,
            "mus_activation": musaz0,
        }

        # MuscleExcitation
        lbmuse = np.ones((n_muscles, n_shooting)) * 1e-6  # 1e-6?
        ubmuse = np.ones((n_muscles, n_shooting))
        muse0 = np.ones((n_muscles, n_shooting)) * 0.01

        # K
        lbk = np.ones((nb_k, n_shooting)) * -10
        ubk = np.ones((nb_k, n_shooting)) * 10
        k0 = np.ones((nb_k, n_shooting)) * 0.001

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
        motor_noise_magnitude = cas.DM(np.array([self.motor_noise_std**2 / self.initial_dt] * self.model.nb_q))
        sensory_noise_magnitude = cas.DM(
            np.array(
                [
                    self.wPq_std**2 / self.initial_dt,
                    self.wPq_std**2 / self.initial_dt,
                    self.wPqdot_std**2 / self.initial_dt,
                    self.wPqdot_std**2 / self.initial_dt,
                ]
            )
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

        # Initial constraint
        g_start, lbg_start, ubg_start = self.null_acceleration(
            discretization_method, dynamics_transcription, variables_vector, noises_vector.get_noise_single(0)
        )
        constraints.add(
            g=g_start,
            lbg=lbg_start,
            ubg=ubg_start,
            g_names=[f"null_acceleration"] * 2,
            node=0,
        )

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
            node=n_shooting + 1,
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
            node=n_shooting + 1,
        )



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

        return

    def get_specific_objectives(
        self,
        model: ModelAbstract,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> cas.MX | cas.SX:

        j: cas.MX | cas.SX = 0

        for i_node in range(self.n_shooting):
            j += (
                self.minimize_stochastic_efforts_and_variations(
                    discretization_method,
                    dynamics_transcription,
                    variables_vector,
                    i_node,
                )
                * self.initial_dt
                / 2
            )

        return j

    # --- helper functions --- #
    def null_acceleration(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        noise_single: cas.MX | cas.SX,
    ) -> tuple[list[cas.MX | cas.SX], list[float], list[float]]:

        xdot = dynamics_transcription.dynamics_func(
            variables_vector.get_states(0),
            variables_vector.get_controls(0),
            cas.DM.zeros(noise_single.shape),
        )
        xdot_mean = discretization_method.get_mean_states(
            variables_vector,
            node,
            squared=True,
        )
        states = variables_vector.get_states_matrix(node)

        exponent = 2 if squared else 1
        states_sq = states**exponent

        states_mean = cas.sum2(states_sq) / variables_vector.nb_random
        
        g = [xdot_mean[self.model.qdot_indices]]
        lbg = [0, 0]
        ubg = [0, 0]
        return g, lbg, ubg

    def mean_start_on_target(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
    ) -> tuple[list[cas.MX | cas.SX], list[float], list[float]]:
        """
        Constraint to impose that the mean trajectory reaches the target at the end of the movement
        """
        ee_pos_mean = discretization_method.get_reference(
            self.model,
            variables_vector.get_states(variables_vector.n_shooting + 1),
            variables_vector.get_controls(variables_vector.n_shooting + 1),
        )[:2]
        g = [ee_pos_mean - HAND_INITIAL_TARGET]
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
            self.model,
            variables_vector.get_states(variables_vector.n_shooting + 1),
            variables_vector.get_controls(variables_vector.n_shooting + 1),
        )[:2]
        g = [ee_pos_mean - HAND_FINAL_TARGET]
        lbg = [0, 0]
        ubg = [0, 0]

        # All hand positions are inside a circle of radius 4 mm around the target
        ee_pos_variability_x, ee_pos_variability_y = discretization_method.get_ee_variance(
            self.model,
            variables_vector.get_states(variables_vector.n_shooting + 1),
            variables_vector.get_controls(variables_vector.n_shooting + 1),
            HAND_FINAL_TARGET,
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
            self.model,
            variables_vector.get_states(variables_vector.n_shooting + 1),
            variables_vector.get_controls(variables_vector.n_shooting + 1),
        )[2:4]
        g = [ee_velo_mean]
        lbg = [0, 0]
        ubg = [0, 0]
        return g, lbg, ubg

    def minimize_stochastic_efforts_and_variations(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variable_vector: VariablesAbstract,
        i_node: int,
    ) -> cas.SX:

        activations_mean = discretization_method.get_mean_states(
            variable_vector,
            i_node,
            squared=True,
        )[4 : 4 + self.model.nb_muscles]
        efforts = cas.sum1(activations_mean)

        activations_variations = discretization_method.get_mus_variance(
            self.model,
            variable_vector.get_states(i_node),
        )

        j = efforts + activations_variations / 2

        return j

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
                marker_position_opt[:, i_node] = self.model.end_effector_position(q_opt[:, i_node])

            ax.plot(marker_position_opt[0, :], marker_position_opt[1, :], "--k", label="Initial guess", linewidth=0.5, alpha=0.3)
            ax.plot(marker_position_opt[0, 0], marker_position_opt[1, 0], "og", label="Optimal initial node")
        else:
            marker_position_opt = np.zeros((2, n_shooting + 1, ocp["ocp_example"].nb_random))
            for i_random in range(ocp["ocp_example"].nb_random):
                for i_node in range(n_shooting + 1):
                    marker_position_opt[:, i_node, i_random] = self.model.end_effector_position(q_opt[:, i_node, i_random])

                if i_random == 0:
                    ax.plot(marker_position_opt[0, 0, i_random], marker_position_opt[1, 0, i_random], "og", label="Optimal initial node")
                else:
                    ax.plot(marker_position_opt[0, 0, i_random], marker_position_opt[1, 0, i_random], "og")
                ax.plot(marker_position_opt[0, :, i_random], marker_position_opt[1, :, i_random], "-", color="g", linewidth=0.5, alpha=0.3)


        marker_position_simulated = np.zeros((2, n_shooting + 1, n_simulations))
        for i_simulation in range(n_simulations):
            for i_node in range(n_shooting + 1):
                marker_position_simulated[:, i_node, i_simulation] = self.model.end_effector_position(q_simulated[:, i_node, i_simulation])

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
            marker_position_mean[:, i_node] = self.model.end_effector_position(q_mean[:, i_node])
        ax.plot(marker_position_mean[0, :], marker_position_mean[1, :], "-o", color="g", markersize=1, linewidth=2, label="Optimal trajectory")
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-0.16, 0.16)

        ax.legend()
        plt.savefig(fig_save_path)
        plt.show()
