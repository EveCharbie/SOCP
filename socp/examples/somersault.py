import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
from typing import Any

from .example_abstract import ExampleAbstract
from ..constraints import Constraints
from ..models.somersault_model import SomersaultModel
from ..models.model_abstract import ModelAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract
from ..transcriptions.mean_and_covariance import MeanAndCovariance
from ..transcriptions.noises_abstract import NoisesAbstract
from ..transcriptions.transcription_abstract import TranscriptionAbstract
from ..transcriptions.variables_abstract import VariablesAbstract
from ..transcriptions.variational import Variational
from ..transcriptions.variational_polynomial import VariationalPolynomial


class Somersault(ExampleAbstract):
    def __init__(self):
        super().__init__()  # Does nothing

        self.nb_random = 10
        self.n_threads = 7
        self.n_simulations = 30
        self.seed = 0
        self.model = SomersaultModel(self.nb_random)

        # Noise parameters (from Charbonneau et al. 2026)
        self.final_time = 0.4
        self.min_time = 0.1
        self.max_time = 1

        self.initial_dt = 0.05
        final_time = 0.8
        self.n_shooting = int(final_time / self.initial_dt)

        self.motor_noise_std = 0.05 * 10
        self.wPq_std = 0.001 * 5
        self.wPqdot_std = 0.003 * 5
        self.initial_state_variability = np.array([1e-4] * 7 + [1e-7] * 7)
        self.initial_covariance = np.diag((self.initial_state_variability**2).tolist())

        # Solver options
        self.tol = 1e-6
        self.max_iter = 10000

    @property
    def name(self) -> str:
        return "Somersault"

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
        pose_at_first_node = np.array(
            [-0.0346, 0.1207, 0.2255, 0.0, 3.1, -0.1787, 0.0]
        )  # Initial position approx from bioviz
        pose_at_last_node = np.array(
            [-0.0346, 0.1207, 5.8292, -0.1801, 0.5377, 0.8506, -0.6856]
        )  # Final position approx from bioviz

        nb_q = self.model.nb_q
        nb_root = 3
        nb_k = self.model.nb_k

        # Q
        lbq = np.zeros((nb_q, n_shooting + 1))
        ubq = np.zeros((nb_q, n_shooting + 1))
        for i_node in range(n_shooting + 1):
            lbq[:, i_node] = [-2.5, -1, -3, -70 * np.pi / 180, -0.7, -0.4, -2.3]
            ubq[:, i_node] = [2.5, 3, 9, np.pi / 8, 3.1, 2.6, -0.02]

        q0 = np.zeros((nb_q, n_shooting + 1))
        for i_dof in range(nb_q):
            q0[i_dof, :] = np.linspace(pose_at_first_node[i_dof], pose_at_last_node[i_dof], n_shooting + 1)

        # Qdot
        lbqdot = np.ones((nb_q, n_shooting + 1)) * -100
        ubqdot = np.ones((nb_q, n_shooting + 1)) * 100
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

        # Q
        qz0 = np.zeros((nb_q, nb_collocation_points, n_shooting + 1))
        for i_dof in range(nb_q):
            qz0[i_dof, :, :] = np.linspace(pose_at_first_node[i_dof], pose_at_last_node[i_dof], n_shooting + 1)
        qdotz0 = np.zeros((nb_q, nb_collocation_points, n_shooting + 1))

        collocation_points_initial_guesses = {
            "q": qz0,
            "qdot": qdotz0,
        }

        # Tau
        lbtau = np.ones((nb_q, n_shooting + 1)) * -500
        ubtau = np.ones((nb_q, n_shooting + 1)) * 500
        tau0 = np.zeros((nb_q, n_shooting + 1))
        tau0[:nb_root, :] = 0.01

        # K
        lbk = np.ones((nb_k, n_shooting + 1)) * -50
        ubk = np.ones((nb_k, n_shooting + 1)) * 50
        k0 = np.ones((nb_k, n_shooting + 1)) * 0.01

        controls_lower_bounds = {
            "tau": lbtau,
            "k": lbk,
        }
        controls_upper_bounds = {
            "tau": ubtau,
            "k": ubk,
        }
        controls_initial_guesses = {
            "tau": tau0,
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
        motor_noise_magnitude = cas.DM(np.array([self.motor_noise_std**2 / self.initial_dt] * (self.model.nb_q - 3)))
        sensory_noise_magnitude = cas.DM(
            np.array(
                [
                    self.wPq_std**2 / self.initial_dt,
                    self.wPq_std**2 / self.initial_dt,
                    self.wPq_std**2 / self.initial_dt,
                    self.wPq_std**2 / self.initial_dt,
                    self.wPq_std**2 / self.initial_dt,
                    self.wPqdot_std**2 / self.initial_dt,
                    self.wPqdot_std**2 / self.initial_dt,
                    self.wPqdot_std**2 / self.initial_dt,
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

        # Terminal constraints
        g_floor, lbg_floor, ubg_floor = self.land_on_floor(
            discretization_method,
            dynamics_transcription,
            variables_vector,
            node=self.n_shooting,
        )
        constraints.add(
            g=g_floor,
            lbg=lbg_floor,
            ubg=ubg_floor,
            g_names=[f"land_on_floor"],
            node=self.n_shooting,
        )

        g_com, lbg_com, ubg_com = self.com_over_toes(
            discretization_method,
            dynamics_transcription,
            variables_vector,
            node=self.n_shooting,
        )
        constraints.add(
            g=g_com,
            lbg=lbg_com,
            ubg=ubg_com,
            g_names=[f"com_over_toes"],
            node=n_shooting,
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

        dt = variables_vector.get_time() / self.n_shooting
        nb_q = self.model.nb_q

        # Declare useful variables
        q = variables_vector.get_state("q", 0)
        qdot = variables_vector.get_state("qdot", 0)
        sensory_noise = noises_vector.get_sensory_noise(0)
        k = variables_vector.get_control("k", 0)
        k_matrix = self.model.reshape_vector_to_matrix(k, self.model.matrix_shape_k)
        ref = discretization_method.get_reference(
            self.model, x=variables_vector.get_states(0), u=variables_vector.get_controls(0)
        )

        # Minimize nominal efforts
        j_tau: cas.MX | cas.SX = 0
        for i_node in range(self.n_shooting):
            tau_control = variables_vector.get_control("tau", i_node)
            tau_friction = (
                -self.model.friction_coefficients
                @ discretization_method.get_mean_states(variables_vector, i_node, False)[nb_q + 3 : 2 * nb_q]
            )
            j_tau += 0.01 * cas.sum1(tau_control**2 * dt) + cas.sum1(tau_friction**2 * dt)

        # Minimize tau derivative
        j_tau_dot: cas.MX | cas.SX = 0
        for i_node in range(self.n_shooting - 1):
            tau_control = variables_vector.get_control("tau", i_node)
            tau_control_next = variables_vector.get_control("tau", i_node + 1)
            j_tau_dot += 0.01 * (tau_control_next - tau_control) ** 2

        # Minimize effort variability
        if discretization_method.name == "MeanAndCovariance":
            tau_fb = k_matrix @ (self.model.sensory_output(q, qdot, sensory_noise) - ref)
            jacobian_fb_x = cas.jacobian(tau_fb, variables_vector.get_states(0))
        else:
            q_this_time = variables_vector.get_specific_state("q", 0, 0)
            qdot_this_time = variables_vector.get_specific_state("qdot", 0, 0)

            sensory_noise_this_time = noises_vector.get_sensory_noise(0)[: self.model.nb_references]
            tau_fb_this_time = k_matrix @ (
                self.model.sensory_output(q_this_time, qdot_this_time, sensory_noise_this_time) - ref
            )
            jacobian_fb_x = cas.jacobian(tau_fb_this_time, cas.vertcat(q_this_time, qdot_this_time))

        cov_matrix = discretization_method.get_covariance(variables_vector, 0, is_matrix=True)
        sigma_ww = cas.diag(noises_vector.get_one_sensory_noise(0, 0))

        expected_feedback_variability = 0.01 * (
            cas.trace(jacobian_fb_x @ cov_matrix @ jacobian_fb_x.T) + cas.trace(k_matrix @ sigma_ww @ k_matrix.T)
        )

        sym_variables = [
            variables_vector.get_states(0),
            variables_vector.get_controls(0),
            noises_vector.get_one_sensory_noise(0, 0),
        ]
        if discretization_method.name == "MeanAndCovariance":
            sym_variables += [variables_vector.get_cov(0)]
        j_func = cas.Function("j_func", sym_variables, [expected_feedback_variability])

        j_fb_variability: cas.MX | cas.SX = 0
        for i_node in range(self.n_shooting):
            _, sensory_noise_magnitude = self.get_noises_magnitude()
            variables_this_time = [
                variables_vector.get_states(i_node),
                variables_vector.get_controls(i_node),
                sensory_noise_magnitude,
            ]
            if discretization_method.name == "MeanAndCovariance":
                variables_this_time += [variables_vector.get_cov(i_node)]

            j_fb_variability += j_func(*variables_this_time)

        # Minimize landing condition variability
        if discretization_method.name == "MeanAndCovariance":
            CoM_position = self.model.center_of_mass()(q)[1]
            CoM_velocity = self.model.center_of_mass_velocity()(q, qdot)[1]
            CoM_angular_velocity = self.model.body_rotation_rate()(q, qdot)[0]
            jacobian_position_x = cas.jacobian(CoM_position, variables_vector.get_states(0))
            jacobian_velocity_x = cas.jacobian(CoM_velocity, variables_vector.get_states(0))
            jacobian_angular_velocity_x = cas.jacobian(CoM_angular_velocity, variables_vector.get_states(0))
        else:
            q_this_time = variables_vector.get_specific_state("q", 0, 0)
            qdot_this_time = variables_vector.get_specific_state("qdot", 0, 0)

            CoM_position_this_time = self.model.center_of_mass()(q_this_time)[1]
            CoM_velocity_this_time = self.model.center_of_mass_velocity()(q_this_time, qdot_this_time)[1]
            CoM_angular_velocity_this_time = self.model.body_rotation_rate()(q_this_time, qdot_this_time)[0]

            jacobian_position_x = cas.jacobian(CoM_position_this_time, cas.vertcat(q_this_time, qdot_this_time))
            jacobian_velocity_x = cas.jacobian(CoM_velocity_this_time, cas.vertcat(q_this_time, qdot_this_time))
            jacobian_angular_velocity_x = cas.jacobian(
                CoM_angular_velocity_this_time, cas.vertcat(q_this_time, qdot_this_time)
            )

        cov_matrix = discretization_method.get_covariance(variables_vector, 0, is_matrix=True)

        landing_variability = 10000 * (
            cas.trace(jacobian_position_x @ cov_matrix @ jacobian_position_x.T)
            + cas.trace(jacobian_velocity_x @ cov_matrix @ jacobian_velocity_x.T)
            + cas.trace(jacobian_angular_velocity_x @ cov_matrix @ jacobian_angular_velocity_x.T)
        )

        sym_variables = [
            variables_vector.get_states(0),
            variables_vector.get_controls(0),
        ]
        if discretization_method.name == "MeanAndCovariance":
            sym_variables += [variables_vector.get_cov(0)]
        j_func = cas.Function("j_func", sym_variables, [landing_variability])

        j_landing_variability: cas.MX | cas.SX = 0
        for i_node in range(self.n_shooting):
            variables_this_time = [
                variables_vector.get_states(i_node),
                variables_vector.get_controls(i_node),
            ]
            if discretization_method.name == "MeanAndCovariance":
                variables_this_time += [variables_vector.get_cov(i_node)]

            j_landing_variability += j_func(*variables_this_time)

        # Minimize time
        j_time = 0.01 * variables_vector.get_time()

        # Minimize feedbacks
        j_k: cas.MX | cas.SX = 0
        for i_node in range(self.n_shooting):
            k_control = variables_vector.get_control("k", i_node)
            j_k += 1e-5 * cas.sum1(k_control**2 * dt)

        # Minimize feedbacks derivative
        j_k_dot: cas.MX | cas.SX = 0
        for i_node in range(self.n_shooting - 1):
            k_control = variables_vector.get_control("k", i_node)
            k_control_next = variables_vector.get_control("k", i_node + 1)
            j_k_dot += 1 * (k_control_next - k_control) ** 2

        return j_tau + j_tau_dot + j_fb_variability + j_landing_variability + j_time + j_k + j_k_dot

    # --- helper functions --- #
    def land_on_floor(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        node: int,
    ) -> tuple[list[cas.MX | cas.SX], list[float], list[float]]:

        toe_idx = self.model.marker_index("Foot_Toe")

        if discretization_method.name == "MeanAndCovariance":
            q = variables_vector.get_state("q", node)
            mean_height = self.model.marker(toe_idx)(q)[2]
        else:
            toe_marker_height = cas.SX() if self.model.use_sx else cas.MX()
            for i_random in range(self.nb_random):
                q_this_time = variables_vector.get_specific_state("q", node, i_random)
                toe_marker_height = cas.vertcat(
                    toe_marker_height, 1 / self.nb_random * self.model.marker(toe_idx)(q_this_time)[2]
                )

            mean_height = cas.sum1(toe_marker_height)

        g = mean_height
        lbg = [0]
        ubg = [0]
        return g, lbg, ubg

    def com_over_toes(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        node: int,
    ) -> tuple[list[cas.MX | cas.SX], list[float], list[float]]:

        toe_idx = self.model.marker_index("Foot_Toe")

        if discretization_method.name == "MeanAndCovariance":
            q = variables_vector.get_state("q", node)
            mean_diff = self.model.marker(toe_idx)(q)[1] - self.model.center_of_mass()(q)[1]
        else:
            CoM_pos = cas.SX() if self.model.use_sx else cas.MX()
            marker_pos = cas.SX() if self.model.use_sx else cas.MX()
            for i_random in range(self.nb_random):
                q_this_time = variables_vector.get_specific_state("q", node, i_random)
                CoM_pos = cas.vertcat(CoM_pos, 1 / self.nb_random * self.model.center_of_mass()(q_this_time)[1])
                marker_pos = cas.vertcat(marker_pos, 1 / self.nb_random * self.model.marker(toe_idx)(q_this_time)[1])

            mean_diff = cas.sum1(marker_pos) - cas.sum1(CoM_pos)

        g = [mean_diff]
        lbg = [0]
        ubg = [0]
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
        pass
        # n_shooting = ocp["n_shooting"]
        # states_opt_mean = data_saved["states_opt_mean"]
        #
        # q_mean = states_opt_mean[ocp["ocp_example"].model.q_indices, :]
        # q_init = data_saved["states_init_array"][ocp["ocp_example"].model.q_indices, :]
        # u_opt = data_saved["controls_opt_array"][ocp["ocp_example"].model.u_indices, :]
        # q_opt = data_saved["states_opt_array"][ocp["ocp_example"].model.q_indices, :]
        # q_simulated = data_saved["x_simulated"][: ocp["ocp_example"].model.nb_q, :, :]
        # n_simulations = q_simulated.shape[2]
        # covariance_simulated = data_saved["covariance_simulated"]
        # cov_opt = data_saved["cov_opt_array"]
        #
        # fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        #
        # if isinstance(ocp["discretization_method"], MeanAndCovariance):
        #     marker_position_opt = np.zeros((2, n_shooting + 1))
        #     for i_node in range(n_shooting + 1):
        #         marker_position_opt[:, i_node] = self.model.end_effector_position(q_opt[:, i_node])
        #
        #     ax.plot(
        #         marker_position_opt[0, :],
        #         marker_position_opt[1, :],
        #         "--k",
        #         label="Initial guess",
        #         linewidth=0.5,
        #         alpha=0.3,
        #     )
        #     ax.plot(marker_position_opt[0, 0], marker_position_opt[1, 0], "og", label="Optimal initial node")
        # else:
        #     marker_position_opt = np.zeros((2, n_shooting + 1, ocp["ocp_example"].nb_random))
        #     for i_random in range(ocp["ocp_example"].nb_random):
        #         for i_node in range(n_shooting + 1):
        #             marker_position_opt[:, i_node, i_random] = self.model.end_effector_position(
        #                 q_opt[:, i_node, i_random]
        #             )
        #
        #         if i_random == 0:
        #             ax.plot(
        #                 marker_position_opt[0, 0, i_random],
        #                 marker_position_opt[1, 0, i_random],
        #                 "og",
        #                 label="Optimal initial node",
        #             )
        #         else:
        #             ax.plot(marker_position_opt[0, 0, i_random], marker_position_opt[1, 0, i_random], "og")
        #         ax.plot(
        #             marker_position_opt[0, :, i_random],
        #             marker_position_opt[1, :, i_random],
        #             "-",
        #             color="g",
        #             linewidth=0.5,
        #             alpha=0.3,
        #         )
        #
        # marker_position_simulated = np.zeros((2, n_shooting + 1, n_simulations))
        # for i_simulation in range(n_simulations):
        #     for i_node in range(n_shooting + 1):
        #         marker_position_simulated[:, i_node, i_simulation] = self.model.end_effector_position(
        #             q_simulated[:, i_node, i_simulation]
        #         )
        #
        # q_simulated_mean = np.mean(q_simulated, axis=2)
        # for i_node in range(n_shooting):
        #     if i_node == 0:
        #         # self.draw_cov_ellipse(
        #         #     cov=cov_opt[:2, :2, i_node], pos=q_mean[:, i_node], ax=ax[0], color="b", label="Cov optimal"
        #         # )
        #         ax.plot(
        #             marker_position_simulated[0, i_node, :],
        #             marker_position_simulated[1, i_node, :],
        #             ".r",
        #             markersize=1,
        #             label="Noisy integration",
        #         )
        #         # self.draw_cov_ellipse(
        #         #     cov=covariance_simulated[:2, :2, i_node],
        #         #     pos=q_simulated_mean[:, i_node],
        #         #     ax=ax[0],
        #         #     color="r",
        #         #     label="Cov simulated",
        #         # )
        #     else:
        #         # self.draw_cov_ellipse(cov=cov_opt[:2, :2, i_node], pos=q_mean[:, i_node], ax=ax[0], color="b")
        #         ax.plot(
        #             marker_position_simulated[0, i_node, :], marker_position_simulated[1, i_node, :], ".r", markersize=1
        #         )
        #         # self.draw_cov_ellipse(
        #         #     cov=covariance_simulated[:2, :2, i_node],
        #         #     pos=q_simulated_mean[:, i_node],
        #         #     ax=ax[0],
        #         #     color="r",
        #         # )
        #
        # marker_position_mean = np.zeros((2, n_shooting + 1))
        # for i_node in range(n_shooting + 1):
        #     marker_position_mean[:, i_node] = self.model.end_effector_position(q_mean[:, i_node])
        # ax.plot(
        #     marker_position_mean[0, :],
        #     marker_position_mean[1, :],
        #     "-o",
        #     color="g",
        #     markersize=1,
        #     linewidth=2,
        #     label="Optimal trajectory",
        # )
        # # ax.set_xlim(-1, 1)
        # # ax.set_ylim(-0.16, 0.16)
        #
        # ax.legend()
        # plt.savefig(fig_save_path)
        # # plt.show()
        # plt.close()
