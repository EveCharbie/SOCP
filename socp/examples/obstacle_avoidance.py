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
from ..models.mass_point_model import MassPointModel
from ..models.model_abstract import ModelAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract
from ..transcriptions.transcription_abstract import TranscriptionAbstract
from ..transcriptions.variables_abstract import VariablesAbstract
from ..constraints import Constraints


# Taken from Gillis et al. 2013
def superellipse(
    a: int = 1,
    b: int = 1,
    n: int = 2,
    x_0: float = 0,
    y_0: float = 0,
    resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-2 * a + x_0, 2 * a + x_0, resolution)
    y = np.linspace(-2 * b + y_0, 2 * b + y_0, resolution)

    X, Y = np.meshgrid(x, y)
    Z = ((X - x_0) / a) ** n + ((Y - y_0) / b) ** n - 1
    return X, Y, Z


class ObstacleAvoidance(ExampleAbstract):
    def __init__(self, is_robustified: bool = True):
        super().__init__()  # Does nothing

        self.nb_random = 15
        self.n_threads = 7
        self.n_simulations = 30
        self.seed = 0
        self.model = MassPointModel(self.nb_random)
        self.is_robustified = is_robustified

        # Noise parameters (from Van Wouwe et al. 2022)
        self.final_time = 4.0
        self.min_time = 0.5
        self.max_time = 10.0
        self.n_shooting = 40
        self.initial_state_variability = np.array([0.01, 0.01, 0.01, 0.01])

        # Solver options
        self.tol = 1e-6
        self.max_iter = 1000

    def name(self) -> str:
        return "ObstacleAvoidance"

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
        lbq = np.ones((nb_q, n_shooting + 1)) * -10
        ubq = np.ones((nb_q, n_shooting + 1)) * 10
        # # Start with X = 0
        # lbq[0, 0] = 0
        # ubq[0, 0] = 0
        # # Make sure it goes around the obstacle
        # ubq[1, int(round(n_shooting / 2))] = -0.9

        # Use a circle as initial guess
        q0 = np.zeros((nb_q, n_shooting + 1))
        for i_node in range(n_shooting + 1):
            q0[0, i_node] = 3 * np.sin(i_node * 2 * np.pi / n_shooting)
            q0[1, i_node] = 3 * np.cos(i_node * 2 * np.pi / n_shooting)

        # Qdot
        lbqdot = np.ones((nb_q, n_shooting + 1)) * -20
        ubqdot = np.ones((nb_q, n_shooting + 1)) * 20
        qdot0 = np.zeros((nb_q, n_shooting + 1))

        # # Covariance
        # lbcov = np.ones((nb_q * 2, nb_q * 2, n_shooting + 1)) * -cas.inf
        # ubcov = np.ones((nb_q * 2, nb_q * 2, n_shooting + 1)) * cas.inf
        # cov0 = np.repeat(
        #     np.array(cas.DM.eye(nb_q * 2) * self.initial_state_variability)[:, :, np.newaxis], n_shooting + 1, axis=2
        # )
        #
        # # helper matrix
        # lbm = np.ones((nb_q * 2, nb_q * 2, nb_collocation_points, n_shooting + 1)) * -cas.inf
        # ubm = np.ones((nb_q * 2, nb_q * 2, nb_collocation_points, n_shooting + 1)) * cas.inf
        # m0 = np.zeros((nb_q * 2, nb_q * 2, nb_collocation_points, n_shooting + 1))

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
            # Initialize with a circle
            # plt.figure()
            # colors = ["y", "g", "c", "b", "tab:purple", "m", "r"]
            qz0 = np.zeros((nb_q, nb_collocation_points, n_shooting + 1))
            nb_point_total = nb_collocation_points * n_shooting
            for i_node in range(n_shooting):
                for i_collocation in range(nb_collocation_points):
                    idx = nb_collocation_points * i_node + i_collocation
                    qz0[0, i_collocation, i_node] = 3 * np.sin(idx * 2 * np.pi / nb_point_total)
                    qz0[1, i_collocation, i_node] = 3 * np.cos(idx * 2 * np.pi / nb_point_total)
            #         if i_collocation == 0:
            #             plt.plot(qz0[0, i_collocation, i_node], qz0[1, i_collocation, i_node], ".", color=colors[i_collocation])
            #         else:
            #             plt.plot(qz0[0, i_collocation, i_node], qz0[1, i_collocation, i_node], "o", color=colors[i_collocation])
            #     plt.plot(q0[0, i_node], q0[1, i_node], "x", color="k")
            # plt.savefig("tt.png")
            # plt.show()

            qz0[0, 0, n_shooting] = 3 * np.sin(nb_point_total * 2 * np.pi / nb_point_total)
            qz0[1, 0, n_shooting] = 3 * np.cos(nb_point_total * 2 * np.pi / nb_point_total)
            qdotz0 = np.zeros((nb_q, nb_collocation_points, n_shooting + 1))

            collocation_points_initial_guesses = {
                "q": qz0,
                "qdot": qdotz0,
            }

        # u
        lbu = np.ones((nb_q, n_shooting)) * -20
        ubu = np.ones((nb_q, n_shooting)) * 20
        # u0 = q0[:, :-1]  # Guide-point initial guess is the same as the mass point position
        u0 = np.zeros((nb_q, n_shooting))  # Guide-point initial guess is the same as the mass point position

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
        motor_noise_magnitude = np.array([1, 1]) * 1
        return motor_noise_magnitude, None

    def set_specific_constraints(
        self,
        model: ModelAbstract,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        noises_single: list,
        noises_numerical: list,
        constraints: Constraints,
    ) -> None:

        # Obstacle avoidance constraints
        for i_node in range(self.n_shooting):
            g_obstacle, lbg_obstacle, ubg_obstacle = self.obstacle_avoidance(
                discretization_method,
                dynamics_transcription,
                variables_vector,
                noises_single,
                i_node,
                is_robustified=self.is_robustified,
            )
            constraints.add(
                g=g_obstacle,
                lbg=lbg_obstacle,
                ubg=ubg_obstacle,
                g_names=[f"obstacle_avoidance"] * len(lbg_obstacle),
                node=i_node,
            )

        # Start with x = 0
        constraints.add(
            g=variables_vector.get_states(0)[0],
            lbg=[0],
            ubg=[0],
            g_names=["initial_position_x"],
            node=0,
        )

        # Cyclicity
        nb_total_states = variables_vector.get_states(0).shape[0]
        constraints.add(
            g=variables_vector.get_states(0) - variables_vector.get_states(self.n_shooting),
            lbg=[0] * nb_total_states,
            ubg=[0] * nb_total_states,
            g_names=[f"cyclicity_states"] * nb_total_states,
            node=0,
        )
        if discretization_method.with_cholesky:
            nb_cov_variables = self.model.nb_cholesky_components(self.model.nb_states)
        else:
            nb_cov_variables = self.model.nb_states * self.model.nb_states
        constraints.add(
            g=variables_vector.get_cov(0) - variables_vector.get_cov(self.n_shooting),
            lbg=[0] * nb_cov_variables,
            ubg=[0] * nb_cov_variables,
            g_names=[f"cyclicity_cov"] * nb_cov_variables,
            node=0,
        )

        # Initial cov matrix must be semidefinite positive (Sylvester's criterion)
        cov_matrix = variables_vector.get_cov_matrix(0)
        epsilon = 1e-6
        for k in range(1, self.model.nb_states + 1):
            minor = cas.det(cov_matrix[:k, :k])
            constraints.add(
                g=minor,
                lbg=epsilon,
                ubg=cas.inf,
                g_names="covariance_positive_definite_minor_" + str(k),
                node=0,
            )

        # # Initial cov matrix is imposed
        # cov_matrix = variables_vector.reshape_matrix_to_vector(variables_vector.get_cov_matrix(0))
        # p_init = variables_vector.reshape_matrix_to_vector(np.diag(self.initial_state_variability.tolist()))
        # constraints.add(
        #     g=cov_matrix - p_init,
        #     lbg=[0] * (variables_vector.nb_states * variables_vector.nb_states),
        #     ubg=[0] * (variables_vector.nb_states * variables_vector.nb_states),
        #     g_names=["initial_covariance"] * (variables_vector.nb_states * variables_vector.nb_states),
        #     node=0,
        # )

        # # No initial acceleration
        # xdot_init = dynamics_transcription.dynamics_func(x_all[0], u_all[0], cas.DM.zeros(self.model.nb_noises))
        # g += [xdot_init[self.model.qdot_indices]]
        # lbg += [0] * self.model.nb_q
        # ubg += [0] * self.model.nb_q
        # g_names += [f"initial_acceleration"] * self.model.nb_q

        return

    def get_specific_objectives(
        self,
        model: ModelAbstract,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        noises_single: list[cas.SX],
        noises_numerical: list[cas.DM],
    ) -> cas.SX:

        # Minimize time
        j_time = variables_vector.get_time()

        # Regularization on controls
        weight = 1e-2 / (2 * self.n_shooting)
        j_controls = 0
        for i_node in range(self.n_shooting):
            j_controls += cas.sum1(variables_vector.get_controls(i_node) ** 2)

        # j_control_derivative = 0
        # for i_node in range(self.n_shooting - 1):
        #     j_control_derivative += cas.sum1((variables_vector.get_controls(i_node + 1) - variables_vector.get_controls(i_node)) ** 2)

        return j_time + weight * j_controls

    # --- helper functions --- #
    def obstacle_avoidance(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        noise_single: cas.SX,
        node: int,
        is_robustified: bool = True,
    ) -> tuple[list[cas.SX], list[float], list[float]]:

        g = []
        lbg = []
        ubg = []

        states = variables_vector.get_states(node)
        p_x = states[0]
        p_y = states[1]
        for i_super_elipse in range(2):
            cx = self.model.super_ellipse_center_x[i_super_elipse]
            cy = self.model.super_ellipse_center_y[i_super_elipse]
            a = self.model.super_ellipse_a[i_super_elipse]
            b = self.model.super_ellipse_b[i_super_elipse]
            n = self.model.super_ellipse_n[i_super_elipse]

            h = ((p_x - cx)/ a) ** n + ((p_y - cy) / b)**n - 1

            if is_robustified:
                """
                I modified the order of the constraint from the original article because I had negative values in the sqrt.
                Instead of a - 1 - safe_guard > 0,
                I use (a - 1) ** 2 - safe_guard ** 2 > 0,
                """
                cov_matrix = variables_vector.get_cov_matrix(node)
                gamma = 1
                dh_dx = cas.jacobian(h, states)
                # safe_guard_squared = gamma ** 2 * dh_dx @ cov_sym @ dh_dx.T
                # h = (a - 1) ** 2 - safe_guard_squared
                safe_guard = gamma * cas.sqrt(dh_dx @ cov_matrix @ dh_dx.T)
                h -= safe_guard

            g += [h]
            lbg += [0]
            ubg += [cas.inf]

        return g, lbg, ubg

    # --- plotting functions --- #
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
        u_opt = data_saved["controls_opt_array"][ocp["ocp_example"].model.q_indices, :]
        q_opt = data_saved["states_opt_array"][ocp["ocp_example"].model.q_indices, :]
        q_simulated = data_saved["x_simulated"][: ocp["ocp_example"].model.nb_q, :, :]
        n_simulations = q_simulated.shape[2]
        covariance_simulated = data_saved["covariance_simulated"]
        cov_opt = data_saved["cov_opt_array"]

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        for i_ellipse in range(2):
            a = ocp["ocp_example"].model.super_ellipse_a[i_ellipse]
            b = ocp["ocp_example"].model.super_ellipse_b[i_ellipse]
            n = ocp["ocp_example"].model.super_ellipse_n[i_ellipse]
            x_0 = ocp["ocp_example"].model.super_ellipse_center_x[i_ellipse]
            y_0 = ocp["ocp_example"].model.super_ellipse_center_y[i_ellipse]

            X, Y, Z = superellipse(a, b, n, x_0, y_0)

            ax[0].contourf(X, Y, Z, levels=[-1000, 0], colors=["#DA1984"], alpha=0.5)

        ax[0].plot(q_init[0, :], q_init[1, :], "-k", label="Initial guess")
        ax[0].plot(q_opt[0, 0], q_opt[1, 0], "og", label="Optimal initial node")

        ax[1].plot(q_opt[0], q_opt[1], "b", label="Optimal trajectory")
        ax[1].plot(u_opt[0], u_opt[1], "r", label="Optimal controls")
        for i_node in range(n_shooting):
            if i_node == 0:
                ax[1].plot(
                    (u_opt[0][i_node], q_opt[0, i_node]),
                    (u_opt[1][i_node], q_opt[1, i_node]),
                    ":k",
                    label="Spring orientation",
                )
            else:
                ax[1].plot((u_opt[0, i_node], q_opt[0, i_node]), (u_opt[1, i_node], q_opt[1, i_node]), ":k")
        ax[1].legend()

        for i_node in range(n_shooting):
            if i_node == 0:
                ax[0].plot(
                    q_simulated[0, i_node, :], q_simulated[1, i_node, :], ".r", markersize=1, label="Noisy integration"
                )
                self.draw_cov_ellipse(
                    cov=cov_opt[:2, :2, i_node], pos=q_opt[:, i_node], ax=ax[0], color="b", label="Cov optimal"
                )
            else:
                ax[0].plot(q_simulated[0, i_node, :], q_simulated[1, i_node, :], ".r", markersize=1)
                self.draw_cov_ellipse(cov=cov_opt[:2, :2, i_node], pos=q_opt[:, i_node], ax=ax[0], color="b")

        ax[0].plot(q_opt[0], q_opt[1], "-o", color="g", markersize=1, label="Optimal trajectory")

        ax[0].legend()
        plt.savefig(fig_save_path)
        plt.show()

    @staticmethod
    def draw_cov_ellipse(cov: np.ndarray, pos: np.ndarray, ax: plt.Axes, **kwargs):
        """
        Draw an ellipse representing the covariance at a given point.
        """

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * np.sqrt(vals)
        ellip = plt.matplotlib.patches.Ellipse(xy=pos, width=width, height=height, angle=theta, alpha=0.3, **kwargs)

        ax.add_patch(ellip)
        return ellip
