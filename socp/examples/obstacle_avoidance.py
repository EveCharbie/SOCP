"""
This example implements a stochastic optimal control problem for mass-point circling around obstacles.
The goal is to find a time-optimal stable periodic trajectory.
The controls are coordinates of a quide point (the mass is attached to this guide point with a sping).
This example was taken from Gillis et al. 2013.
"""

import numpy as np
import casadi as cas

from .example_abstract import ExampleAbstract
from ..models.mass_point_model import MassPointModel
from ..models.model_abstract import ModelAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract
from ..transcriptions.transcription_abstract import TranscriptionAbstract


# Taken from Gillis et al. 2013
def superellipse(
        a: int = 1,
        b: int = 1,
        n: int= 2,
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
    def __init__(self):
        super().__init__()  # Does nothing

        self.nb_random = 15
        self.n_threads = 7
        self.n_simulations = 30
        self.seed = 0
        self.model = MassPointModel(self.nb_random)

        # Noise parameters (from Van Wouwe et al. 2022)
        self.final_time = 4.0
        self.min_time = 0.1
        self.max_time = 40.0
        self.n_shooting = 40
        self.initial_state_variability = np.array([0.01, 0.01, 0.01, 0.01])

        # Solver options
        self.tol = 1e-6
        self.max_iter = 1000

    def name(self) -> str:
        return "ObstacleAvoidance"

    def get_bounds_and_init(
        self,
        n_shooting,
    ) -> tuple[
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

        # This is the real initialization, but I decided to have linear interpolation for now on z
        # q_init = np.zeros((nb_q, (order + 2) * n_shooting + 1))
        # zq_init = initialize_circle((order + 1) * n_shooting + 1)
        # for i in range(n_shooting + 1):
        #     j = i * (order + 1)
        #     k = i * (order + 2)
        #     q_init[:, k] = zq_init[:, j]
        #     q_init[:, k + 1: k + 1 + (order + 1)] = zq_init[:, j: j + (order + 1)]

        # Q
        lbq = np.ones((nb_q, n_shooting + 1)) * -10
        ubq = np.ones((nb_q, n_shooting + 1)) * 10
        lbq[0, 0] = 0  # Start with X = 0
        lbq[0, 0] = 0
        q0 = np.zeros((nb_q, n_shooting + 1))
        # Use a circle as initial guess
        q_init = np.zeros((2, n_shooting + 1))
        for i_node in range(n_shooting + 1):
            q_init[0, i_node] = 3 * np.sin(i_node * 2 * np.pi / n_shooting)
            q_init[1, i_node] = 3 * np.cos(i_node * 2 * np.pi / n_shooting)

        # Qdot
        lbqdot = np.ones((nb_q, n_shooting + 1)) * -20
        ubqdot = np.ones((nb_q, n_shooting + 1)) * 20
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

        # u
        lbu = np.ones((nb_q, n_shooting)) * -20
        ubu = np.ones((nb_q, n_shooting)) * 20
        u0 = np.zeros((nb_q, n_shooting))

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
        )

    def get_noises_magnitude(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the motor and sensory noise magnitude.
        """
        motor_noise_magnitude = np.array([1, 1]) * 1
        return motor_noise_magnitude, None

    def get_specific_constraints(
        self,
        model: ModelAbstract,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        x_all: list,
        u_all: list,
        noises_single: list,
        noises_numerical: list,
    ):

        g = []
        lbg = []
        ubg = []
        g_names = []

        # Obstacle avoidance constraints
        for i_node in range(self.n_shooting + 1):
            g_obstacle, lbg_obstacle, ubg_obstacle = self.obstacle_avoidance(
                discretization_method, dynamics_transcription, x_all[i_node], u_all[i_node], noises_single
            )
            g += g_obstacle
            lbg += lbg_obstacle
            ubg += ubg_obstacle
            g_names += [f"obstacle_avoidance_node_{i_node}"] * len(lbg_obstacle)

        # Cyclicity
        g += [x_all[0][: self.model.nb_states] - x_all[-1][: self.model.nb_states]]
        lbg += [0] * self.model.nb_states
        ubg += [0] * self.model.nb_states
        g_names += [f"cyclicity_states"] * self.model.nb_states
        if discretization_method.with_cholesky:
            nb_cov_variables = self.model.nb_cholesky_components(self.model.nb_states)
        else:
            nb_cov_variables = self.model.nb_states * self.model.nb_states
        g += [x_all[0][self.model.nb_states: self.model.nb_states + nb_cov_variables] - x_all[-1][self.model.nb_states: self.model.nb_states + nb_cov_variables]]
        lbg += [0] * nb_cov_variables
        ubg += [0] * nb_cov_variables
        g_names += [f"cyclicity_cov"] * nb_cov_variables

        return g, lbg, ubg, g_names

    def get_specific_objectives(
        self,
        model: ModelAbstract,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        T: cas.SX,
        x_all: list[cas.SX],
        u_all: list[cas.SX],
        noises_single: list[cas.SX],
        noises_numerical: list[cas.DM],
    ) -> cas.SX:

        # Minimize time
        j_time = T

        # Regularization on controls
        weight = 1e-2 / (2 * self.n_shooting)
        j_controls = 0
        for i_node in range(self.n_shooting):
            j_controls += cas.sum1(u_all[i_node] ** 2)
        return j_time + weight * j_controls

    # --- helper functions --- #
    def obstacle_avoidance(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        x_single: cas.SX,
        u_single: cas.SX,
        noise_single: cas.SX,
    ) -> tuple[list[cas.SX], list[float], list[float]]:

        g = []
        lbg = []
        ubg = []

        q = x_single[: self.model.nb_q]
        p_x = q[0]
        p_y = q[1]
        for i_super_elipse in range(2):
            h = (
                    (
                            (p_x - self.model.super_ellipse_center_x[i_super_elipse])
                            / self.model.super_ellipse_a[i_super_elipse]
                    )
                    ** self.model.super_ellipse_n[i_super_elipse]
                    + (
                            (p_y - self.model.super_ellipse_center_y[i_super_elipse])
                            / self.model.super_ellipse_b[i_super_elipse]
                    )
                    ** self.model.super_ellipse_n[i_super_elipse]
                    - 1
            )

            g += [h]
            lbg += [0]
            ubg += [cas.inf]

        # TODO: Implement robustified constraint properly
        # if is_robustified:
        #     gamma = 1
        #     dh_dx = cas.jacobian(h, controller.states.cx)
        #     cov = StochasticBioModel.reshape_to_matrix(controller.controls["cov"].cx, controller.model.matrix_shape_cov)
        #     safe_guard = gamma * cas.sqrt(dh_dx @ cov @ dh_dx.T)
        #     out -= safe_guard

        return g, lbg, ubg

    def mean_start_on_target(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        x_single: cas.SX,
        u_single: cas.SX,
    ) -> tuple[list[cas.SX], list[float], list[float]]:
        """
        Constraint to impose that the mean trajectory reaches the target at the end of the movement
        """
        ee_pos_mean = discretization_method.get_reference(
            self.model,
            x_single,
            u_single,
        )[:2]
        g = [ee_pos_mean - HAND_INITIAL_TARGET]
        lbg = [0, 0]
        ubg = [0, 0]
        return g, lbg, ubg

    def mean_reach_target(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        x_single: cas.SX,
        u_single: cas.SX,
    ) -> tuple[list[cas.SX], list[float], list[float]]:
        """
        Constraint to impose that the mean trajectory reaches the target at the end of the movement
        """
        # The mean end-effector position is on the target
        ee_pos_mean = discretization_method.get_reference(
            self.model,
            x_single,
            u_single,
        )[:2]
        g = [ee_pos_mean - HAND_FINAL_TARGET]
        lbg = [0, 0]
        ubg = [0, 0]

        # All hand positions are inside a circle of radius 4 mm around the target
        ee_pos_variability_x, ee_pos_variability_y = discretization_method.get_ee_variance(
            self.model,
            x_single,
            u_single,
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
        x_single: cas.SX,
        u_single: cas.SX,
    ) -> tuple[list[cas.SX], list[float], list[float]]:
        """
        Constraint to impose that the mean hand velocity is null at the end of the movement
        """
        ee_velo_mean = discretization_method.get_reference(
            self.model,
            x_single,
            u_single,
        )[2:4]
        g = [ee_velo_mean]
        lbg = [0, 0]
        ubg = [0, 0]
        return g, lbg, ubg

    def minimize_stochastic_efforts_and_variations(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        x_single: cas.SX,
        u_single: cas.SX,
    ) -> cas.SX:

        activations_mean = discretization_method.get_mean_states(
            self.model,
            x_single,
            squared=True,
        )[4 : 4 + self.model.nb_muscles]
        efforts = cas.sum1(activations_mean)

        activations_variations = discretization_method.get_mus_variance(
            self.model,
            x_single,
        )

        j = efforts + activations_variations / 2

        return j
