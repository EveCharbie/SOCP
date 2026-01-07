import numpy as np
import casadi as cas

from .example_abstract import ExampleAbstract
from ..models.arm_model import ArmModel
from ..models.model_abstract import ModelAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract


# Taken from Van Wouwe et al. 2022
HAND_INITIAL_TARGET = np.array([0.0, 0.2742])
HAND_FINAL_TARGET = np.array([0.0, 0.527332023564034])


class ArmReaching(ExampleAbstract):
    def __init__(self):
        super().__init__()  # Does nothing

        self.n_random = 15
        self.n_threads = 7
        self.n_simulations = 30
        self.seed = 0
        self.model = ArmModel(self.n_random)

        # Noise parameters (from Van Wouwe et al. 2022)
        self.dt = 0.05
        self.final_time = 0.8
        self.n_shooting = int(self.final_time / self.dt)
        self.motor_noise_std = 0.05  # Tau noise
        self.wPq_std = 3e-4  # Hand position noise
        self.wPqdot_std = 2.4e-3  # Hand velocity noise

        # Solver options
        self.tol = 1e-6
        self.max_iter = 1000

    def name(self) -> str:
        return "ArmReaching"

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
        ubq[0, :] = np.pi / 2
        ubq[1, :] = 7 / 8 * np.pi
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
        lbmusa = np.ones((n_muscles, n_shooting + 1)) * 1e-6
        ubmusa = np.ones((n_muscles, n_shooting + 1))
        musa0 = np.ones((n_muscles, n_shooting + 1)) * 0.1

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

        # MuscleExcitation
        lbmuse = np.ones((n_muscles, n_shooting + 1)) * 1e-6
        ubmuse = np.ones((n_muscles, n_shooting + 1))
        muse0 = np.ones((n_muscles, n_shooting + 1)) * 0.1

        # K
        lbk = np.ones((nb_k, n_shooting + 1)) * -10
        ubk = np.ones((nb_k, n_shooting + 1)) * 10
        k0 = np.ones((nb_k, n_shooting + 1)) * 0.1

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
        )

    def get_noises_magnitude(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the motor and sensory noise magnitude.
        """
        motor_noise_magnitude = cas.DM(np.array([self.motor_noise_std**2 / self.dt] * self.model.nb_q))
        sensory_noise_magnitude = cas.DM(
            np.array(
                [
                    self.wPq_std**2 / self.dt,
                    self.wPq_std**2 / self.dt,
                    self.wPqdot_std**2 / self.dt,
                    self.wPqdot_std**2 / self.dt,
                ]
            )
        )
        return motor_noise_magnitude, sensory_noise_magnitude

    def get_specific_constraints(
        self,
        model: ModelAbstract,
        discretization: DiscretizationAbstract,
        x: list,
        u: list,
        noises_single: list,
        noises_numerical: list,
    ):

        g = []
        lbg = []
        ubg = []
        g_names = []

        # Initial constraint
        g_target, lbg_target, ubg_target = self.mean_start_on_target(discretization, x[0], u[0])
        g += g_target
        lbg += lbg_target
        ubg += ubg_target
        g_names += [f"mean_start_on_target"] * len(lbg_target)

        # Terminal constraint
        g_target, lbg_target, ubg_target = self.mean_reach_target(discretization, x[-1], u[0])
        g += g_target
        lbg += lbg_target
        ubg += ubg_target
        g_names += [f"mean_reach_target"] * len(lbg_target)

        g_target, lbg_target, ubg_target = self.mean_end_effector_velocity(discretization, x[-1], u[0])
        g += g_target
        lbg += lbg_target
        ubg += ubg_target
        g_names += [f"mean_end_effector_velocity"] * 2

        return g, lbg, ubg, g_names

    def get_specific_objectives(
        self,
        model: object,
        discretization: DiscretizationAbstract,
        x: list,
        u: list,
        noises_single: list,
        noises_numerical: list,
    ) -> cas.MX:
        j = 0
        for i_node in range(self.n_shooting):
            j += self.minimize_stochastic_efforts_and_variations(discretization, x[i_node]) * self.dt / 2
        return j

    # --- helper functions --- #
    def get_end_effector_for_all_random(
            self,
            discretization: DiscretizationAbstract,
            x_single: cas.MX,
            u_single: cas.MX,
    ) -> tuple[cas.MX, cas.MX]:
        """
        Get the end-effector position and velocity for all random trials
        """
        ee_pos_velo_mean = discretization.get_reference(
            self.model,
            x_single,
            u_single,
            )
        ee_pos = ee_pos_velo_mean[:2]
        ee_vel = ee_pos_velo_mean[2:4]
        return ee_pos, ee_vel

    def mean_start_on_target(
            self,
            discretization: DiscretizationAbstract,
            x_single: cas.MX,
            u_single: cas.MX,
    ) -> tuple[list[cas.MX], list[float], list[float]]:
        """
        Constraint to impose that the mean trajectory reaches the target at the end of the movement
        """
        ee_pos_mean = discretization.get_reference(
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
            discretization: DiscretizationAbstract,
            x_single: cas.MX,
            u_single: cas.MX,
    ) -> tuple[list[cas.MX], list[float], list[float]]:
        """
        Constraint to impose that the mean trajectory reaches the target at the end of the movement
        """
        ee_pos_mean = discretization.get_reference(
            self.model,
            x_single,
            u_single,
            )[:2]
        g = [ee_pos_mean - HAND_FINAL_TARGET]
        lbg = [0, 0]
        ubg = [0, 0]

        return g, lbg, ubg

    def mean_end_effector_velocity(
            self,
            discretization: DiscretizationAbstract,
            x_single: cas.MX,
            u_single: cas.MX,
    ) -> tuple[list[cas.MX], list[float], list[float]]:
        """
        Constraint to impose that the mean hand velocity is null at the end of the movement
        """
        ee_velo_mean = discretization.get_reference(
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
            discretization: DiscretizationAbstract,
            x_single: cas.MX,
    ) -> cas.MX:

        activations_mean = discretization.get_mean_states(
            self.model,
            x_single,
            squared=True,
            )[4: 4 + self.model.nb_muscles]
        efforts = cas.sum1(activations_mean)

        activations_variations = discretization.get_states_variance(
            self.model,
            x_single,
            squared=True,
            )[4: 4 + self.model.nb_muscles]
        variations = cas.sum1(activations_variations)

        return efforts + variations / 2
