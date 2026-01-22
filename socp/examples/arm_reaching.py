"""
This example implements a stochastic optimal control problem for an arm reaching task using muscle-driven dynamics.
The goal is to move the arm from an initial target to a final target while minimizing effort and variability, considering motor and sensory noise.
This example was taken from Van Wouwe et al. 2022.
"""

import numpy as np
import casadi as cas

from .example_abstract import ExampleAbstract
from ..constraints import Constraints
from ..models.arm_model import ArmModel
from ..models.model_abstract import ModelAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract
from ..transcriptions.noises_abstract import NoisesAbstract
from ..transcriptions.transcription_abstract import TranscriptionAbstract
from ..transcriptions.variables_abstract import VariablesAbstract


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

        collocation_points_initial_guesses = None

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
            discretization_method, dynamics_transcription, variables_vector, noises_vector.get_noise_single()
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

        return

    def get_specific_objectives(
        self,
        model: ModelAbstract,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
        noises_vector: NoisesAbstract,
    ) -> cas.SX:
        j: cas.SX = 0
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
        noise_single: cas.SX,
    ) -> tuple[list[cas.SX], list[float], list[float]]:

        xdot = dynamics_transcription.dynamics_func(
            variables_vector.get_states(0),
            variables_vector.get_controls(0),
            cas.DM.zeros(noise_single.shape),
        )
        xdot_mean = discretization_method.get_mean_states(
            self.model,
            xdot,
            squared=True,
        )
        g = [xdot_mean[self.model.qdot_indices]]
        lbg = [0, 0]
        ubg = [0, 0]
        return g, lbg, ubg

    def mean_start_on_target(
        self,
        discretization_method: DiscretizationAbstract,
        dynamics_transcription: TranscriptionAbstract,
        variables_vector: VariablesAbstract,
    ) -> tuple[list[cas.SX], list[float], list[float]]:
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
    ) -> tuple[list[cas.SX], list[float], list[float]]:
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
            self.model,
            variable_vector.get_states(i_node),
            squared=True,
        )[4 : 4 + self.model.nb_muscles]
        efforts = cas.sum1(activations_mean)

        activations_variations = discretization_method.get_mus_variance(
            self.model,
            variable_vector.get_states(i_node),
        )

        j = efforts + activations_variations / 2

        return j
