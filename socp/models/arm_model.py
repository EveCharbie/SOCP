"""
This file contains the model used in the article.
"""

from typing import Callable
import casadi as cas
import numpy as np

from .model_abstract import ModelAbstract
from ..transcriptions.discretization_abstract import DiscretizationAbstract


class ArmModel(ModelAbstract):
    """
    This allows to generate the same model as in the paper.
    """

    def __init__(self, nb_random: int):

        super().__init__(nb_random=nb_random)

        self.force_field_magnitude = 0  # TODO: for now

        self.nb_q = 2
        self.nb_muscles = 6
        self.nb_states = self.nb_q * 2 + self.nb_muscles
        self.nb_noised_controls = self.nb_muscles
        self.nb_references = 4
        self.nb_k = self.nb_references * self.nb_muscles
        self.nb_controls = self.nb_muscles + self.nb_k
        self.nb_noised_states = self.nb_q * 2 + self.nb_muscles
        self.nb_noises = self.nb_q + self.nb_references  # motor + sensory

        self.matrix_shape_k = (self.nb_noised_controls, self.nb_references)
        self.matrix_shape_c = (self.nb_noised_states, self.nb_noises)
        self.matrix_shape_a = (self.nb_noised_states, self.nb_noised_states)
        self.matrix_shape_cov = (self.nb_noised_states, self.nb_noised_states)
        self.matrix_shape_cov_cholesky = (self.nb_noised_states, self.nb_noised_states)
        self.matrix_shape_m = (self.nb_noised_states, self.nb_noised_states)

        self.dM_coefficients = np.array(
            [
                [0, 0, 0.0100, 0.0300, -0.0110, 1.9000],
                [0, 0, 0.0100, -0.0190, 0, 0.0100],
                [0.0400, -0.0080, 1.9000, 0, 0, 0.0100],
                [-0.0420, 0, 0.0100, 0, 0, 0.0100],
                [0.0300, -0.0110, 1.9000, 0.0320, -0.0100, 1.9000],
                [-0.0390, 0, 0.0100, -0.0220, 0, 0.0100],
            ]
        )
        self.LMT_coefficients = np.array(
            [
                [1.1000, -5.206336195535022],
                [0.8000, -7.538918356984516],
                [1.2000, -3.938098437958920],
                [0.7000, -3.031522725559912],
                [1.1000, -2.522778221157014],
                [0.8500, -1.826368199415192],
            ]
        )
        self.vMtilde_max = np.ones((6, 1)) * 10
        self.Fiso = np.array([572.4000, 445.2000, 699.6000, 381.6000, 159.0000, 318.0000])
        self.Faparam = np.array(
            [
                0.814483478343008,
                1.055033428970575,
                0.162384573599574,
                0.063303448465465,
                0.433004984392647,
                0.716775413397760,
                -0.029947116970696,
                0.200356847296188,
            ]
        )
        self.Fvparam = np.array([-0.318323436899127, -8.149156043475250, -0.374121508647863, 0.885644059915004])
        self.Fpparam = np.array([-0.995172050006169, 53.598150033144236])
        self.muscleDampingCoefficient = np.ones((6, 1)) * 0.01

        self.a_shoulder = self.dM_coefficients[:, 0]
        self.b_shoulder = self.dM_coefficients[:, 1]
        self.c_shoulder = self.dM_coefficients[:, 2]
        self.a_elbow = self.dM_coefficients[:, 3]
        self.b_elbow = self.dM_coefficients[:, 4]
        self.c_elbow = self.dM_coefficients[:, 5]
        self.l_base = self.LMT_coefficients[:, 0]
        self.l_multiplier = self.LMT_coefficients[:, 1]

        # Active muscle force-length characteristic
        self.b11 = self.Faparam[0]
        self.b21 = self.Faparam[1]
        self.b31 = self.Faparam[2]
        self.b41 = self.Faparam[3]
        self.b12 = self.Faparam[4]
        self.b22 = self.Faparam[5]
        self.b32 = self.Faparam[6]
        self.b42 = self.Faparam[7]
        self.b13 = 0.1
        self.b23 = 1
        self.b33 = 0.5 * cas.sqrt(0.5)
        self.b43 = 0

        self.e0 = 0.6
        self.e1 = self.Fvparam[0]
        self.e2 = self.Fvparam[1]
        self.e3 = self.Fvparam[2]
        self.e4 = self.Fvparam[3]

        self.kpe = 4
        self.tau_coef = 0.1500

        self.l1 = 0.3
        self.l2 = 0.33
        self.m2 = 1
        self.lc2 = 0.16
        self.I1 = 0.025
        self.I2 = 0.045

        self.friction_coefficients = np.array([[0.05, 0.025], [0.025, 0.05]])

    @property
    def name_dof(self):
        return ["shoulder", "elbow"]

    @property
    def muscle_names(self):
        return [f"muscle_{i}" for i in range(self.nb_muscles)]

    def get_muscle_force(self, q, qdot):
        """
        Fa: active muscle force [N]
        Fp: passive muscle force [N]
        lMtilde: normalized fiber lenght [-]
        vMtilde: optimal fiber lenghts per second at which muscle is lengthening or shortening [-]
        FMltilde: force-length multiplier [-]
        FMvtilde: force-velocity multiplier [-]
        Fce: Active muscle force [N]
        Fpe: Passive elastic force [N]
        Fm: Passive viscous force [N]
        """
        theta_shoulder = q[0]
        theta_elbow = q[1]
        dtheta_shoulder = qdot[0]
        dtheta_elbow = qdot[1]

        # Normalized muscle fiber length (without tendon)
        l_full = (
            self.a_shoulder * theta_shoulder
            + self.b_shoulder * cas.sin(self.c_shoulder * theta_shoulder) / self.c_shoulder
            + self.a_elbow * theta_elbow
            + self.b_elbow * cas.sin(self.c_elbow * theta_elbow) / self.c_elbow
        )
        lMtilde = l_full * self.l_multiplier + self.l_base

        # Fiber velocity normalized by the optimal fiber length
        nCoeff = self.a_shoulder.shape[0]
        v_full = (
            self.a_shoulder * dtheta_shoulder
            + self.b_shoulder * cas.cos(self.c_shoulder * theta_shoulder) * cas.repmat(dtheta_shoulder, nCoeff, 1)
            + self.a_elbow * dtheta_elbow
            + self.b_elbow * cas.cos(self.c_elbow * theta_elbow) * cas.repmat(dtheta_elbow, nCoeff, 1)
        )
        vMtilde = self.l_multiplier * v_full

        vMtilde_normalizedToMaxVelocity = vMtilde / self.vMtilde_max

        num3 = lMtilde - self.b23
        den3 = self.b33 + self.b43 * lMtilde
        FMtilde3 = self.b13 * cas.exp(-0.5 * num3**2 / den3**2)

        num1 = lMtilde - self.b21
        den1 = self.b31 + self.b41 * lMtilde
        FMtilde1 = self.b11 * cas.exp(-0.5 * num1**2 / den1**2)

        num2 = lMtilde - self.b22
        den2 = self.b32 + self.b42 * lMtilde
        FMtilde2 = self.b12 * cas.exp(-0.5 * num2**2 / den2**2)

        FMltilde = FMtilde1 + FMtilde2 + FMtilde3

        FMvtilde = (
            self.e1
            * cas.log(
                (self.e2 @ vMtilde_normalizedToMaxVelocity + self.e3)
                + cas.sqrt((self.e2 @ vMtilde_normalizedToMaxVelocity + self.e3) ** 2 + 1)
            )
            + self.e4
        )

        # Active muscle force
        Fce = FMltilde * FMvtilde

        t5 = cas.exp(self.kpe * (lMtilde - 0.10e1) / self.e0)
        Fpe = ((t5 - 0.10e1) - self.Fpparam[0]) / self.Fpparam[1]

        # Muscle force + damping
        Fpv = self.muscleDampingCoefficient * vMtilde_normalizedToMaxVelocity
        Fa = self.Fiso * Fce
        Fp = self.Fiso * (Fpe + Fpv)

        return Fa, Fp

    def torque_force_relationship(self, Fm, q):
        theta_shoulder = q[0]
        theta_elbow = q[1]
        dM_matrix = cas.horzcat(
            self.a_shoulder + self.b_shoulder * cas.cos(self.c_shoulder @ theta_shoulder),
            self.a_elbow + self.b_elbow * cas.cos(self.c_elbow @ theta_elbow),
        ).T
        tau = dM_matrix @ Fm
        return tau

    def get_muscle_torque(self, q, qdot, mus_activations):
        Fa, Fp = self.get_muscle_force(q, qdot)
        Fm = mus_activations * Fa + Fp
        muscles_tau = self.torque_force_relationship(Fm, q)
        return muscles_tau

    def get_muscle_excitations(self, q, qdot, mus_excitations, ref, k, sensory_noise):
        sensory_input = self.end_effector_pos_velo(q, qdot)
        k_matrix = self.reshape_vector_to_matrix(k, self.matrix_shape_k)
        muscle_fb = k_matrix @ ((sensory_input - ref) + sensory_noise)
        return mus_excitations + muscle_fb

    def force_field(self, q, force_field_magnitude):
        F_forceField = force_field_magnitude * (self.l1 * cas.cos(q[0]) + self.l2 * cas.cos(q[0] + q[1]))
        hand_pos = type(q)(2, 1)
        hand_pos[0] = self.l2 * cas.sin(q[0] + q[1]) + self.l1 * cas.sin(q[0])
        hand_pos[1] = self.l2 * cas.sin(q[0] + q[1])
        tau_force_field = -F_forceField @ hand_pos
        return tau_force_field

    def get_total_noised_torque(
        self,
        q,
        qdot,
        mus_activations,
        motor_noise,
    ):

        tau_muscle = self.get_muscle_torque(q, qdot, mus_activations)

        tau_force_field = self.force_field(q, self.force_field_magnitude)

        tau_friction = -self.friction_coefficients @ qdot

        tau = tau_muscle + tau_force_field + tau_friction + motor_noise

        return tau

    def forward_dynamics(self, q: cas.SX, qdot: cas.SX, tau: cas.SX) -> cas.SX:

        theta_shoulder = q[0]
        theta_elbow = q[1]
        dtheta_shoulder = qdot[0]
        dtheta_elbow = qdot[1]

        a1 = self.I1 + self.I2 + self.m2 * self.l1**2
        a2 = self.m2 * self.l1 * self.lc2
        a3 = self.I2

        M = cas.SX.zeros(2, 2)
        M[0, 0] = a1 + 2 * a2 * cas.cos(theta_elbow)
        M[0, 1] = a3 + a2 * cas.cos(theta_elbow)
        M[1, 0] = a3 + a2 * cas.cos(theta_elbow)
        M[1, 1] = a3

        c = cas.SX.zeros(2, 1)
        c[0] = -dtheta_elbow * (2 * dtheta_shoulder + dtheta_elbow)
        c[1] = dtheta_shoulder**2
        nl_effects = a2 * cas.sin(theta_elbow) * c

        ddtheta = cas.inv(M) @ (tau - nl_effects)
        return ddtheta

    def end_effector_position(self, q):
        theta_shoulder = q[0]
        theta_elbow = q[1]
        ee_pos = cas.vertcat(
            cas.cos(theta_shoulder) * self.l1 + cas.cos(theta_shoulder + theta_elbow) * self.l2,
            cas.sin(theta_shoulder) * self.l1 + cas.sin(theta_shoulder + theta_elbow) * self.l2,
        )
        return ee_pos

    def end_effector_velocity(self, q, qdot):
        theta_shoulder = q[0]
        theta_elbow = q[1]
        a = theta_shoulder + theta_elbow
        dtheta_shoulder = qdot[0]
        dtheta_elbow = qdot[1]
        da = dtheta_shoulder + dtheta_elbow
        ee_vel = cas.vertcat(
            dtheta_shoulder * cas.sin(theta_shoulder) * self.l1 + da * cas.sin(a) * self.l2,
            -dtheta_shoulder * cas.cos(theta_shoulder) * self.l1 - da * cas.cos(a) * self.l2,
        )
        return ee_vel

    def end_effector_pos_velo(self, q: cas.SX, qdot: cas.SX) -> cas.SX:
        hand_pos = self.end_effector_position(q)
        hand_vel = self.end_effector_velocity(q, qdot)
        ee = cas.vertcat(hand_pos, hand_vel)
        return ee

    @property
    def q_indices(self):
        return range(0, self.nb_q)

    @property
    def qdot_indices(self):
        return range(self.nb_q, 2 * self.nb_q)

    @property
    def muscle_activation_indices(self):
        return range(2 * self.nb_q, 2 * self.nb_q + self.nb_muscles)

    @property
    def state_indices(self):
        return {"q": self.q_indices, "qdot": self.qdot_indices, "muscle_activation": self.muscle_activation_indices}

    @property
    def muscle_excitation_indices(self):
        return range(0, self.nb_muscles)

    @property
    def k_indices(self):
        offset = self.nb_muscles
        return range(offset, offset + self.nb_k)

    @property
    def control_indices(self):
        return {
            "muscle_excitation": self.muscle_excitation_indices,
            "k": self.k_indices,
        }

    @property
    def motor_noise_indices(self):
        return range(0, self.nb_q)

    @property
    def sensory_noise_indices(self):
        return range(self.nb_q, self.nb_q + self.nb_references)

    @property
    def noise_indices(self):
        return [self.motor_noise_indices, self.sensory_noise_indices]

    def sensory_output(self, q: cas.SX, qdot: cas.SX, sensory_noise: cas.SX):
        """
        Sensory feedback: hand position and velocity
        """
        ee_pos = self.end_effector_position(q)
        ee_vel = self.end_effector_velocity(q, qdot)
        return cas.vertcat(ee_pos, ee_vel) + sensory_noise

    def dynamics(
        self,
        x_simple,
        u_simple,
        ref,
        noise_simple,
    ) -> cas.SX:

        # Collect variables
        k = u_simple[self.k_indices]
        mus_excitations_original = u_simple[self.muscle_excitation_indices]
        q = x_simple[: self.nb_q]
        qdot = x_simple[self.nb_q : 2 * self.nb_q]
        mus_activation = x_simple[2 * self.nb_q : 2 * self.nb_q + self.nb_muscles]
        motor_noise = noise_simple[:2]
        sensory_noise = noise_simple[2:6]

        # Collect tau components
        muscle_excitations = self.get_muscle_excitations(
            q,
            qdot,
            mus_excitations_original,
            ref,
            k,
            sensory_noise,
        )

        torques_computed = self.get_total_noised_torque(
            q=q,
            qdot=qdot,
            mus_activations=mus_activation,
            motor_noise=motor_noise,
        )

        # Dynamics
        d_q = x_simple[self.nb_q : 2 * self.nb_q]
        d_qdot = self.forward_dynamics(q, qdot, torques_computed)
        d_activations = (muscle_excitations - mus_activation) / self.tau_coef

        dxdt = cas.vertcat(d_q, d_qdot, d_activations)
        return dxdt

    def inverse_kinematics_target(self, target_pos: np.ndarray) -> np.ndarray:
        """
        Get the inverse kinematics function to reach the target position.
        """
        q = cas.SX.sym("q", self.nb_q)
        marker_pos = self.end_effector_position(q)

        # Inverse kinematics
        nlp = {"f": cas.sum1((marker_pos - target_pos) ** 2), "x": q}
        solver = cas.nlpsol("solver", "ipopt", nlp)
        sol = solver()
        w_opt = sol["x"].full().flatten()

        # Test with forward kinematics that everything was OK
        marker_pos_opt = self.end_effector_position(w_opt)
        if not np.allclose(
            np.array(marker_pos_opt).reshape(
                2,
            ),
            np.array(target_pos).reshape(
                2,
            ),
            atol=1e-6,
        ):
            raise RuntimeError("Inverse kinematics did not converge to the target position.")

        return np.array(w_opt)
