"""
This file contains the model used in the article.
"""

from typing import Callable
import casadi as cas
import numpy as np

from .model_abstract import ModelAbstract


class ArmModel(ModelAbstract):
    """
    This allows to generate the same model as in the paper.
    """

    def __init__(self, n_random: int):

        self.n_random = n_random

        self.nb_q = 2
        self.nb_muscles = 6
        self.n_noised_controls = self.nb_muscles
        self.n_references = 4
        self.n_noised_states = self.nb_q * 2 + self.nb_muscles
        self.n_noises = self.nb_q + self.n_references  # motor + sensory
        self.matrix_shape_k = (self.n_noised_controls, self.n_references)
        self.matrix_shape_c = (self.n_noised_states, self.n_noises)
        self.matrix_shape_a = (self.n_noised_states, self.n_noised_states)
        self.matrix_shape_cov = (self.n_noised_states, self.n_noised_states)
        self.matrix_shape_cov_cholesky = (self.n_noised_states, self.n_noised_states)
        self.matrix_shape_m = (self.n_noised_states, self.n_noised_states)

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

    def serialize(self) -> tuple[Callable, dict]:
        return ArmModel, dict(
            dM_coefficients=self.dM_coefficients,
            LMT_coefficients=self.LMT_coefficients,
            vMtilde_max=self.vMtilde_max,
            Fiso=self.Fiso,
            Faparam=self.Faparam,
            Fvparam=self.Fvparam,
            Fpparam=self.Fpparam,
            muscleDampingCoefficient=self.muscleDampingCoefficient,
            friction_coefficients=self.friction_coefficients,
        )

    @staticmethod
    def reshape_matrix_to_vector(matrix: cas.MX | cas.DM) -> cas.MX | cas.DM:
        matrix_shape = matrix.shape
        vector = type(matrix)()
        for i_shape in range(matrix_shape[0]):
            for j_shape in range(matrix_shape[1]):
                vector = cas.vertcat(vector, matrix[i_shape, j_shape])
        return vector

    @staticmethod
    def reshape_vector_to_matrix(vector: cas.MX | cas.DM, matrix_shape: tuple[int, ...]) -> cas.MX | cas.DM:
        matrix = type(vector).zeros(matrix_shape)
        idx = 0
        for i_shape in range(matrix_shape[0]):
            for j_shape in range(matrix_shape[1]):
                matrix[i_shape, j_shape] = vector[idx]
                idx += 1
        return matrix

    def get_mean_q(self, x):
        q = x[: self.nb_q]
        for i_random in range(1, self.n_random):
            q = cas.cas.horzcat(q, x[i_random * self.nb_q : (i_random + 1) * self.nb_q])
        q_mean = cas.sum2(q) / self.n_random
        return q_mean

    def get_mean_qdot(self, x):
        qdot = x[self.q_offset : self.q_offset + self.nb_q]
        for i_random in range(1, self.n_random):
            qdot = cas.cas.horzcat(
                qdot, x[self.q_offset + i_random * self.nb_q : self.q_offset + (i_random + 1) * self.nb_q]
            )
        qdot_mean = cas.sum2(qdot) / self.n_random
        return qdot_mean

    @property
    def name_dof(self):
        return ["shoulder", "elbow"]

    @property
    def muscle_names(self):
        return [f"muscle_{i}" for i in range(self.nb_muscles)]

    @property
    def q_offset(self):
        return self.nb_q * self.n_random

    @property
    def matrix_shape_k_fb(self):
        return (self.nb_muscles, self.n_references)

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
            ref,
            k,
            tau,
            sensory_noise,
            motor_noise,
    ):
        tau_nominal = tau + self.get_muscle_torque(q, qdot, mus_activations)
        k_matrix = self.reshape_vector_to_matrix(k, self.matrix_shape_k)

        sensory_input = self.end_effector_pos_velo(q, qdot)
        tau_fb = k_matrix @ ((sensory_input - ref) + sensory_noise)

        tau_motor_noise = motor_noise

        tau = tau_nominal + tau_fb + tau_motor_noise

        return tau


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

    def end_effector_pos_velo(self, q, qdot) -> cas.MX:
        hand_pos = self.end_effector_position(q)
        hand_vel = self.end_effector_velocity(q, qdot)
        ee = cas.vertcat(hand_pos, hand_vel)
        return ee

    @property
    def q_all_indices(self):
        return range(0, self.nb_q * self.n_random)

    @property
    def qdot_all_indices(self):
        return range(self.nb_q * self.n_random, 2 * self.nb_q * self.n_random)

    @property
    def muscle_indices(self):
        return range(0, self.nb_muscles)

    @property
    def n_k_fb(self):
        return self.n_references * self.nb_muscles

    @property
    def k_fb_indices(self):
        offset = self.nb_muscles
        return range(offset, offset + self.n_k_fb)

    @property
    def tau_indices(self):
        offset = self.nb_muscles + self.n_k_fb
        return range(offset, offset + self.nb_q)

    @property
    def motor_noise_indices(self):
        return range(0, self.nb_q)

    @property
    def sensory_noise_indices(self):
        return range(self.nb_q, self.nb_q + self.n_references)

    def q_indices_this_random(self, i_random):
        return range(i_random * self.nb_q, (i_random + 1) * self.nb_q)

    def qdot_indices_this_random(self, i_random):
        return range(self.q_offset + i_random * self.nb_q, self.q_offset + (i_random + 1) * self.nb_q)

    def sensory_output(self, q, qdot, sensory_noise):
        """
        Sensory feedback: hand position and velocity
        """
        ee_pos = self.end_effector_position(q)
        ee_vel = self.end_effector_velocity(q, qdot)
        return cas.vertcat(ee_pos, ee_vel) + sensory_noise

    def sensory_reference(self, x_single):
        """
        Compute the mean sensory output from all random trials
        """
        ref_fb = np.zeros((4, ))
        for i_random in range(self.n_random):
            q_this_time = x_single[self.q_indices_this_random(i_random)]
            qdot_this_time = x_single[self.qdot_indices_this_random(i_random)]
            ref_fb += self.sensory_output(q_this_time, qdot_this_time, cas.DM.zeros(self.n_references))
        ref_fb /= self.n_random
        return ref_fb

    def collect_tau(self, q, qdot, muscle_activations, k_fb, ref_fb, tau, motor_noise_this_time, sensory_noise_this_time):
        """
        Collect all tau components

        Note: that the following line compromises convergence :(
        `muscles_tau = self.get_muscle_torque(q, qdot, muscle_activations + motor_noise_this_time)`
        So we add the noise on tau instead
        """
        muscle_fb = k_fb @ (self.sensory_output(q, qdot, sensory_noise_this_time) - ref_fb)
        muscles_tau = self.get_muscle_torque(q, qdot, muscle_activations + muscle_fb)
        tau_force_field = self.force_field(q, self.force_field_magnitude)
        tau_friction = -self.friction_coefficients @ qdot
        torques_computed = muscles_tau + tau_force_field + tau_friction + tau + motor_noise_this_time
        return torques_computed

    def dynamics(
        self,
        x_single,
        u_single,
        noise_single,
    ) -> cas.Function:
        """
        Variables:
        - q (2 x n_random, n_shooting + 1)
        - qdot (2 x n_random, n_shooting + 1)
        - muscle (6, n_shooting)
        - k_fb (4 x 6, n_shooting)
        - tau (2, n_shooting)
        Noises:
        - motor_noise (2 x n_random, n_shooting)
        - sensory_noise (4 x n_random, n_shooting)
        """

        # Collect variables
        muscle_activations = u_single[self.muscle_indices]
        k_fb = self.reshape_vector_to_matrix(
            u_single[self.k_fb_indices], self.matrix_shape_k_fb
        )
        tau = u_single[self.tau_indices]
        qddot = cas.MX.zeros(self.nb_q * self.n_random)
        noise_offset = 0

        ref_fb = self.sensory_reference(x_single)

        for i_random in range(self.n_random):
            q_this_time = x_single[self.q_indices_this_random(i_random)]
            qdot_this_time = x_single[self.qdot_indices_this_random(i_random)]
            motor_noise_this_time = noise_single[noise_offset : noise_offset + self.nb_q]
            noise_offset += self.nb_q
            sensory_noise_this_time = noise_single[
                noise_offset : noise_offset + self.n_references
            ]
            noise_offset += self.n_references

            # Collect tau components
            torques_computed = self.collect_tau(
                q_this_time,
                qdot_this_time,
                muscle_activations,
                k_fb,
                ref_fb,
                tau,
                motor_noise_this_time,
                sensory_noise_this_time
            )

            # Dynamics
            qddot[self.q_indices_this_random(i_random)] = self.forward_dynamics(
                q_this_time, qdot_this_time, torques_computed
            )

        dxdt = cas.vertcat(x_single[self.q_offset :], qddot)
        return dxdt
