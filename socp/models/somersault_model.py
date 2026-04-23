"""
This file contains the model used in the article.
"""

import casadi as cas
import numpy as np

from .biorbd_model import BiorbdModel


class SomersaultModel(BiorbdModel):
    """
    This allows to generate the same model as in the paper.
    """

    def __init__(self, nb_random: int):

        super().__init__(nb_random=nb_random, model_name="somersault_model")

        self.nb_references = (self.nb_q - self.nb_root + 1) * 2
        self.nb_noised_controls = self.nb_q - self.nb_root

        self.nb_states = self.nb_q * 2
        if self.nb_random == 1:
            self.nb_k = 0
        else:
            self.nb_k = self.nb_noised_controls * self.nb_references
        self.nb_controls = (self.nb_q - self.nb_root) + self.nb_k
        if self.nb_random == 1:
            self.nb_noises = 0
        else:
            self.nb_noises = self.nb_noised_controls + self.nb_references

        self.matrix_shape_k = (self.nb_noised_controls, self.nb_references)
        self.matrix_shape_cov = (self.nb_states, self.nb_states)
        self.matrix_shape_m = (self.nb_states, self.nb_states)

        friction_coefficients = cas.DM.zeros(self.nb_q - self.nb_root, self.nb_q - self.nb_root)
        for i in range(self.nb_q - self.nb_root):
            friction_coefficients[i, i] = 0.1
        self.friction_coefficients = friction_coefficients

    def get_tau_full(
            self,
            tau: cas.SX | cas.MX | cas.DM | np.ndarray,
            motor_noise: cas.SX | cas.MX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.MX | cas.DM | np.ndarray:

        if isinstance(tau, np.ndarray):
            tau_full = np.zeros((self.nb_q,))
        elif isinstance(tau, cas.SX):
            tau_full = cas.SX.zeros(self.nb_q)
        elif isinstance(tau, cas.MX):
            tau_full = cas.MX.zeros(self.nb_q)
        elif isinstance(tau, cas.DM):
            tau_full = cas.DM.zeros(self.nb_q)
        else:
            raise TypeError(f"Type {type(tau)} not supported, please use DM, MX, SX or ndarray.")

        tau_full[self.nb_root:] = tau + motor_noise
        return tau_full

    def forward_dynamics(
        self,
        q: cas.SX | cas.DM | np.ndarray,
        qdot: cas.SX | cas.DM | np.ndarray,
        tau: cas.SX | cas.DM | np.ndarray,
        motor_noise: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:

        tau_full = self.get_tau_full(tau, motor_noise)
        return self.forward_dynamics_biorbd()(q, qdot, tau_full)

    @property
    def q_indices(self):
        return range(0, self.nb_q)

    @property
    def qdot_indices(self):
        return range(self.nb_q, 2 * self.nb_q)

    @property
    def state_indices(self):
        return {
            "q": self.q_indices,
            "qdot": self.qdot_indices,
        }

    @property
    def tau_indices(self):
        return range(0, self.nb_q - self.nb_root)

    @property
    def k_indices(self):
        if self.nb_random == 1:
            return []
        else:
            return range(self.nb_q - self.nb_root, self.nb_q - self.nb_root + self.nb_k)

    @property
    def control_indices(self):
        if self.nb_random == 1:
            return {
                "tau": self.tau_indices,
            }
        else:
            return {
                "tau": self.tau_indices,
                "k": self.k_indices,
            }

    @property
    def motor_noise_indices(self):
        if self.nb_random == 1:
            return []
        else:
            return range(0, self.nb_q - self.nb_root)

    @property
    def sensory_noise_indices(self):
        if self.nb_random == 1:
            return []
        else:
            return range(self.nb_q - self.nb_root, self.nb_q - self.nb_root + self.nb_references)

    @property
    def noise_indices(self):
        if self.nb_random == 1:
            return []
        else:
            return [self.motor_noise_indices, self.sensory_noise_indices]

    def dynamics(
        self,
        x_simple: cas.SX | cas.DM | np.ndarray,
        u_simple: cas.SX | cas.DM | np.ndarray,
        ref: list[cas.SX | cas.DM | np.ndarray],
        noise_simple: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:

        # Collect variables
        q = x_simple[self.q_indices]
        qdot = x_simple[self.qdot_indices]
        tau_control = u_simple[self.tau_indices]
        if self.nb_random > 1:
            k = u_simple[self.k_indices]
            k_matrix = self.reshape_vector_to_matrix(k, self.matrix_shape_k)
            sensory_noise = noise_simple[self.sensory_noise_indices]

        motor_noise = noise_simple[self.motor_noise_indices]
        if motor_noise.shape[0] == 0:
            motor_noise = cas.DM.zeros(self.nb_q - self.nb_root)

        tau_friction = -self.friction_coefficients @ qdot[self.nb_root :]
        if self.nb_random == 1:
            tau_fb = cas.DM.zeros(self.nb_q - self.nb_root)
        else:
            tau_fb = k_matrix @ (self.sensory_output(q, qdot, sensory_noise) - ref)
        u = tau_control + tau_friction + tau_fb

        # Dynamics
        d_q = x_simple[self.qdot_indices]
        d_qdot = self.forward_dynamics(q, qdot, u, motor_noise)

        dxdt = cas.vertcat(d_q, d_qdot)
        return dxdt

    def sensory_output(self, q: cas.MX | cas.SX, qdot: cas.MX | cas.SX, sensory_noise: cas.MX | cas.SX):
        """
        Sensory feedback: hand position and velocity
        """
        proprioceptive_feedback = cas.vertcat(q[self.nb_root:], qdot[self.nb_root:])
        pelvis_orientation = q[2]
        somersault_velocity = self.body_rotation_rate()(q, qdot)[0]
        return cas.vertcat(proprioceptive_feedback, pelvis_orientation, somersault_velocity) + sensory_noise

    def lagrangian(
        self,
        q: cas.MX | cas.SX,
        qdot: cas.MX | cas.SX,
        u: cas.MX | cas.SX,
    ) -> cas.MX | cas.SX:
        return self.lagrangian_biorbd()(q, qdot)

    def momentum(
        self,
        q: cas.MX | cas.SX,
        qdot: cas.MX | cas.SX,
        u: cas.MX | cas.SX,
    ) -> cas.MX | cas.SX:
        return self.momentum_biorbd()(q, qdot)

    def non_conservative_forces(
        self,
        q: cas.MX | cas.SX,
        qdot: cas.MX | cas.SX,
        x: cas.MX | cas.SX,
        u: cas.MX | cas.SX,
        noise: cas.MX | cas.SX,
        ref: cas.MX | cas.SX,
    ) -> cas.MX | cas.SX:

        if self.nb_random > 1:
            motor_noise = noise[self.motor_noise_indices]
            sensory_noise = noise[self.sensory_noise_indices]
        else:
            motor_noise = cas.DM.zeros(self.nb_q - self.nb_root)

        if self.nb_random == 1:
             tau_fb = cas.DM.zeros(self.nb_q - self.nb_root)
        else:
            k = u[self.k_indices]
            k_matrix = self.reshape_vector_to_matrix(k, self.matrix_shape_k)
            tau_fb = k_matrix @ (self.sensory_output(q, qdot, sensory_noise) - ref)

        tau_control = u[self.tau_indices]
        tau_friction = -self.friction_coefficients @ qdot

        return tau_control + tau_friction + tau_fb + motor_noise
