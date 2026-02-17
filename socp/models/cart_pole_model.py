"""
This file contains a simple rigid pendulum attached to a slider that can move sideways.
The degrees of freedom are the position of the slider and the angle of the pendulum.
The control is a force applied to the slider (the pendulum is unactuated).
"""

import casadi as cas
import numpy as np

from .model_abstract import ModelAbstract


class CartPoleModel(ModelAbstract):

    def __init__(self, nb_random: int):

        super().__init__(nb_random=nb_random)

        self.nb_q = 2
        self.nb_states = self.nb_q * 2
        self.nb_controls = 1
        self.nb_noises = 1

        self.mass_cart = 1
        self.mass_pole = 2
        self.pole_length = 0.15

    @property
    def name_dof(self):
        return ["cart_position", "pole_angle"]

    def mass_matrix(self, q: cas.SX | cas.DM | np.ndarray) -> cas.SX | cas.DM | np.ndarray:
        if isinstance(q, np.ndarray):
            mass_matrix = np.zeros((2, 2))
        else:
            mass_matrix = cas.SX.zeros(2, 2)
        mass_matrix[0, 0] = self.mass_cart + self.mass_pole
        mass_matrix[0, 1] = self.mass_pole * self.pole_length * cas.cos(q[1])
        mass_matrix[1, 0] = self.mass_pole * self.pole_length * cas.cos(q[1])
        mass_matrix[1, 1] = self.mass_pole * self.pole_length ** 2
        return mass_matrix

    def force_term(
            self,
            q: cas.SX | cas.DM | np.ndarray,
            qdot: cas.SX | cas.DM | np.ndarray,
            u: cas.SX | cas.DM | np.ndarray,
            motor_noise: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:

        gravity = cas.vertcat(0, -self.mass_pole * 9.81 * self.pole_length * cas.sin(q[1]))
        friction = cas.vertcat(-0.1 * qdot[0], 0)
        nl_effect = cas.vertcat(
            self.mass_pole * self.pole_length * cas.sin(q[1]) * qdot[1]**2,
            0,
        )
        controls = cas.vertcat(u[0], 0)
        motor_noises =  cas.vertcat(motor_noise[0], 0)

        force = gravity + controls + nl_effect + friction + motor_noises
        return force

    def forward_dynamics(
            self,
            q: cas.SX | cas.DM | np.ndarray,
            qdot: cas.SX | cas.DM | np.ndarray,
            u: cas.SX | cas.DM | np.ndarray,
            motor_noise: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:
        qddot = cas.inv(self.mass_matrix(q))  @ self.force_term(q, qdot, u, motor_noise)
        return qddot

    @property
    def q_indices(self):
        return range(0, self.nb_q)

    @property
    def qdot_indices(self):
        return range(self.nb_q, 2 * self.nb_q)

    @property
    def state_indices(self):
        return {"q": self.q_indices, "qdot": self.qdot_indices}

    @property
    def u_indices(self):
        return range(0, 1)

    @property
    def control_indices(self):
        return {"u": self.u_indices}

    @property
    def motor_noise_indices(self):
        return range(0, 1)

    @property
    def noise_indices(self):
        return [self.motor_noise_indices]

    def dynamics(
        self,
        x_simple: cas.SX | cas.DM | np.ndarray,
        u_simple: cas.SX | cas.DM | np.ndarray,
        ref: list[cas.SX | cas.DM | np.ndarray],
        noise_simple: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:

        # Collect variables
        q = x_simple[: self.nb_q]
        qdot = x_simple[self.nb_q : 2 * self.nb_q]
        u = u_simple[:]
        motor_noise = noise_simple[:]

        # Dynamics
        d_q = x_simple[self.nb_q : 2 * self.nb_q]
        d_qdot = self.forward_dynamics(q, qdot, u, motor_noise)

        dxdt = cas.vertcat(d_q, d_qdot)
        return dxdt

    def sensory_output(
            self,
            q: cas.SX | cas.DM | np.ndarray,
            qdot: cas.SX | cas.DM | np.ndarray,
            sensory_noise: cas.SX | cas.DM | np.ndarray):
        return []

    def lagrangian(
        self,
        q: cas.SX | cas.DM | np.ndarray,
        qdot: cas.SX | cas.DM | np.ndarray,
        u: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:
        kinetic_energy = 0.5 * qdot.T @ self.mass_matrix(q) @ qdot
        potential_energy = -self.mass_pole * 9.81 * self.pole_length * cas.cos(q[1])
        return kinetic_energy - potential_energy

    def momentum(
        self,
        q: cas.SX | cas.DM | np.ndarray,
        qdot: cas.SX | cas.DM | np.ndarray,
        u: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:
        p = self.mass_matrix(q) @ qdot
        return p

    def non_conservative_forces(
        self,
        q: cas.SX | cas.DM | np.ndarray,
        qdot: cas.SX | cas.DM | np.ndarray,
        u: cas.SX | cas.DM | np.ndarray,
        noise: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:

        motor_noise = noise[:]
        motor_noises = cas.vertcat(motor_noise[0], 0)

        friction = cas.vertcat(-0.1 * qdot[0], 0)

        controls = cas.vertcat(u[0], 0)

        return controls + friction + motor_noises
