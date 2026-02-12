"""
This file contains the model used in the article.
"""

import casadi as cas

from .model_abstract import ModelAbstract


class ParabolaModel(ModelAbstract):
    """
    This allows to generate the same model as in the paper.
    """

    def __init__(self, nb_random: int):

        super().__init__(nb_random=nb_random)

        self.nb_q = 1
        self.nb_states = 2
        self.nb_controls = 1
        self.nb_noises = 1

    @property
    def name_dof(self):
        return ["Q"]

    def forward_dynamics(self, q: cas.SX, qdot: cas.SX, u: cas.SX, motor_noise: cas.SX) -> cas.SX:
        qddot = 1.0 + u + motor_noise
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
        return range(0, self.nb_q)

    @property
    def control_indices(self):
        return {"u": self.u_indices}

    @property
    def motor_noise_indices(self):
        return range(0, self.nb_q)

    @property
    def noise_indices(self):
        return [self.motor_noise_indices]

    def dynamics(
        self,
        x_simple: cas.SX,
        u_simple: cas.SX,
        ref: list[cas.SX],
        noise_simple: cas.SX,
    ) -> cas.SX:

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

    def sensory_output(self, q: cas.SX, qdot: cas.SX, sensory_noise: cas.SX):
        return []

    def lagrangian(
        self,
        q: cas.SX,
        qdot: cas.SX,
        u: cas.SX,
    ) -> cas.SX:
        mass = 1
        kinetic_energy = 0.5 * mass * cas.dot(qdot, qdot)
        potential_energy = -q
        return kinetic_energy - potential_energy

    @staticmethod
    def momentum(
        q: cas.SX,
        qdot: cas.SX,
        u: cas.SX,
    ) -> cas.SX:
        mass = 1
        p = mass * qdot
        return p

    def non_conservative_forces(
        self,
        q: cas.SX,
        qdot: cas.SX,
        u: cas.SX,
        noise: cas.SX,
    ) -> cas.SX:
        # Since mass = 1, F = a
        motor_noise = noise[:]
        f = u + motor_noise
        return f
