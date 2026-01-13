"""
This file contains the model used in the article.
"""

import casadi as cas

from .model_abstract import ModelAbstract


class MassPointModel(ModelAbstract):
    """
    This allows to generate the same model as in the paper.
    """

    def __init__(self, nb_random: int):

        super().__init__(nb_random=nb_random)

        self.nb_q = 2
        self.nb_states = self.nb_q * 2
        self.nb_controls = 2
        self.nb_noises = 2

        self.kapa = 10
        self.c = 1
        self.beta = 1

        self.super_ellipse_center_x = [0, 1]
        self.super_ellipse_center_y = [0, 0.5]
        self.super_ellipse_a = [1, 0.5]
        self.super_ellipse_b = [1, 2]
        self.super_ellipse_n = [4, 4]

    @property
    def name_dof(self):
        return ["X", "Y"]

    def forward_dynamics(self, q: cas.SX, qdot: cas.SX, u: cas.SX, motor_noise: cas.SX) -> cas.SX:
        qddot = (
            -self.kapa * (q - u) - self.beta * qdot * cas.sqrt(qdot[0] ** 2 + qdot[1] ** 2 + self.c**2) + motor_noise
        )
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
