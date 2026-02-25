
import casadi as cas
import numpy as np
import biorbd_casadi as biorbd

from .biorbd_model import BiorbdModel


class VertebrateModel(BiorbdModel):

    def __init__(self, nb_random: int):

        super().__init__(nb_random=nb_random, model_name="vertebrate_model")

        self.nb_states = self.nb_q * 2
        self.nb_controls = self.nb_q
        self.nb_noises = self.nb_q

    def forward_dynamics(
            self,
            q: cas.SX | cas.DM | np.ndarray,
            qdot: cas.SX | cas.DM | np.ndarray,
            u: cas.SX | cas.DM | np.ndarray,
            motor_noise: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:
        return self.forward_dynamics_biorbd()(q, qdot, u + motor_noise)

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
        return {"tau": self.u_indices}

    @property
    def motor_noise_indices(self):
        return range(0, self.nb_q)

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
        return self.lagrangian_biorbd()(q, qdot)

    def momentum(
        self,
        q: cas.SX | cas.DM | np.ndarray,
        qdot: cas.SX | cas.DM | np.ndarray,
        tau: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:
        return self.momentum_biorbd()(q, qdot, tau)

    def non_conservative_forces(
        self,
        q: cas.SX | cas.DM | np.ndarray,
        qdot: cas.SX | cas.DM | np.ndarray,
        u: cas.SX | cas.DM | np.ndarray,
        noise: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:
        return u + noise
