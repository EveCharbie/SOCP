import casadi as cas
import numpy as np
import biorbd_casadi as biorbd

from .biorbd_model import BiorbdModel


class VertebrateArmModel(BiorbdModel):

    def __init__(self, nb_random: int):

        super().__init__(nb_random=nb_random, model_name="vertebrate_arm_model")

        self.nb_states = self.nb_q * 2
        self.nb_controls = self.nb_q

        if self.nb_random == 1:
            self.nb_noises = 0
        else:
            self.nb_noises = self.nb_q

    def forward_dynamics(
        self,
        q: cas.SX | cas.DM | np.ndarray,
        qdot: cas.SX | cas.DM | np.ndarray,
        tau: cas.SX | cas.DM | np.ndarray,
        motor_noise: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:
        return self.forward_dynamics_biorbd()(q, qdot, tau + motor_noise)

    def marker_position(
        self,
        q: cas.MX | cas.SX | cas.DM,
    ) -> cas.MX | cas.SX | cas.DM:
        marker_index = self.marker_index("end_effector")
        marker_pos = self.marker(marker_index)(q)[:2]
        return marker_pos

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
    def tau_indices(self):
        return range(0, self.nb_q)

    @property
    def control_indices(self):
        return {"tau": self.tau_indices}

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
        tau = u_simple[self.tau_indices]

        motor_noise = noise_simple[:]
        if motor_noise.shape[0] == 0:
            motor_noise = cas.DM.zeros(self.nb_q)

        # Dynamics
        d_q = x_simple[self.nb_q : 2 * self.nb_q]
        d_qdot = self.forward_dynamics(q, qdot, tau, motor_noise)

        dxdt = cas.vertcat(d_q, d_qdot)
        return dxdt

    def inverse_kinematics_target(self, target_pos: np.ndarray) -> np.ndarray:
        """
        Get the inverse kinematics function to reach the target position.
        """
        q = cas.SX.sym("q", self.nb_q) if self.use_sx else cas.MX.sym("q", self.nb_q)
        marker_index = self.marker_index("end_effector")
        marker_pos = self.marker(marker_index)(q)[:2]

        # Inverse kinematics
        nlp = {"f": cas.sum1((marker_pos - target_pos) ** 2), "x": q}
        solver = cas.nlpsol("solver", "ipopt", nlp)
        sol = solver()
        w_opt = sol["x"].full().flatten()

        # Test with forward kinematics that everything was OK
        marker_pos_opt = self.marker(marker_index)(w_opt)[:2]
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

    def sensory_output(
        self,
        q: cas.SX | cas.DM | np.ndarray,
        qdot: cas.SX | cas.DM | np.ndarray,
        sensory_noise: cas.SX | cas.DM | np.ndarray,
    ):
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
        return self.momentum_biorbd()(q, qdot)

    def non_conservative_forces(
        self,
        q: cas.SX | cas.DM | np.ndarray,
        qdot: cas.SX | cas.DM | np.ndarray,
        u: cas.SX | cas.DM | np.ndarray,
        noise: cas.SX | cas.DM | np.ndarray,
    ) -> cas.SX | cas.DM | np.ndarray:

        if noise.shape[0] == 0:
            noise = cas.DM.zeros(self.nb_q)

        return u + noise
