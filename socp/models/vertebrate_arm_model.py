import casadi as cas
import numpy as np
import biorbd_casadi as biorbd

from .biorbd_model import BiorbdModel


class VertebrateArmModel(BiorbdModel):

    def __init__(self, nb_random: int):

        super().__init__(nb_random=nb_random, model_name="vertebrate_arm_model")

        self.nb_references = 4

        self.nb_states = self.nb_q * 2
        if self.nb_random == 1:
            self.nb_k = 0
        else:
            self.nb_k = self.nb_q * self.nb_references
        self.nb_controls = self.nb_q

        if self.nb_random == 1:
            self.nb_noises = 0
        else:
            self.nb_noises = self.nb_q + self.nb_references

        self.matrix_shape_k = (self.nb_q, self.nb_references)
        self.matrix_shape_cov = (self.nb_states, self.nb_states)
        self.matrix_shape_m = (self.nb_states, self.nb_states)

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
    def k_indices(self):
        if self.nb_random == 1:
            return []
        else:
            return range(self.nb_q, self.nb_q + self.nb_k)

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
            return range(0, self.nb_q)

    @property
    def sensory_noise_indices(self):
        if self.nb_random == 1:
            return []
        else:
            return range(self.nb_q, self.nb_q + self.nb_references)

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
        q = x_simple[: self.nb_q]
        qdot = x_simple[self.nb_q : 2 * self.nb_q]
        tau = u_simple[self.tau_indices]
        if self.nb_random > 1:
            k = u_simple[self.k_indices]
            k_matrix = self.reshape_vector_to_matrix(k, self.matrix_shape_k)
            sensory_noise = noise_simple[self.sensory_noise_indices]

        motor_noise = noise_simple[self.motor_noise_indices]
        if motor_noise.shape[0] == 0:
            motor_noise = cas.DM.zeros(self.nb_q)

        if self.nb_random == 1:
            tau_fb = cas.DM.zeros(self.nb_q)
        else:
            tau_fb = k_matrix @ (self.sensory_output(q, qdot, sensory_noise) - ref)

        # Dynamics
        d_q = x_simple[self.qdot_indices]
        d_qdot = self.forward_dynamics(q, qdot, tau + tau_fb, motor_noise)

        dxdt = cas.vertcat(d_q, d_qdot)
        return dxdt

    def sensory_output(self, q: cas.MX | cas.SX, qdot: cas.MX | cas.SX, sensory_noise: cas.MX | cas.SX):
        """
        Sensory feedback: hand position and velocity
        """
        marker_index = self.marker_index("end_effector")
        hand_position = self.marker(marker_index)(q)[:2]
        hand_velocity = self.marker_velocity(marker_index)(q, qdot)[:2]
        noised_fb = cas.vertcat(hand_position, hand_velocity) + sensory_noise
        return noised_fb

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
        x: cas.MX | cas.SX,
        u: cas.SX | cas.DM | np.ndarray,
        noise: cas.SX | cas.DM | np.ndarray,
        ref: cas.MX | cas.SX,
    ) -> cas.SX | cas.DM | np.ndarray:

        if self.nb_random > 1:
            motor_noise = noise[self.motor_noise_indices]
            sensory_noise = noise[self.sensory_noise_indices]
        else:
            motor_noise = cas.DM.zeros(self.nb_q)

        if self.nb_random == 1:
             tau_fb = cas.DM.zeros(self.nb_q)
        else:
            k = u[self.k_indices]
            k_matrix = self.reshape_vector_to_matrix(k, self.matrix_shape_k)
            tau_fb = k_matrix @ (self.sensory_output(q, qdot, sensory_noise) - ref)

        tau_control = u[self.tau_indices]

        return tau_control + tau_fb + motor_noise
