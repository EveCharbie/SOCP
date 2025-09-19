
import casadi as cas
import numpy as np

from bioptim import (
    States,
    Controls,
    StateDynamics,
    DynamicsEvaluation,
    NonLinearProgram,
    DynamicsFunctions,
)
from ..models.arm_model import ArmModel

    
def skip(*args, **kwargs):
    pass

class DeterministicArmModel(StateDynamics, ArmModel):
    def __init__(
        self,
        force_field_magnitude: float = 0,
    ):
        StateDynamics.__init__(self)
        ArmModel.__init__(
            self,
            sensory_noise_magnitude=np.zeros((4, 1)),
            motor_noise_magnitude=np.zeros((6, 1)),
            sensory_reference=lambda time, states, controls, parameters, algebraic_states, nlp: nlp.model.end_effector_pos_velo(
                DynamicsFunctions.get(nlp.states["q"], states),
                DynamicsFunctions.get(nlp.states["qdot"], states),
            ),
            compute_torques_from_noise_and_feedback=skip,
            force_field_magnitude=force_field_magnitude,
        )

        # Variable configurations
        self.state_configuration = [States.Q, States.QDOT, States.MUSCLE_ACTIVATION]
        self.control_configuration = [Controls.TAU, Controls.MUSCLE_EXCITATION]
        self.with_residual_torque = True
        self.contact_types = []

    def dynamics(
        self,
        time: cas.MX,
        states: cas.MX,
        controls: cas.MX,
        parameters: cas.MX,
        algebraic_states: cas.MX,
        numerical_timeseries: cas.MX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        """
        OCP dynamics
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        mus_activations = DynamicsFunctions.get(nlp.states["muscles"], states)
        mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
        tau_residuals = DynamicsFunctions.get(nlp.controls["tau"], controls)

        muscles_tau = nlp.model.get_muscle_torque(q, qdot, mus_activations)

        tau_force_field = nlp.model.force_field(q, self.force_field_magnitude)

        torques_computed = muscles_tau + tau_force_field + tau_residuals

        dq_computed = qdot
        dactivations_computed = (mus_excitations - mus_activations) / nlp.model.tau_coef

        a1 = nlp.model.I1 + nlp.model.I2 + nlp.model.m2 * nlp.model.l1**2
        a2 = nlp.model.m2 * nlp.model.l1 * nlp.model.lc2
        a3 = nlp.model.I2

        theta_elbow = q[1]
        dtheta_shoulder = qdot[0]
        dtheta_elbow = qdot[1]

        cx = type(theta_elbow)
        mass_matrix = cx(2, 2)
        mass_matrix[0, 0] = a1 + 2 * a2 * cas.cos(theta_elbow)
        mass_matrix[0, 1] = a3 + a2 * cas.cos(theta_elbow)
        mass_matrix[1, 0] = a3 + a2 * cas.cos(theta_elbow)
        mass_matrix[1, 1] = a3

        nleffects = cx(2, 1)
        nleffects[0] = a2 * cas.sin(theta_elbow) * (-dtheta_elbow * (2 * dtheta_shoulder + dtheta_elbow))
        nleffects[1] = a2 * cas.sin(theta_elbow) * dtheta_shoulder**2

        friction = nlp.model.friction_coefficients

        dqdot_computed = cas.inv(mass_matrix) @ (torques_computed - nleffects - friction @ qdot)
        dxdt = cas.vertcat(dq_computed, dqdot_computed, dactivations_computed)

        return DynamicsEvaluation(dxdt=dxdt, defects=None)


