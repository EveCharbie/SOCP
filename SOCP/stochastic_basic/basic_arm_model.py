"""
This file contains the model used in the article.
"""
import numpy as np
import casadi as cas

from bioptim import (
    DynamicsFunctions,
    StochasticBioModel,
    NonLinearProgram,
    DynamicsEvaluation,
    Controls,
    ConfigureVariables,
    StateDynamics,
)

from ..models.arm_model import ArmModel


class BasicArmModel(StateDynamics, ArmModel):
    def __init__(
        self,
        sensory_noise_magnitude: np.ndarray | cas.DM,
        motor_noise_magnitude: np.ndarray | cas.DM,
        sensory_reference: callable,
        compute_torques_from_noise_and_feedback: callable = None,
        force_field_magnitude: float = 0,
        nb_random: int = 1,
    ):
        StateDynamics.__init__(self)
        ArmModel.__init__(
            self,
            sensory_noise_magnitude=sensory_noise_magnitude,
            motor_noise_magnitude=motor_noise_magnitude,
            sensory_reference=sensory_reference,
            compute_torques_from_noise_and_feedback=compute_torques_from_noise_and_feedback,
            force_field_magnitude=force_field_magnitude,
            nb_random=nb_random,
        )

        # Variable configurations
        self.state_configuration = [
            self.configure_stochastic_q,
            self.configure_stochastic_qdot,
            self.configure_stochastic_muscle_activations,
        ]
        self.control_configuration = [
            Controls.TAU,
            Controls.MUSCLE_EXCITATION,
            self.configure_stochastic_k,
            self.configure_stochastic_ref,
        ]
        self.with_residual_torque = True
        self.contact_types = []

    def configure_stochastic_q(
        self,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_algebraic_states=False,
    ):
        name_q = []
        for j in range(self.nb_random):
            for i in range(self.nb_q):
                name_q += [f"{self.name_dof[i]}_{j}"]
        ConfigureVariables.configure_new_variable(
            "q",
            name_q,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_algebraic_states=False,
        )

    def configure_stochastic_qdot(
        self,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_algebraic_states=False,
    ):
        name_qdot = []
        for j in range(self.nb_random):
            for i in range(self.nb_q):
                name_qdot += [f"{nlp.model.name_dof[i]}_{j}"]
        ConfigureVariables.configure_new_variable(
            "qdot",
            name_qdot,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_algebraic_states=False,
        )

    def configure_stochastic_muscle_activations(
        self,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_algebraic_states=False,
    ):

        name_muscles = []
        for j in range(self.nb_random):
            for i in range(self.nb_muscles):
                name_muscles += [f"{self.muscle_names[i]}_{j}"]
        ConfigureVariables.configure_new_variable(
            "muscle_activations",
            name_muscles,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
            as_algebraic_states=False,
        )

    def configure_stochastic_k(
        self,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_algebraic_states=False,
    ):

        name_k = []
        control_names = [f"control_{i}" for i in range(self.n_noised_controls)]
        ref_names = [f"feedback_{i}" for i in range(self.n_references)]
        for name_1 in control_names:
            for name_2 in ref_names:
                name_k += [name_1 + "_&_" + name_2]
        ConfigureVariables.configure_new_variable(
            "k",
            name_k,
            ocp,
            nlp,
            as_states=False,
            as_controls=True,
            as_algebraic_states=False,
        )

    def configure_stochastic_ref(
        self,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_algebraic_states=False,
    ):

        ref_names = [f"feedback_{i}" for i in range(self.n_references)]
        ConfigureVariables.configure_new_variable(
            "ref",
            ref_names,
            ocp,
            nlp,
            as_states=False,
            as_controls=True,
            as_algebraic_states=False,
        )

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

        nb_q = nlp.model.nb_q
        nb_muscles = nlp.model.nb_muscles
        nb_random = nlp.model.nb_random
        nb_states = 2 * nb_q + nb_muscles

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        mus_activations = DynamicsFunctions.get(nlp.states["muscle_activations"], states)
        mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
        tau_residuals = DynamicsFunctions.get(nlp.controls["tau"], controls)
        k = DynamicsFunctions.get(nlp.controls["k"], controls)
        k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)
        ref = DynamicsFunctions.get(nlp.controls["ref"], controls)

        motor_noise = None
        sensory_noise = None
        for i in range(nb_random):
            if motor_noise == None:
                motor_noise = DynamicsFunctions.get(
                    nlp.numerical_timeseries[f"motor_noise_numerical_{i}"], numerical_timeseries
                )
                sensory_noise = DynamicsFunctions.get(
                    nlp.numerical_timeseries[f"sensory_noise_numerical_{i}"], numerical_timeseries
                )
            else:
                motor_noise = cas.horzcat(
                    motor_noise,
                    DynamicsFunctions.get(nlp.numerical_timeseries[f"motor_noise_numerical_{i}"], numerical_timeseries),
                )
                sensory_noise = cas.horzcat(
                    sensory_noise,
                    DynamicsFunctions.get(nlp.numerical_timeseries[f"sensory_noise_numerical_{i}"], numerical_timeseries),
                )

        dxdt = cas.MX(nb_states * nb_random, 1)
        for i in range(nb_random):
            q_this_time = q[i * nb_q : (i + 1) * nb_q]
            qdot_this_time = qdot[i * nb_q : (i + 1) * nb_q]
            mus_activations_this_time = mus_activations[i * nb_muscles : (i + 1) * nb_muscles]

        hand_pos_velo = nlp.model.sensory_reference(
            time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp
        )

        mus_excitations_fb = mus_excitations + nlp.model.get_excitation_feedback(
            k_matrix, hand_pos_velo, ref, sensory_noise[:, i]
        )

        muscles_tau = nlp.model.get_muscle_torque(q_this_time, qdot_this_time, mus_activations_this_time)

        tau_force_field = nlp.model.force_field(q_this_time, self.force_field_magnitude)

        torques_computed = muscles_tau + motor_noise[:, i] + tau_force_field + tau_residuals

        dq_computed = qdot_this_time[:]
        dactivations_computed = (mus_excitations_fb - mus_activations_this_time) / nlp.model.tau_coef

        a1 = nlp.model.I1 + nlp.model.I2 + nlp.model.m2 * nlp.model.l1**2
        a2 = nlp.model.m2 * nlp.model.l1 * nlp.model.lc2
        a3 = nlp.model.I2

        theta_elbow = q_this_time[1]
        dtheta_shoulder = qdot_this_time[0]
        dtheta_elbow = qdot_this_time[1]

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

        dqdot_computed = cas.inv(mass_matrix) @ (torques_computed - nleffects - friction @ qdot_this_time)

        dxdt[i * nb_states : (i + 1) * nb_states] = cas.vertcat(dq_computed, dqdot_computed, dactivations_computed)

        return DynamicsEvaluation(dxdt=dxdt, defects=None)
