import casadi as cas

from bioptim import (
    DynamicsEvaluation,
    NonLinearProgram,
    DynamicsFunctions,
    StochasticBioModel,
)


def deterministic_forward_dynamics(
    time: cas.MX,
    states: cas.MX,
    controls: cas.MX,
    parameters: cas.MX,
    algebraic_states: cas.MX,
    numerical_timeseries: cas.MX,
    nlp: NonLinearProgram,
    force_field_magnitude,
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

    tau_force_field = nlp.model.force_field(q, force_field_magnitude)

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


def stochastic_forward_dynamics(
    time: cas.MX,
    states: cas.MX,
    controls: cas.MX,
    parameters: cas.MX,
    algebraic_states: cas.MX,
    numerical_timeseries: cas.MX,
    nlp: NonLinearProgram,
    force_field_magnitude,
) -> DynamicsEvaluation:
    """
    SOCP dynamics
    """

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

        tau_force_field = nlp.model.force_field(q_this_time, force_field_magnitude)

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
