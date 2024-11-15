"""
Motor command : Voluntary + feedback + anticipatory feedback
Feedback: Joint position and velocity
Anticipatory feedback: Hand position (target miss)
"""

import pickle
import git
from datetime import date
import sys

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

from bioptim import (
    OptimalControlProgram,
    PhaseDynamics,
    InitialGuessList,
    ObjectiveFcn,
    Solver,
    ObjectiveList,
    NonLinearProgram,
    DynamicsEvaluation,
    DynamicsFunctions,
    ConfigureProblem,
    DynamicsList,
    BoundsList,
    InterpolationType,
    PenaltyController,
    Node,
    ConstraintList,
    SolutionMerge,
    OnlineOptim,
    ConstraintFcn,
    StochasticBioModel,
    OdeSolver,
)

from utils import ExampleType
sys.path.append("models/")
from leuven_arm_model import LeuvenArmModel


def constraint_final_marker_position(controller: PenaltyController, example_type) -> cas.MX:
    """
    Track the hand position.
    """

    nb_random = controller.model.nb_random
    nb_q = controller.model.nb_q

    q = controller.q.mx
    ee_pos = cas.MX.zeros(2, nb_random)
    for i in range(nb_random):
        q_this_time = q[i * nb_q: (i + 1) * nb_q]
        ee_pos[:, i] = controller.model.end_effector_position(q_this_time)
    ee_pos_mean = cas.sum2(ee_pos) / nb_random
    out = (ee_pos_mean if example_type == ExampleType.CIRCLE else ee_pos_mean[1])
    casadi_out = cas.Function("marker", [q], [out])(controller.states["q"].cx)
    return casadi_out


def get_forward_dynamics_func(nlp):

    nb_random = nlp.model.nb_random

    n_states = nlp.states.shape
    n_controls = nlp.controls.shape
    n_numerical_timeseries = nlp.numerical_timeseries.shape
    x = cas.MX.sym("x", n_states)
    u = cas.MX.sym("u", n_controls)
    numerical_timeseries = cas.MX.sym("numerical_timeseries", n_numerical_timeseries)

    dxdt = nlp.model.forward_dynamics([], x, u, [], [], numerical_timeseries, nlp).dxdt

    casadi_dynamics = cas.Function(
        "forward_dynamics",
        [x, u, numerical_timeseries],
        [cas.reshape(dxdt, (10, 1))],
        ["x", "u", "noises"],
        ["dxdt"],
    )
    return casadi_dynamics


def integrate_RK4(time_vector, dt, states, controls, dyn_fun, n_shooting=30, n_steps=5):
    n_q = 2
    tf = time_vector[-1]
    h = dt / n_steps
    x_integrated = cas.DM.zeros((n_q * 2, n_shooting + 1))
    x_integrated[:, 0] = states[:, 0]
    for i_shooting in range(n_shooting):
        x_this_time = x_integrated[:, i_shooting]
        u_this_time = controls[:, i_shooting]
        current_time = dt * i_shooting
        for i_step in range(n_steps):
            k1 = dyn_fun(current_time, tf, x_this_time, u_this_time)
            k2 = dyn_fun(current_time + h / 2, tf, x_this_time + h / 2 * k1, u_this_time)
            k3 = dyn_fun(current_time + h / 2, tf, x_this_time + h / 2 * k2, u_this_time)
            k4 = dyn_fun(current_time + h, tf, x_this_time + h * k3, u_this_time)
            x_this_time += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            current_time += h
        x_integrated[:, i_shooting + 1] = x_this_time
    return x_integrated


def continuity_constraint(controllers):

    current_idx = controllers[0].node_idx
    t_span = controllers[0].t_span.cx
    continuity = controllers[0].states.cx_end

    x0 = controllers[0].states.cx_start
    # Single shooting on the expected dynamics
    for i_node in range(len(controllers)):
        expected_states_end = integrate_RK4(time_vector, dt, states, controls, dyn_fun, n_shooting=30, n_steps=5)
        expected_ee_pos = controllers[0].model.ee_pos(expected_states_end[:controllers[0].model.n_q])
        hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
        expected_forward_dynamics()


    continuity -= controller.integrate(
        t_span=t_span,
        x0=controller.states.cx_start,
        u=u,
        p=controller.parameters.cx_start,
        a=controller.algebraic_states.cx_start,
        d=controller.numerical_timeseries.cx,
    )["xf"]



    penalty.phase = controller.phase_idx
    penalty.explicit_derivative = True
    penalty.multi_thread = True

    return constraint



def expected_forward_dynamics(
    time: cas.MX | cas.SX,
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    algebraic_states: cas.MX | cas.SX,
    numerical_timeseries: cas.MX | cas.SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:

    nb_q = nlp.model.nb_q
    nb_muscles = nlp.model.nb_muscles
    nb_random = nlp.model.nb_random

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    mus_activations = DynamicsFunctions.get(nlp.states["muscle_activations"], states)
    mus_excitations = DynamicsFunctions.get(nlp.controls["muscle_excitations"], controls)
    tau_residuals = DynamicsFunctions.get(nlp.controls["residual_tau"], controls)
    k = DynamicsFunctions.get(nlp.controls["k"], controls)
    k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)
    ref_fb = DynamicsFunctions.get(nlp.controls["ref_fb"], controls)

    q_this_time = q[: nb_q]
    qdot_this_time = qdot[: nb_q]
    mus_activations_this_time = mus_activations[: nb_muscles]
    for i in range(1, nb_random):
        q_this_time = cas.vertcat(q_this_time, q[i * nb_q: (i + 1) * nb_q])
        qdot_this_time = cas.vertcat(qdot_this_time, qdot[i * nb_q: (i + 1) * nb_q])
        mus_activations_this_time = cas.vertcat(mus_activations_this_time, mus_activations[i * nb_muscles: (i + 1) * nb_muscles])

    q_mean = cas.sum1(q_this_time) / nb_random
    qdot_mean = cas.sum1(qdot_this_time) / nb_random
    mus_activations_mean = cas.sum1(mus_activations_this_time) / nb_random
    mus_excitations_mean = mus_excitations[:]
    tau_mean = tau_residuals[:]

    sensory_info = nlp.model.sensory_reference(
        time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp
    )

    mus_excitations_fb = mus_excitations_mean
    mus_excitations_fb += nlp.model.get_excitation_feedback(k_matrix, sensory_info, ref_fb, cas.DM.zeros(nlp.model.sensory_noise_magnitude.shape[0], 1))

    muscles_tau = nlp.model.get_muscle_torque(q_mean, qdot_mean, mus_activations_mean)

    tau_force_field = nlp.model.force_field(q_mean, nlp.model.force_field_magnitude)

    torques_computed = muscles_tau + tau_force_field + tau_mean

    dq_computed = qdot_mean[:]
    dactivations_computed = (mus_excitations_fb - mus_activations_mean) / nlp.model.tau_coef

    a1 = nlp.model.I1 + nlp.model.I2 + nlp.model.m2 * nlp.model.l1**2
    a2 = nlp.model.m2 * nlp.model.l1 * nlp.model.lc2
    a3 = nlp.model.I2

    theta_elbow = q_mean[1]
    dtheta_shoulder = qdot_mean[0]
    dtheta_elbow = qdot_mean[1]

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

    dqdot_computed = cas.inv(mass_matrix) @ (torques_computed - nleffects - friction @ qdot_mean)

    dxdt = cas.vertcat(dq_computed, dqdot_computed, dactivations_computed)

    return dxdt


def stochastic_forward_dynamics(
    time: cas.MX | cas.SX,
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    algebraic_states: cas.MX | cas.SX,
    numerical_timeseries: cas.MX | cas.SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:

    nb_q = nlp.model.nb_q
    nb_muscles = nlp.model.nb_muscles
    nb_random = nlp.model.nb_random
    nb_states = 2 * nb_q + nb_muscles

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    mus_activations = DynamicsFunctions.get(nlp.states["muscle_activations"], states)
    mus_excitations = DynamicsFunctions.get(nlp.controls["muscle_excitations"], controls)
    tau_residuals = DynamicsFunctions.get(nlp.controls["residual_tau"], controls)
    k = DynamicsFunctions.get(nlp.controls["k"], controls)
    k_matrix = StochasticBioModel.reshape_to_matrix(k, nlp.model.matrix_shape_k)
    ref_fb = DynamicsFunctions.get(nlp.controls["ref_fb"], controls)
    hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
    ref_ff = hand_final_position[:]

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
        q_this_time = q[i * nb_q: (i + 1) * nb_q]
        qdot_this_time = qdot[i * nb_q: (i + 1) * nb_q]
        mus_activations_this_time = mus_activations[i * nb_muscles: (i + 1) * nb_muscles]
        mus_excitations_this_time = mus_excitations[:]
        tau_this_time = tau_residuals[:]

        hand_pos_velo = nlp.model.sensory_reference(
            time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp
        )

        mus_excitations_fb = mus_excitations_this_time[:]
        mus_excitations_fb += nlp.model.get_excitation_feedback(k_matrix, hand_pos_velo, ref_fb, sensory_noise[:, i])

        mus_ecxitations_ff += nlp.model.get_excitation_feedforward(k_matrix, hand_pos_velo, ref_fb, sensory_noise[:, i])

        muscles_tau = nlp.model.get_muscle_torque(q_this_time, qdot_this_time, mus_activations_this_time)

        tau_force_field = nlp.model.force_field(q_this_time, nlp.model.force_field_magnitude)

        torques_computed = muscles_tau + motor_noise[:, i] + tau_force_field + tau_this_time

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

        dxdt[i * nb_states: (i+1) * nb_states] = cas.vertcat(dq_computed, dqdot_computed, dactivations_computed)

    return DynamicsEvaluation(dxdt=dxdt, defects=None)


def configure_stochastic_optimal_control_problem(
    ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None
):
    nb_random = nlp.model.nb_random
    nb_q = nlp.model.nb_q
    nb_muscles = nlp.model.nb_muscles

    # States
    name_q = []
    for j in range(nb_random):
        for i in range(nb_q):
            name_q += [f"{nlp.model.name_dof[i]}_{j}"]
    ConfigureProblem.configure_new_variable(
        "q",
        name_q,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_states_dot=False,
    )

    ConfigureProblem.configure_new_variable(
        "qdot",
        name_q,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
        as_states_dot=True,
    )

    ConfigureProblem.configure_new_variable(
        "qddot",
        name_q,
        ocp,
        nlp,
        as_states=False,
        as_controls=False,
        as_states_dot=True,
    )

    name_muscles = []
    for j in range(nb_random):
        for i in range(nb_muscles):
            name_muscles += [f"{nlp.model.muscle_names[i]}_{j}"]
    ConfigureProblem.configure_new_variable(
        "muscle_activations",
        name_muscles,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
    )

    # Controls
    name_muscles = [nlp.model.muscle_names[i] for i in range(nb_muscles)]
    ConfigureProblem.configure_new_variable(
        "muscle_excitations",
        name_muscles,
        ocp,
        nlp,
        as_states=False,
        as_controls=True,
    )

    name_tau = [nlp.model.name_dof[i] for i in range(nb_q)]
    ConfigureProblem.configure_new_variable(
        "residual_tau",
        name_tau,
        ocp,
        nlp,
        as_states=False,
        as_controls=True,
        as_states_dot=False,
    )

    name_k = []
    control_names = [f"control_{i}" for i in range(nlp.model.n_noised_controls)]
    ref_names = [f"feedback_{i}" for i in range(nlp.model.n_references)]
    for name_1 in control_names:
        for name_2 in ref_names:
            name_k += [name_1 + "_&_" + name_2]
    ConfigureProblem.configure_new_variable(
        "k",
        name_k,
        ocp,
        nlp,
        as_states=False,
        as_controls=True,
        as_states_dot=False,
        as_algebraic_states=False,
    )
    ConfigureProblem.configure_new_variable(
        "ref_fb",
        ref_names,
        ocp,
        nlp,
        as_states=False,
        as_controls=True,
        as_states_dot=False,
        as_algebraic_states=False,
    )

    ConfigureProblem.configure_dynamics_function(ocp, nlp, stochastic_forward_dynamics)


def sensory_reference_func(
    q,
    qdot,
    model,
):
    """
    This functions returns the sensory reference for the feedback gains.
    """
    return cas.vertcat(q, qdot)


def sensory_reference_fb(
    time: cas.MX | cas.SX,
    states: cas.MX | cas.SX,
    controls: cas.MX | cas.SX,
    parameters: cas.MX | cas.SX,
    algebraic_states: cas.MX | cas.SX,
    numerical_timeseries: cas.MX | cas.SX,
    nlp: NonLinearProgram,
):
    """
    This functions returns the sensory reference for the feedback gains.
    """
    q = states[nlp.states["q"].index]
    qdot = states[nlp.states["qdot"].index]
    return sensory_reference_func(q, qdot, nlp.model)


def reach_target_consistantly(controller, example_type) -> cas.MX:

    nb_random = controller.model.nb_random
    nb_q = controller.model.nb_q

    q = controller.q.mx
    qdot = controller.qdot.mx
    ee_pos = cas.MX.zeros(2, nb_random)
    ee_vel = cas.MX.zeros(2, nb_random)
    for i in range(nb_random):
        q_this_time = q[i * nb_q: (i + 1) * nb_q]
        qdot_this_time = qdot[i * nb_q: (i + 1) * nb_q]
        ee_pos[:, i] = controller.model.end_effector_position(q_this_time)
        ee_vel[:, i] = controller.model.end_effector_velocity(q_this_time, qdot_this_time)
    ee_pos_mean = cas.sum2(ee_pos) / nb_random
    ee_vel_mean = cas.sum2(ee_vel) / nb_random

    if example_type == ExampleType.CIRCLE:
        deviations = cas.sum1((
            cas.sum2((ee_pos - ee_pos_mean) ** 2)
            + cas.sum2((ee_vel - ee_vel_mean) ** 2)
        ))
    else:
        deviations = (
                cas.sum2((ee_pos[1] - ee_pos_mean[1]) ** 2)
                + cas.sum2((ee_vel[1] - ee_vel_mean[1]) ** 2)
        )

    casadi_out = cas.Function("marker", [q, qdot], [deviations])(controller.states["q"].cx, controller.states["qdot"].cx)
    return casadi_out

def minimize_nominal_and_feedback_efforts(controller, sensory_noise_numerical) -> cas.MX:

    nb_q = controller.model.nb_q
    nb_muscles = controller.model.nb_muscles
    nb_random = controller.model.nb_random

    q = controller.q.mx
    qdot = controller.qdot.mx
    mus_activations = controller.states["muscle_activations"].mx
    mus_excitations = controller.controls["muscle_excitations"].mx
    tau_residuals = controller.controls["residual_tau"].mx
    k = controller.controls["k"].mx
    k_matrix = StochasticBioModel.reshape_to_matrix(k, controller.model.matrix_shape_k)
    ref_fb = controller.controls["ref_ff"].mx
    hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])
    ref_ff = hand_final_position[:]

    all_tau = 0
    for i in range(nb_random):
        q_this_time = q[i * nb_q: (i + 1) * nb_q]
        qdot_this_time = qdot[i * nb_q: (i + 1) * nb_q]
        mus_activations_this_time = mus_activations[i * nb_muscles: (i + 1) * nb_muscles]
        mus_excitations_this_time = mus_excitations[:]
        tau_this_time = tau_residuals[:]

        hand_pos_velo = controller.model.sensory_reference(
            [],
            controller.states.mx,
            controller.controls.mx,
            [],
            [],
            [],
            controller.get_nlp
        )

        mus_excitations_fb = mus_excitations_this_time[:]
        mus_excitations_fb += controller.model.get_excitation_feedback(k_matrix, hand_pos_velo, ref,
                                                                     sensory_noise_numerical[:, i])

        muscles_tau = controller.model.get_muscle_torque(q_this_time, qdot_this_time, mus_activations_this_time)

        tau_force_field = controller.model.force_field(q_this_time, controller.model.force_field_magnitude)

        a1 = controller.model.I1 + controller.model.I2 + controller.model.m2 * controller.model.l1 ** 2
        a2 = controller.model.m2 * controller.model.l1 * controller.model.lc2
        a3 = controller.model.I2

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
        nleffects[1] = a2 * cas.sin(theta_elbow) * dtheta_shoulder ** 2

        friction_tau = controller.model.friction_coefficients @ qdot_this_time

        all_tau += muscles_tau + tau_force_field + tau_this_time + friction_tau

    all_tau_cx = cas.Function(
        "all_tau",
        [controller.states.mx,
         controller.controls.mx],
        [all_tau],
    )(controller.states.cx, controller.controls.cx,)
    return all_tau_cx


def prepare_socp(
    final_time: float,
    n_shooting: int,
    hand_final_position: np.ndarray,
    motor_noise_magnitude: np.ndarray,
    sensory_noise_magnitude: np.ndarray,
    force_field_magnitude: float = 0,
    example_type=ExampleType.CIRCLE,
    nb_random: float = 30,
    seed: int = 0,
):

    bio_model = LeuvenArmModel(
        sensory_noise_magnitude=sensory_noise_magnitude,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_reference=sensory_reference,
    )
    n_q = bio_model.nb_q

    bio_model.force_field_magnitude = force_field_magnitude
    bio_model.nb_random = nb_random
    bio_model.n_noised_controls = bio_model.nb_muscles
    bio_model.n_references = 2 * n_q


    # Prepare the noises
    np.random.seed(seed)
    # the last node deos not need motor and sensory noise
    motor_noise_numerical = np.zeros((n_q, nb_random, n_shooting + 1))
    sensory_noise_numerical = np.zeros((2 * n_q, nb_random, n_shooting + 1))
    for i_random in range(nb_random):
        for i_shooting in range(n_shooting):
            motor_noise_numerical[:, i_random, i_shooting] = np.random.normal(
                loc=np.zeros(motor_noise_magnitude.shape[0]),
                scale=np.reshape(np.array(motor_noise_magnitude), (n_q,)),
                size=n_q,
            )
            sensory_noise_numerical[:, i_random, i_shooting] = np.random.normal(
                loc=np.zeros(sensory_noise_magnitude.shape[0]),
                scale=np.reshape(np.array(sensory_noise_magnitude), (2 * n_q,)),
                size=2 * n_q,
            )

    shoulder_pos_initial = 0.349065850398866
    shoulder_pos_final = 0.959931088596881
    elbow_pos_initial = 2.245867726451909  # Optimized in Tom's version
    elbow_pos_final = 1.159394851847144  # Optimized in Tom's version

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        minimize_nominal_and_feedback_efforts,
        sensory_noise_numerical=sensory_noise_numerical,
        custom_type=ObjectiveFcn.Lagrange,
        weight=1/2,
        quadratic=True,
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, node=Node.ALL, key="muscle_activations", weight=1 / 2, quadratic=True
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL_SHOOTING, key="residual_tau", weight=10, quadratic=True
    )
    # New
    objective_functions.add(
        reach_target_consistantly,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.END,
        weight=1/2,
        quadratic=True,
        example_type=example_type,
    )


    # Constraints
    constraints = ConstraintList()
    target = hand_final_position if example_type == ExampleType.CIRCLE else hand_final_position[1]
    constraints.add(constraint_final_marker_position, node=Node.END, target=target, example_type=example_type)
    # All tau_residual must be zero
    constraints.add(ConstraintFcn.TRACK_CONTROL, key="residual_tau", node=Node.ALL_SHOOTING)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        configure_stochastic_optimal_control_problem,
        dynamic_function=stochastic_forward_dynamics,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries={
            "motor_noise_numerical": motor_noise_numerical,
            "sensory_noise_numerical": sensory_noise_numerical,
        },
    )

    x_bounds = BoundsList()
    n_muscles = 6
    n_qdot = bio_model.nb_qdot

    # # initial variability
    # pose_at_first_node = np.array([shoulder_pos_initial, elbow_pos_initial])
    # initial_cov = np.eye(2 * n_q) * np.hstack((np.ones((n_q,)) * 1e-4, np.ones((n_q,)) * 1e-7))  # P
    # noised_states = np.random.multivariate_normal(
    #     np.hstack((pose_at_first_node, np.array([0, 0]))), initial_cov, nb_random
    # ).T

    q_min = np.ones((n_q*nb_random, 3)) * 0
    q_max = np.ones((n_q*nb_random, 3)) * np.pi
    for i_random in range(nb_random):
        q_min[i_random*n_q:(i_random+1)*n_q, 0] = np.array([shoulder_pos_initial, elbow_pos_initial])
        q_max[i_random*n_q:(i_random+1)*n_q, 0] = np.array([shoulder_pos_initial, elbow_pos_initial])
    x_bounds.add(
        "q", min_bound=q_min, max_bound=q_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    qdot_min = np.ones((n_q*nb_random, 3)) * -10*np.pi
    qdot_max = np.ones((n_q*nb_random, 3)) * 10*np.pi
    qdot_min[:, 0] = np.zeros((n_q*nb_random,))
    qdot_max[:, 0] = np.zeros((n_q*nb_random,))
    qdot_min[:, 2] = np.zeros((n_q*nb_random,))
    qdot_max[:, 2] = np.zeros((n_q*nb_random,))
    x_bounds.add(
        "qdot",
        min_bound=qdot_min,
        max_bound=qdot_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    muscle_min = np.ones((n_muscles*nb_random, 3)) * 0
    muscle_max = np.ones((n_muscles*nb_random, 3)) * 1
    muscle_min[:, 0] = np.zeros((n_muscles*nb_random,))
    muscle_max[:, 0] = np.zeros((n_muscles*nb_random,))
    x_bounds.add(
        "muscle_activations",
        min_bound=muscle_min,
        max_bound=muscle_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    u_bounds = BoundsList()
    controls_min = np.ones((n_muscles, 3)) * 0.001
    controls_max = np.ones((n_muscles, 3)) * 1
    controls_min[:, 0] = np.zeros((n_muscles,))
    controls_max[:, 0] = np.zeros((n_muscles,))
    u_bounds.add("muscle_excitations", min_bound=controls_min, max_bound=controls_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    tau_min = np.ones((n_q, 3)) * -10
    tau_max = np.ones((n_q, 3)) * 10
    tau_min[:, 0] = np.zeros((n_q,))
    tau_max[:, 0] = np.zeros((n_q,))
    u_bounds.add("residual_tau", min_bound=tau_min, max_bound=tau_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    # Initial guesses
    q_init = np.zeros((n_q*nb_random, n_shooting + 1))
    for i_random in range(nb_random):
        q_init[i_random*n_q:(i_random+1)*n_q, :] = np.linspace(
            [shoulder_pos_initial, elbow_pos_initial],
            [shoulder_pos_final, elbow_pos_final],
            n_shooting + 1,
        ).T

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=q_init, interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", initial_guess=np.zeros((n_q*nb_random, )), interpolation=InterpolationType.CONSTANT)
    x_init.add("muscle_activations", initial_guess=np.ones((n_muscles*nb_random, )) * 0.01, interpolation=InterpolationType.CONSTANT)

    u_init = InitialGuessList()
    u_init.add("muscle_excitations", initial_guess=np.ones((n_muscles, )) * 0.01, interpolation=InterpolationType.CONSTANT)
    u_init.add("residual_tau", initial_guess=np.ones((n_q, )) * 0.01, interpolation=InterpolationType.CONSTANT)


    # The stochastic variables will be put in the controls for simplicity
    n_ref = 4  # ref(2 ee_pos + 2 ee_vel)
    n_k = n_muscles * n_ref  # K(3x8)

    u_init.add("k", initial_guess=[0.01] * n_k, interpolation=InterpolationType.CONSTANT)

    u_bounds.add("k", min_bound=[-50] * n_k, max_bound=[50] * n_k, interpolation=InterpolationType.CONSTANT)

    ref_min = [-1000] * n_ref
    ref_max = [1000] * n_ref

    q_sym = cas.MX.sym("q", n_q, 1)
    qdot_sym = cas.MX.sym("qdot", n_q, 1)
    ref_fun = cas.Function(
        "ref_func", [q_sym, qdot_sym], [sensory_reference_func(q_sym, qdot_sym, bio_model)]
    )

    ref_init = np.zeros((n_ref, n_shooting + 1))
    for i in range(n_shooting):
        q_this_time = q_init[:n_q, i].T
        for j in range(1, nb_random):
            q_this_time = np.vstack((q_this_time, q_init[j * n_q : (j + 1) * n_q, i].T))
        q_mean = np.mean(q_this_time, axis=0)
        ref_init[:, i] = np.reshape(ref_fun(q_mean, np.zeros((n_q, ))), (n_ref,))

    u_bounds.add(
        "ref_fb",
        min_bound=ref_min,
        max_bound=ref_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )


    return OptimalControlProgram(
        bio_model=bio_model,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=OdeSolver.RK4(),
        n_threads=32,
    )


def main():
    # --- Options --- #
    plot_sol_flag = True
    vizualise_sol_flag = False

    hand_initial_position = np.array([0.0, 0.2742])  # Directly from Tom's version
    hand_final_position = np.array([9.359873986980460e-12, 0.527332023564034])  # Directly from Tom's version

    # --- Prepare the ocp --- #
    dt = 0.01
    final_time = 0.8
    n_shooting = int(final_time / dt)

    # --- Noise constants --- #
    motor_noise_std = 0.05
    wPq_std = 3e-4
    wPqdot_std = 0.0024

    motor_noise_magnitude = cas.DM(np.array([motor_noise_std**2 / dt, motor_noise_std**2 / dt]))
    wPq_magnitude = cas.DM(np.array([wPq_std**2 / dt, wPq_std**2 / dt]))
    wPqdot_magnitude = cas.DM(np.array([wPqdot_std**2 / dt, wPqdot_std**2 / dt]))
    sensory_noise_magnitude = cas.vertcat(wPq_magnitude, wPqdot_magnitude)

    # Solver parameters
    solver = Solver.IPOPT(show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma97")
    solver.set_bound_frac(1e-8)
    solver.set_bound_push(1e-8)
    solver.set_maximum_iterations(50000)
    solver.set_tol(1e-6)


    example_type = ExampleType.CIRCLE
    force_field_magnitude = 0
    ocp = prepare_socp(
        final_time=final_time,
        n_shooting=n_shooting,
        hand_final_position=hand_final_position,
        motor_noise_magnitude=motor_noise_magnitude,
        sensory_noise_magnitude=sensory_noise_magnitude,
        force_field_magnitude=force_field_magnitude,
        example_type=example_type,
        nb_random=30,
        seed=0,
    )
    # ocp.add_plot_ipopt_outputs()
    # ocp.add_plot_penalty()

    sol_ocp = ocp.solve(solver)
    # sol_ocp.print_cost()
    # sol_ocp.graphs(show_bounds=True)

    n_simulations = 100

    q_sol = sol_ocp.decision_states(to_merge=SolutionMerge.NODES)["q"]
    qdot_sol = sol_ocp.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
    activations_sol = sol_ocp.decision_states(to_merge=SolutionMerge.NODES)["muscle_activations"]
    excitations_sol = sol_ocp.decision_controls(to_merge=SolutionMerge.NODES)["muscle_excitations"]
    tau_sol = sol_ocp.decision_controls(to_merge=SolutionMerge.NODES)["residual_tau"]

    # --- Visualize the solution --- #
    if vizualise_sol_flag:
        import bioviz

        if example_type == ExampleType.CIRCLE:
            model_path="models/LeuvenArmModel_circle.bioMod"
        else:
            model_path = "models/LeuvenArmModel_bar.bioMod"
        b = bioviz.Viz(model_path=model_path)
        b.load_movement(q_sol)
        b.exec()


    # --- Plot the results --- #
    def RK4(x_prev, u, dt, motor_noise, forward_dyn_func, n_steps=5):
        h = dt / n_steps
        x_all = cas.DM.zeros((n_steps + 1, x_prev.shape[0]))
        x_all[0, :] = x_prev
        for i_step in range(n_steps):
            k1 = forward_dyn_func(
                x_prev,
                u,
                motor_noise,
            )
            k2 = forward_dyn_func(
                x_prev + h / 2 * k1,
                u,
                motor_noise,
            )
            k3 = forward_dyn_func(
                x_prev + h / 2 * k2,
                u,
                motor_noise,
            )
            k4 = forward_dyn_func(
                x_prev + h * k3,
                u,
                motor_noise,
            )

            x_all[i_step + 1, :] = x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x_prev = x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_all


    if plot_sol_flag:
        motor_noise_std = 0.05
        OCP_color = "#5DC962"

        model = LeuvenArmModel(
            sensory_noise_magnitude=np.zeros((1, 1)),
            motor_noise_magnitude=np.zeros((1, 1)),
            sensory_reference=lambda: np.zeros((1, 1)),
        )
        q_sym = cas.MX.sym("q_sym", 2, 1)
        qdot_sym = cas.MX.sym("qdot_sym", 2, 1)
        hand_pos_fcn = cas.Function("hand_pos", [q_sym], [model.end_effector_position(q_sym)])
        hand_vel_fcn = cas.Function("hand_vel", [q_sym, qdot_sym], [model.end_effector_velocity(q_sym, qdot_sym)])
        forward_dyn_func = get_forward_dynamics_func(ocp.nlp[0], force_field_magnitude)

        fig, axs = plt.subplots(3, 2)
        q_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        qdot_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        mus_activation_simulated = np.zeros((n_simulations, 6, n_shooting + 1))
        hand_pos_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        hand_vel_simulated = np.zeros((n_simulations, 2, n_shooting + 1))
        dt_actual = final_time / n_shooting
        for i_simulation in range(n_simulations):
            np.random.seed(i_simulation)
            motor_noise = np.random.normal(0, motor_noise_std, (2, n_shooting + 1))
            q_simulated[i_simulation, :, 0] = q_sol[:, 0]
            qdot_simulated[i_simulation, :, 0] = qdot_sol[:, 0]
            mus_activation_simulated[i_simulation, :, 0] = activations_sol[:, 0]
            for i_node in range(n_shooting):
                x_prev = cas.vertcat(
                    q_simulated[i_simulation, :, i_node],
                    qdot_simulated[i_simulation, :, i_node],
                    mus_activation_simulated[i_simulation, :, i_node],
                )
                hand_pos_simulated[i_simulation, :, i_node] = np.reshape(hand_pos_fcn(x_prev[:2])[:2], (2,))
                hand_vel_simulated[i_simulation, :, i_node] = np.reshape(
                    hand_vel_fcn(x_prev[:2], x_prev[2:4])[:2], (2,)
                )
                u = cas.vertcat(excitations_sol[:, i_node],
                                tau_sol[:, i_node])
                x_next = RK4(x_prev, u, dt_actual, motor_noise[:, i_node], forward_dyn_func, n_steps=5)
                q_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[-1, :2], (2,))
                qdot_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[-1, 2:4], (2,))
                mus_activation_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[-1, 4:], (6,))
            hand_pos_simulated[i_simulation, :, i_node + 1] = np.reshape(hand_pos_fcn(x_next[-1, :2])[:2], (2,))
            hand_vel_simulated[i_simulation, :, i_node + 1] = np.reshape(
                hand_vel_fcn(x_next[-1, :2], x_next[-1, 2:4])[:2], (2,)
            )
            axs[0, 0].plot(
                hand_pos_simulated[i_simulation, 0, :], hand_pos_simulated[i_simulation, 1, :], color=OCP_color, linewidth=0.5
            )
            axs[1, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_simulated[i_simulation, 0, :], color=OCP_color, linewidth=0.5)
            axs[2, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_simulated[i_simulation, 1, :], color=OCP_color, linewidth=0.5)
            axs[0, 1].plot(
                np.linspace(0, final_time, n_shooting + 1), np.linalg.norm(hand_vel_simulated[i_simulation, :, :], axis=0), color=OCP_color, linewidth=0.5
            )
            axs[1, 1].plot(np.linspace(0, final_time, n_shooting + 1), qdot_simulated[i_simulation, 0, :], color=OCP_color, linewidth=0.5)
            axs[2, 1].plot(np.linspace(0, final_time, n_shooting + 1), qdot_simulated[i_simulation, 1, :], color=OCP_color, linewidth=0.5)
        hand_pos_without_noise = np.zeros((2, n_shooting + 1))
        hand_vel_without_noise = np.zeros((2, n_shooting + 1))
        for i_node in range(n_shooting + 1):
            hand_pos_without_noise[:, i_node] = np.reshape(hand_pos_fcn(q_sol[:, i_node])[:2], (2,))
            hand_vel_without_noise[:, i_node] = np.reshape(hand_vel_fcn(q_sol[:, i_node], qdot_sol[:, i_node])[:2], (2,))

        axs[0, 0].plot(hand_pos_without_noise[0, :], hand_pos_without_noise[1, :], color="k")
        axs[0, 0].plot(hand_initial_position[0], hand_initial_position[1], color="tab:green", marker="o")
        axs[0, 0].plot(hand_final_position[0], hand_final_position[1], color="tab:red", marker="o")
        axs[0, 0].set_xlabel("X [m]")
        axs[0, 0].set_ylabel("Y [m]")
        axs[0, 0].set_title("Hand position simulated")
        axs[1, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_sol[0, :], color="k")
        axs[1, 0].set_xlabel("Time [s]")
        axs[1, 0].set_ylabel("Shoulder angle [rad]")
        axs[2, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_sol[1, :], color="k")
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 0].set_ylabel("Elbow angle [rad]")
        axs[0, 1].plot(np.linspace(0, final_time, n_shooting + 1), np.linalg.norm(hand_vel_without_noise, axis=0), color="k")
        axs[0, 1].set_xlabel("Time [s]")
        axs[0, 1].set_ylabel("Hand velocity [m/s]")
        axs[0, 1].set_title("Hand velocity simulated")
        axs[1, 1].plot(np.linspace(0, final_time, n_shooting + 1), qdot_sol[0, :], color="k")
        axs[1, 1].set_xlabel("Time [s]")
        axs[1, 1].set_ylabel("Shoulder velocity [rad/s]")
        axs[2, 1].plot(np.linspace(0, final_time, n_shooting + 1), qdot_sol[1, :], color="k")
        axs[2, 1].set_xlabel("Time [s]")
        axs[2, 1].set_ylabel("Elbow velocity [rad/s]")
        axs[0, 0].axis("equal")
        plt.tight_layout()
        plt.savefig(f"simulated_results_ocp_forcefield{force_field_magnitude}.png", dpi=300)
        plt.show()


        # --- Saving the solver's output after the optimization --- #

        # Save the version of bioptim and the date of the optimization for future reference
        repo = git.Repo(search_parent_directories=True)
        commit_id = str(repo.commit())
        branch = str(repo.active_branch)
        tag = repo.git.describe("--tags")
        bioptim_version = repo.git.version_info
        git_date = repo.git.log("-1", "--format=%cd")
        version_dic = {
            "commit_id": commit_id,
            "git_date": git_date,
            "branch": branch,
            "tag": tag,
            "bioptim_version": bioptim_version,
            "date_of_the_optimization": date.today().strftime("%b-%d-%Y-%H-%M-%S"),
        }

        # --- Save the results --- #
        with open(f"leuvenarm_muscle_driven_ocp_{example_type}_forcefield{force_field_magnitude}.pkl", "wb") as file:
            data = {
                "version": version_dic,
                "status": sol_ocp.status,
                "real_time_to_optimize": sol_ocp.real_time_to_optimize,
                "q_sol": q_sol,
                "qdot_sol": qdot_sol,
                "activations_sol": activations_sol,
                "excitations_sol": excitations_sol,
                "residual_tau": tau_sol,
                "q_simulated": q_simulated,
                "qdot_simulated": qdot_simulated,
                "mus_activation_simulated": mus_activation_simulated,
                "hand_pos_simulated": hand_pos_simulated,
                "hand_vel_simulated": hand_vel_simulated,
            }
            pickle.dump(data, file)

        # Save the solution for future use, you will only need to do sol.ocp = prepare_ocp() to get the same solution object as above.
        with open(f"leuvenarm_muscle_driven_ocp_{example_type}_forcefield{force_field_magnitude}.pkl", "wb") as file:
            del sol_ocp.ocp
            pickle.dump(sol_ocp, file)

if __name__ == "__main__":
    main()
