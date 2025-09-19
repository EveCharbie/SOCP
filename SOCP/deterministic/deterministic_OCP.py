import casadi as cas
import numpy as np

from bioptim import (
    OptimalControlProgram,
    PhaseDynamics,
    InitialGuessList,
    ObjectiveFcn,
    ObjectiveList,
    DynamicsOptionsList,
    BoundsList,
    InterpolationType,
    PenaltyController,
    Node,
    ConstraintList,
    ConstraintFcn,
)

from ..utils import ExampleType
from .deterministic_arm_model import DeterministicArmModel


def track_final_marker(controller: PenaltyController, example_type) -> cas.MX:
    """
    Track the hand position.
    """
    q = controller.q
    ee_pos = controller.model.end_effector_position(q)
    out = ee_pos if example_type == ExampleType.CIRCLE else ee_pos[1]
    return out


def prepare_ocp(
    final_time: float,
    n_shooting: int,
    hand_final_position: np.ndarray,
    force_field_magnitude: float = 0,
    example_type=ExampleType.CIRCLE,
):

    # Model
    bio_model = DeterministicArmModel()
    bio_model.force_field_magnitude = force_field_magnitude

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL_SHOOTING, key="muscles", weight=1 / 2, quadratic=True
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE, node=Node.ALL, key="muscles", weight=1 / 2, quadratic=True
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, node=Node.ALL_SHOOTING, key="tau", weight=10, quadratic=True
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        track_final_marker,
        node=Node.END,
        target=hand_final_position,
        example_type=example_type,
        quadratic=False,
    )
    # All tau_residual must be zero
    constraints.add(ConstraintFcn.TRACK_CONTROL, key="tau", node=Node.ALL_SHOOTING)

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(
        expand_dynamics=False,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=None,
    )

    # Bounds
    shoulder_pos_initial = 0.349065850398866
    elbow_pos_initial = 2.245867726451909  # Optimized in Tom's version

    x_bounds = BoundsList()
    n_muscles = 6
    n_q = bio_model.nb_q
    n_qdot = bio_model.nb_qdot
    n_states = n_q + n_qdot + n_muscles

    q_min = np.ones((n_q, 3)) * 0
    q_max = np.ones((n_q, 3)) * np.pi
    q_min[:, 0] = np.array([shoulder_pos_initial, elbow_pos_initial])
    q_max[:, 0] = np.array([shoulder_pos_initial, elbow_pos_initial])
    x_bounds.add(
        "q",
        min_bound=q_min,
        max_bound=q_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    qdot_min = np.ones((n_q, 3)) * -10 * np.pi
    qdot_max = np.ones((n_q, 3)) * 10 * np.pi
    qdot_min[:, 0] = np.array([0, 0])
    qdot_max[:, 0] = np.array([0, 0])
    qdot_min[:, 2] = np.array([0, 0])
    qdot_max[:, 2] = np.array([0, 0])
    x_bounds.add(
        "qdot",
        min_bound=qdot_min,
        max_bound=qdot_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    muscle_min = np.ones((n_muscles, 3)) * 0
    muscle_max = np.ones((n_muscles, 3)) * 1
    muscle_min[:, 0] = np.array([0, 0, 0, 0, 0, 0])
    muscle_max[:, 0] = np.array([0, 0, 0, 0, 0, 0])
    x_bounds.add(
        "muscles",
        min_bound=muscle_min,
        max_bound=muscle_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    u_bounds = BoundsList()
    controls_min = np.ones((n_muscles, 3)) * 0.001
    controls_max = np.ones((n_muscles, 3)) * 1
    controls_min[:, 0] = np.array([0, 0, 0, 0, 0, 0])
    controls_max[:, 0] = np.array([0, 0, 0, 0, 0, 0])
    u_bounds.add(
        "muscles",
        min_bound=controls_min,
        max_bound=controls_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )
    tau_min = np.ones((n_q, 3)) * -10
    tau_max = np.ones((n_q, 3)) * 10
    tau_min[:, 0] = np.array([0, 0])
    tau_max[:, 0] = np.array([0, 0])
    u_bounds.add(
        "tau",
        min_bound=tau_min,
        max_bound=tau_max,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

    # Initial guesses
    elbow_pos_final = 1.159394851847144  # Optimized in Tom's version
    shoulder_pos_final = 0.959931088596881

    states_init = np.zeros((n_states, n_shooting + 1))
    states_init[0, :] = np.linspace(shoulder_pos_initial, shoulder_pos_final, n_shooting + 1)
    states_init[1, :] = np.linspace(elbow_pos_initial, elbow_pos_final, n_shooting + 1)
    states_init[n_q + n_qdot :, :] = 0.01

    x_init = InitialGuessList()
    x_init.add("q", initial_guess=states_init[:n_q, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", initial_guess=states_init[n_q : n_q + n_qdot, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("muscles", initial_guess=states_init[n_q + n_qdot :, :], interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    u_init.add("muscles", initial_guess=np.ones((n_muscles,)) * 0.01, interpolation=InterpolationType.CONSTANT)
    u_init.add("tau", initial_guess=np.ones((n_q,)) * 0.01, interpolation=InterpolationType.CONSTANT)

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
        n_threads=1,
    )

