import casadi as cas
import matplotlib.pyplot as plt
import numpy as np

from bioptim import Solution, SolutionMerge

from .deterministic_arm_model import DeterministicArmModel
from ..utils import RK4



def get_forward_dynamics_func(nlp, force_field_magnitude):

    x = cas.MX.sym("x", 10)
    q = x[:2]
    qdot = x[2:4]
    mus_activations = x[4:]
    u = cas.MX.sym("u", 8)
    mus_excitations = u[:6]
    tau_residuals = u[6:]
    motor_noise = cas.MX.sym("motor_noise", 2)

    muscles_tau = nlp.model.get_muscle_torque(q, qdot, mus_activations)

    tau_force_field = nlp.model.force_field(q, force_field_magnitude)

    torques_computed = muscles_tau + tau_force_field + tau_residuals + motor_noise

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

    casadi_dynamics = cas.Function(
        "forward_dynamics",
        [x, u, motor_noise],
        [cas.reshape(dxdt, (10, 1))],
        ["x", "u", "motor_noise"],
        ["dxdt"],
    )
    return casadi_dynamics

def plot_ocp(
        sol_ocp: Solution,
        motor_noise_std: float,
        hand_initial_position: np.ndarray,
        hand_final_position: np.ndarray,
        force_field_magnitude: float,
        n_shooting: int,
        final_time: float,
        n_simulations: int = 100,
):

    q_sol = sol_ocp.decision_states(to_merge=SolutionMerge.NODES)["q"]
    qdot_sol = sol_ocp.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
    activations_sol = sol_ocp.decision_states(to_merge=SolutionMerge.NODES)["muscles"]
    excitations_sol = sol_ocp.decision_controls(to_merge=SolutionMerge.NODES)["muscles"]
    tau_sol = sol_ocp.decision_controls(to_merge=SolutionMerge.NODES)["tau"]

    OCP_color = "#5DC962"

    model = DeterministicArmModel()
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
            u = cas.vertcat(excitations_sol[:, i_node], tau_sol[:, i_node])
            x_next = RK4(x_prev, u, dt_actual, motor_noise[:, i_node], forward_dyn_func, n_steps=5)
            q_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[-1, :2], (2,))
            qdot_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[-1, 2:4], (2,))
            mus_activation_simulated[i_simulation, :, i_node + 1] = np.reshape(x_next[-1, 4:], (6,))
        hand_pos_simulated[i_simulation, :, i_node + 1] = np.reshape(hand_pos_fcn(x_next[-1, :2])[:2], (2,))
        hand_vel_simulated[i_simulation, :, i_node + 1] = np.reshape(
            hand_vel_fcn(x_next[-1, :2], x_next[-1, 2:4])[:2], (2,)
        )
        axs[0, 0].plot(
            hand_pos_simulated[i_simulation, 0, :],
            hand_pos_simulated[i_simulation, 1, :],
            color=OCP_color,
            linewidth=0.5,
        )
        axs[1, 0].plot(
            np.linspace(0, final_time, n_shooting + 1),
            q_simulated[i_simulation, 0, :],
            color=OCP_color,
            linewidth=0.5,
        )
        axs[2, 0].plot(
            np.linspace(0, final_time, n_shooting + 1),
            q_simulated[i_simulation, 1, :],
            color=OCP_color,
            linewidth=0.5,
        )
        axs[0, 1].plot(
            np.linspace(0, final_time, n_shooting + 1),
            np.linalg.norm(hand_vel_simulated[i_simulation, :, :], axis=0),
            color=OCP_color,
            linewidth=0.5,
        )
        axs[1, 1].plot(
            np.linspace(0, final_time, n_shooting + 1),
            qdot_simulated[i_simulation, 0, :],
            color=OCP_color,
            linewidth=0.5,
        )
        axs[2, 1].plot(
            np.linspace(0, final_time, n_shooting + 1),
            qdot_simulated[i_simulation, 1, :],
            color=OCP_color,
            linewidth=0.5,
        )
    hand_pos_without_noise = np.zeros((2, n_shooting + 1))
    hand_vel_without_noise = np.zeros((2, n_shooting + 1))
    for i_node in range(n_shooting + 1):
        hand_pos_without_noise[:, i_node] = np.reshape(hand_pos_fcn(q_sol[:, i_node])[:2], (2,))
        hand_vel_without_noise[:, i_node] = np.reshape(
            hand_vel_fcn(q_sol[:, i_node], qdot_sol[:, i_node])[:2], (2,)
        )

    axs[0, 0].plot(hand_pos_without_noise[0, :], hand_pos_without_noise[1, :], color="k")
    axs[0, 0].plot(hand_initial_position[0], hand_initial_position[1], color="tab:green", marker="o", markersize=2)
    axs[0, 0].plot(hand_final_position[0], hand_final_position[1], color="tab:red", marker="o", markersize=2)
    axs[0, 0].set_xlabel("X [m]")
    axs[0, 0].set_ylabel("Y [m]")
    axs[0, 0].set_title("Hand position simulated")
    axs[1, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_sol[0, :], color="k")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Shoulder angle [rad]")
    axs[2, 0].plot(np.linspace(0, final_time, n_shooting + 1), q_sol[1, :], color="k")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Elbow angle [rad]")
    axs[0, 1].plot(
        np.linspace(0, final_time, n_shooting + 1), np.linalg.norm(hand_vel_without_noise, axis=0), color="k"
    )
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
    plt.savefig(f"figures/simulated_results_ocp_forcefield{force_field_magnitude}.png", dpi=300)
    plt.show()
