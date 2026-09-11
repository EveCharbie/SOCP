import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from socp.models.vertebrate_arm_model import VertebrateArmModel


def Euler_Maruyama_step(dt, x, u_prev, u_next, noise, model):
    q = x[:2]
    qdot = x[2:]

    # drift
    u_this_time = u_prev  # Euler evaluates only at the start of the interval
    xdot = np.array(model.dynamics(x, u_this_time, None, np.zeros_like(noise), with_q_qdot=True)).flatten()
    qddot = xdot[2:]

    # diffusion: dW ~ N(0, dt) per channel
    dW = np.sqrt(dt) * noise
    Minv = np.linalg.inv(model.mass_matrix()(q))

    qdot_new = qdot + qddot * dt + Minv @ dW
    q_new    = q + qdot * dt
    return np.hstack((q_new, qdot_new))


def Euler_step(dt, x, u_prev, u_next, model):
    q = x[:2]
    qdot = x[2:]

    # drift
    u_this_time = u_prev  # Euler evaluates only at the start of the interval
    xdot = np.array(model.dynamics(x, u_this_time, None, np.zeros_like(u_this_time), with_q_qdot=True)).flatten()
    qddot = xdot[2:]

    qdot_new = qdot + qddot * dt
    q_new    = q + qdot * dt
    return np.hstack((q_new, qdot_new))


def dynamics_wrapper(t, dt, x, u_prev, u_next, noise, model):
    u_this_time = u_prev  # + (u_next - u_prev) * t / dt
    dW = noise * np.sqrt(dt)
    return np.array(model.dynamics(x, u_this_time, None, dW, with_q_qdot=True)).flatten()


def SRK4_step(dt, x, u_prev, u_next, noise, model):
    # 4th order RK for the drift, plus a single non-anticipating (Ito) diffusion kick.
    # Minv is evaluated once, at the state at the start of the step, so the diffusion is never
    # re-evaluated at intermediate stages (that would silently turn this into a Stratonovich scheme).
    u_this_time = u_prev #  + (u_next - u_prev) * t / dt

    def drift(x_):
        return np.array(model.dynamics(x_, u_this_time, None, np.zeros_like(noise), with_q_qdot=True)).flatten()

    k1 = drift(x)
    k2 = drift(x + dt / 2 * k1)
    k3 = drift(x + dt / 2 * k2)
    k4 = drift(x + dt * k3)
    x_drift = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    q0 = x[:2]
    dW = np.sqrt(dt) * noise
    Minv = np.linalg.inv(model.mass_matrix()(q0))
    diffusion = np.hstack((np.zeros(2), Minv @ dW))

    return x_drift + diffusion


np.random.seed(0)
nb_states = 4
n_shooting = 100
n_simulations = 1000
model = VertebrateArmModel(1)
time_vector = np.linspace(0, 1, n_shooting+1)
dt = time_vector[1] - time_vector[0]

tau = np.vstack((np.sin(time_vector*10), np.cos(time_vector*10)))
mean_states_init = np.array([1.5, 1.5, 0, 0])
std_states_init = np.array([1e-4, 1e-4, 1e-4, 1e-4])
noise_magnitude = np.array([0.01] * 2)

x_simulated = np.zeros((nb_states, n_shooting + 1))
x_simulated[:, 0] = mean_states_init

x_simulated_EM = np.zeros((nb_states, n_shooting + 1, n_simulations))
x_simulated_SRK4 = np.zeros((nb_states, n_shooting + 1, n_simulations))

# Set the random initial state for all simulations
# Normalization : exactly zero mean, exactly unit variance
init_noised_states = np.random.normal(
    loc=mean_states_init,
    scale=std_states_init,
    size=(n_simulations, nb_states),
).T
init_noised_states_standardized = (init_noised_states - np.mean(init_noised_states, axis=1)[
    :, np.newaxis]) / np.std(init_noised_states, axis=1)[:, np.newaxis]
init_noised_states_nomralized = np.repeat(mean_states_init[:, np.newaxis], n_simulations, axis=1) + np.repeat(
    std_states_init[:, np.newaxis], n_simulations,
    axis=1) * init_noised_states_standardized
x_simulated_EM[:, 0, :] = init_noised_states_nomralized[:, :]
x_simulated_SRK4[:, 0, :] = init_noised_states_nomralized[:, :]

# Set the random noise for all simulations
motor_noises = np.random.normal(
    scale=noise_magnitude,
    size=(n_simulations, n_shooting, 2),
).T  # (2, 100, 1000)

for i_simulation in range(n_simulations):
    for i_node in range(n_shooting):

        # Common
        u_prev = tau[:, i_node]
        u_next = tau[:, i_node + 1]
        noise_this_time = motor_noises[:, i_node, i_simulation]

        # Without noise
        if i_simulation == 0:
            x_prev = x_simulated[:, i_node].flatten()
            sol = solve_ivp(
                fun=lambda t, x: dynamics_wrapper(t, dt, x, u_prev, u_next, np.zeros((2, )), model),
                t_span=(0.0, dt),
                y0=x_prev,
                method="DOP853",
                rtol=1e-6,
                atol=1e-8,
            )
            x_simulated[:, i_node + 1] = sol.y[:, -1]

        # EM
        x_prev_EM = x_simulated_EM[:, i_node, i_simulation].flatten()
        x_simulated_EM[:, i_node + 1, i_simulation] = Euler_Maruyama_step(dt, x_prev_EM, u_prev, u_next, noise_this_time, model)

        # SRK4
        x_prev_SRK4 = x_simulated_SRK4[:, i_node, i_simulation].flatten()
        x_simulated_SRK4[:, i_node + 1, i_simulation] = SRK4_step(dt, x_prev_SRK4, u_prev, u_next, noise_this_time, model)


# Parameters
T = 1.0  # Total time
N = 100  # Number of time steps
dt = T / N  # Time step size
t = np.linspace(0, T, N + 1)

# Plotting the result
fig, axs = plt.subplots(1, 3)

for i_simulation in range(n_simulations):
    axs[0].plot(t, x_simulated_EM[0, :, i_simulation], "-r", alpha=0.01)
    axs[1].plot(t, x_simulated_EM[1, :, i_simulation], "-r", alpha=0.01)
axs[0].plot(t, np.mean(x_simulated_EM[0, :, :], axis=1), "-r", label="Euler-Maruyama")
axs[1].plot(t, np.mean(x_simulated_EM[1, :, :], axis=1), "-r", label="Euler-Maruyama")

for i_simulation in range(n_simulations):
    axs[0].plot(t, x_simulated_SRK4[0, :, i_simulation], "-g", alpha=0.01)
    axs[1].plot(t, x_simulated_SRK4[1, :, i_simulation], "-g", alpha=0.01)
axs[0].plot(t, np.mean(x_simulated_SRK4[0, :, :], axis=1), "-g", label="SRK4")
axs[1].plot(t, np.mean(x_simulated_SRK4[1, :, :], axis=1), "-g", label="SRK4")

axs[0].plot(t, x_simulated[0, :], "--k", label="without noise (DOP853)")
axs[1].plot(t, x_simulated[1, :], "--k", label="without noise (DOP853)")

axs[2].plot(t, np.mean(x_simulated_EM[0, :, :], axis=1) - x_simulated[0, :], "-r", label="Euler-Maruyama error")
axs[2].plot(t, np.mean(x_simulated_EM[1, :, :], axis=1) - x_simulated[1, :], "-r", label="Euler-Maruyama error")
axs[2].plot(t, np.mean(x_simulated_SRK4[0, :, :], axis=1) - x_simulated[0, :], "-g", label="SRK4 error")
axs[2].plot(t, np.mean(x_simulated_SRK4[1, :, :], axis=1) - x_simulated[1, :], "-g", label="SRK4 error")

axs[1].legend()
axs[0].set_xlabel("Time")
axs[1].set_xlabel("Time")
axs[2].set_xlabel("Time")
axs[0].set_ylabel("q(t)")
axs[1].set_ylabel("q(t)")
axs[2].set_ylabel("Error")
plt.savefig("reintegration_comparison.png")



# Parameters
N_overdiscretized = 10000  # Number of time steps
dt_overdiscretized = T / N_overdiscretized  # Time step size
t_overdiscretized = np.linspace(0, T, N_overdiscretized + 1)
ratio = int(N_overdiscretized / N)

x_simulated_Euler = np.zeros((nb_states, N_overdiscretized + 1))
x_simulated_Euler[:, 0] = mean_states_init[:]

for i_node in range(N_overdiscretized):

    # Common
    u_prev = tau[:, int(i_node//ratio)]
    u_next = tau[:, int((i_node + 1)//ratio)]

    # Euler
    x_prev_Euler = x_simulated_Euler[:, i_node].flatten()
    x_simulated_Euler[:, i_node + 1] = Euler_step(dt_overdiscretized, x_prev_Euler, u_prev, u_next, model)

# Plotting the result
fig, axs = plt.subplots(1, 3)

axs[0].plot(t_overdiscretized, x_simulated_Euler[0, :], "-r", label="Euler")
axs[1].plot(t_overdiscretized, x_simulated_Euler[1, :], "-r", label="Euler")

axs[0].plot(t, x_simulated[0, :], "--k", label="without noise (DOP853)")
axs[1].plot(t, x_simulated[1, :], "--k", label="without noise (DOP853)")

axs[2].plot(t, x_simulated_Euler[0, 0::ratio] - x_simulated[0, :], "-r", label="Euler error")
axs[2].plot(t, x_simulated_Euler[1, 0::ratio] - x_simulated[1, :], "-r", label="Euler error")
axs[1].legend()
axs[0].set_xlabel("Time")
axs[1].set_xlabel("Time")
axs[2].set_xlabel("Time")
axs[0].set_ylabel("q(t)")
axs[1].set_ylabel("q(t)")
axs[2].set_ylabel("Error")
plt.savefig("reintegration_ground_truth.png")
plt.show()
