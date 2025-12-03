import casadi as cas
import numpy as np
import matplotlib.pyplot as plt


def get_dm_value(function, values):
    """
    Get the DM value of a CasADi function.
    """
    variables = []
    for i_var in range(len(values)):
        variables += [cas.MX.sym(f"var_{i_var}", values[i_var].shape[0], 1)]
    func = cas.Function("temp_func", variables, [function(*variables)])
    output = func(*values)
    return output


def plot_jacobian(g: cas.MX, w: cas.MX):
    """Plot the Jacobian matrix using matplotlib"""
    sparsity = cas.jacobian_sparsity(g, w)
    plt.figure()
    plt.imshow(sparsity)
    plt.title("Jacobian Sparsity Pattern")
    plt.xlabel("Variables")
    plt.ylabel("Constraints")
    plt.savefig("jacobian_sparsity.png")
    plt.show()


def prepare_ocp(
        ocp_example: OcpExample,
        dynamics_transcription: DynamicsTranscription,
):

    # Fix the random seed for the noise generation
    np.random.seed(ocp_example.seed)

    # Variables
    x, u, w, lbw, ubw, w0 = ocp_example.declare_variables()
    noises_numerical, noises_single = ocp_example.declare_noises(
        ocp_example.n_shooting, ocp_example.n_random, ocp_example.model.motor_noise_magnitude, ocp_example.model.sensory_noise_magnitude
    )

    # Start with an empty NLP
    j = 0
    g = []
    lbg = []
    ubg = []
    g_names = []

    # Add constraints specific to this problem
    g_example, lbg_example, ubg_example, g_names_example = ocp_example.get_specific_constraints(
        ocp_example.model,
        x,
        u,
        noises_single,
        noises_numerical,
    )
    g += g_example
    lbg += lbg_example
    ubg += ubg_example
    g_names += ubg_example

    # Add dynamics constraints
    g_dynamics, lbg_dynamics, ubg_dynamics, g_names_dynamics = dynamics_transcription.get_dynamics_constraints(
        ocp_example.model,
        x,
        u,
        noises_single,
        noises_numerical,
        ocp_example.dt,
        n_threads = ocp_example.n_threads,
        )
    g += g_dynamics
    lbg += lbg_dynamics
    ubg += ubg_dynamics
    g_names += g_names_dynamics

    # Add objectives specific to this problem
    j_example = ocp_example.get_specific_objectives(
        ocp_example.model,
        x,
        u,
        noises_single,
        noises_numerical,
    )
    j += j_example

    ocp = {
        "model": ocp_example.model,
        "w": cas.vertcat(*w),
        "w0": cas.vertcat(*w0),
        "lbw": cas.vertcat(*lbw),
        "ubw": cas.vertcat(*ubw),
        "j": j,
        "g": cas.vertcat(*g),
        "lbg": cas.vertcat(*lbg),
        "ubg": cas.vertcat(*ubg),
        "g_names": g_names,
        "n_shooting": ocp_example.n_shooting,
        "final_time": ocp_example.final_time,
    }
    return ocp


def solve_ocp(
    ocp: dict[str, any],
    ocp_example: OcpExample,
    hessian_approximation: str = "exact",  # or "limited-memory",
    output_file: str = None,
    pre_optim_plot: bool = False,
    show_online_optim: bool = True,
) -> tuple[np.ndarray, dict[str, any]]:
    """Solve the problem using IPOPT solver"""

    # Extract the problem
    w = ocp["w"]
    j = ocp["j"]
    g = ocp["g"]
    w0 = ocp["w0"]
    lbw = ocp["lbw"]
    ubw = ocp["ubw"]
    lbg = ocp["lbg"]
    ubg = ocp["ubg"]
    g_names = ocp["g_names"]

    if len(g_names) != g.shape[0]:
        raise ValueError("The length of g_names must be equal to the number of constraints in g.")

    if pre_optim_plot:
        plot_jacobian(g, w)

    # Set IPOPT options
    opts = {
        "ipopt.max_iter": ocp_example.max_iter,
        "ipopt.tol": ocp_example.tol,
        "ipopt.linear_solver": "ma97",
        "ipopt.hessian_approximation": hessian_approximation,
        # "ipopt.output_file": output_file,
        # "expand": True,
    }

    # Create an NLP solver
    nlp = {"f": j, "x": w, "g": g}
    solver = cas.nlpsol("solver", "ipopt", nlp, opts)

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol["x"].full().flatten()

    return w_opt, solver