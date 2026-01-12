import casadi as cas
import numpy as np
import matplotlib.pyplot as plt

from .examples.example_abstract import ExampleAbstract
from .transcriptions.transcription_abstract import TranscriptionAbstract
from .transcriptions.discretization_abstract import DiscretizationAbstract
from .transcriptions.mean_and_covariance import MeanAndCovariance
from .transcriptions.direct_multiple_shooting import DirectMultipleShooting
from .transcriptions.noise_discretization import NoiseDiscretization

from .live_plot_utils import OnlineCallback


def get_dm_value(function, values):
    """
    Get the DM value of a CasADi function.
    """
    variables = []
    for i_var in range(len(values)):
        variables += [cas.SX.sym(f"var_{i_var}", values[i_var].shape[0], 1)]
    func = cas.Function("temp_func", variables, [function(*variables)])
    output = func(*values)
    return output


def plot_jacobian(g: cas.SX, w: cas.SX):
    """Plot the Jacobian matrix using matplotlib"""
    sparsity = cas.jacobian_sparsity(g, w)
    plt.figure()
    plt.imshow(sparsity)
    plt.title("Jacobian Sparsity Pattern")
    plt.xlabel("Variables")
    plt.ylabel("Constraints")
    plt.savefig("jacobian_sparsity.png")
    plt.show()


def print_constraints_at_init(g: cas.SX, g_names: list[str], w: cas.SX, w0: cas.DM):
    """Print the constraints at the initial guess"""
    g_func = cas.Function("constraints", [w], [g])
    g_eval = g_func(w0)
    for i_g in range(g_eval.shape[0]):
        g_value = g_eval[i_g].full().flatten()[0]
        if np.abs(g_value) > 0.1:
            print(f"Constraint {g_names[i_g]} ({i_g}-th): {g_value}  *********")
        else:
            print(f"Constraint {g_names[i_g]} ({i_g}-th): {g_value}")


def check_the_configuration(
    ocp_example: ExampleAbstract,
    dynamics_transcription: TranscriptionAbstract,
    discretization_method: DiscretizationAbstract,
):
    if isinstance(dynamics_transcription, DirectMultipleShooting):
        if discretization_method.with_cholesky:
            raise ValueError(
                "Cholesky decomposition is not compatible with DirectMultipleShooting transcription."
            )
        if discretization_method.with_helper_matrix:
            raise ValueError(
                "Helper matrix is not compatible with DirectMultipleShooting transcription."
            )


def prepare_ocp(
    ocp_example: ExampleAbstract,
    dynamics_transcription: TranscriptionAbstract,
    discretization_method: DiscretizationAbstract,
):

    check_the_configuration(ocp_example, dynamics_transcription, discretization_method)

    # Fix the random seed for the noise generation
    np.random.seed(ocp_example.seed)

    # Variables
    (
        states_lower_bounds,
        states_upper_bounds,
        states_initial_guesses,
        controls_lower_bounds,
        controls_upper_bounds,
        controls_initial_guesses,
    ) = ocp_example.get_bounds_and_init(ocp_example.n_shooting)
    motor_noise_magnitude, sensory_noise_magnitude = ocp_example.get_noises_magnitude()

    x, u, w, lbw, ubw, w0 = discretization_method.declare_variables(
        ocp_example=ocp_example,
        states_lower_bounds=states_lower_bounds,
        states_upper_bounds=states_upper_bounds,
        states_initial_guesses=states_initial_guesses,
        controls_lower_bounds=controls_lower_bounds,
        controls_upper_bounds=controls_upper_bounds,
        controls_initial_guesses=controls_initial_guesses,
    )
    noises_numerical, noises_single = discretization_method.declare_noises(
        ocp_example.model, ocp_example.n_shooting, ocp_example.nb_random, motor_noise_magnitude, sensory_noise_magnitude
    )

    # Start with an empty NLP
    j = 0
    g = []
    lbg = []
    ubg = []
    g_names = []

    # Add dynamics constraints (continuity and others)
    dynamics_transcription.initialize_dynamics_integrator(
        ocp_example=ocp_example,
        discretization_method=discretization_method,
        x=x,
        u=u,
        noises_single=noises_single,
    )
    g_dynamics, lbg_dynamics, ubg_dynamics, g_names_dynamics = dynamics_transcription.get_dynamics_constraints(
        ocp_example.model,
        discretization_method,
        ocp_example.n_shooting,
        x,
        u,
        noises_single,
        noises_numerical,
        ocp_example.dt,
        n_threads=ocp_example.n_threads,
    )
    g += g_dynamics
    lbg += lbg_dynamics
    ubg += ubg_dynamics
    g_names += g_names_dynamics

    # Add constraints specific to this problem
    g_example, lbg_example, ubg_example, g_names_example = ocp_example.get_specific_constraints(
        ocp_example.model,
        discretization_method,
        dynamics_transcription,
        x,
        u,
        noises_single,
        noises_numerical,
    )
    g += g_example
    lbg += lbg_example
    ubg += ubg_example
    g_names += g_names_example

    # Add objectives specific to this problem
    j_example = ocp_example.get_specific_objectives(
        ocp_example.model,
        discretization_method,
        dynamics_transcription,
        x,
        u,
        noises_single,
        noises_numerical,
    )

    j += j_example

    ocp = {
        "model": ocp_example.model,
        "ocp_example": ocp_example,
        "dynamics_transcription": dynamics_transcription,
        "discretization_method": discretization_method,
        "states_lower_bounds": states_lower_bounds,
        "states_upper_bounds": states_upper_bounds,
        "states_initial_guesses": states_initial_guesses,
        "controls_lower_bounds": controls_lower_bounds,
        "controls_upper_bounds": controls_upper_bounds,
        "controls_initial_guesses": controls_initial_guesses,
        "motor_noise_magnitude": motor_noise_magnitude,
        "sensory_noise_magnitude": sensory_noise_magnitude,
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
    ocp_example: ExampleAbstract,
    hessian_approximation: str = "exact",  # or "limited-memory",
    output_file: str = None,
    linear_solver: str = "ma97",
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
        raise ValueError(
            f"The length of g_names ({len(g_names)}) must be equal to the number of constraints in g ({g.shape[0]})."
        )

    # Set IPOPT options
    opts = {
        "ipopt.max_iter": ocp_example.max_iter,
        "ipopt.tol": ocp_example.tol,
        "ipopt.linear_solver": linear_solver,
        "ipopt.hessian_approximation": hessian_approximation,
        # "ipopt.output_file": output_file,
        # "expand": True,
    }

    # Online callback for live plotting
    grad_f_func = cas.Function("grad_f", [w], [cas.gradient(j, w)])
    grad_g_func = cas.Function("grad_g", [w], [cas.jacobian(g, w).T])

    if pre_optim_plot:
        plot_jacobian(g, w)
        print_constraints_at_init(g, g_names, w, w0)

    if show_online_optim:
        online_callback = OnlineCallback(
            nx=w.shape[0],
            ng=g.shape[0],
            grad_f_func=grad_f_func,
            grad_g_func=grad_g_func,
            g_names=g_names,
            ocp=ocp,
        )
        opts["iteration_callback"] = online_callback

    # Create an NLP solver
    nlp = {"f": j, "x": w, "g": g}
    solver = cas.nlpsol("solver", "ipopt", nlp, opts)

    # Solve the NLP
    print("\n\n\n Solving the SOCP \n\n\n")
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol["x"].full().flatten()

    return w_opt, solver
