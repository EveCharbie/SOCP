import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from .examples.example_abstract import ExampleAbstract
from .transcriptions.transcription_abstract import TranscriptionAbstract
from .transcriptions.discretization_abstract import DiscretizationAbstract
from .transcriptions.mean_and_covariance import MeanAndCovariance
from .transcriptions.direct_multiple_shooting import DirectMultipleShooting
from .transcriptions.noise_discretization import NoiseDiscretization
from .live_plot_utils import create_variable_plot_out, update_variable_plot_out, OnlineCallback
from .constraints import Constraints


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


def get_the_save_path(
    solver,
    tol,
    ocp_example: ExampleAbstract,
    dynamics_transcription: TranscriptionAbstract,
    discretization_method: DiscretizationAbstract,
) -> str:
    # Save the results
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    status = "CVG" if solver.stats()["success"] else "DVG"
    print_tol = "{:1.1e}".format(tol).replace(".", "p")
    save_path = f"results/{ocp_example.name()}_{dynamics_transcription.name()}_{discretization_method.name()}_{status}_{print_tol}_{current_time}.pkl"
    return save_path


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


def print_constraints_at_init(
    g: cas.SX,
    lbg: cas.DM,
    ubg: cas.DM,
    g_names: list[str],
    w: cas.SX,
    w0: cas.DM,
):
    """Print the constraints at the initial guess"""
    g_func = cas.Function("constraints", [w], [g])
    g_eval = g_func(w0)
    for i_g in range(g_eval.shape[0]):
        g_value = g_eval[i_g].full().flatten()[0]
        if g_value < lbg[i_g] - 1e-6 or g_value > ubg[i_g] + 1e-6:
            print(f"Constraint {g_names[i_g]} ({i_g}-th): {g_value}")
        # else:
        #     print(f"Constraint {g_names[i_g]} ({i_g}-th): {g_value}")


def check_the_configuration(
    ocp_example: ExampleAbstract,
    dynamics_transcription: TranscriptionAbstract,
    discretization_method: DiscretizationAbstract,
):
    # TODO: I think this is possible now
    if isinstance(dynamics_transcription, DirectMultipleShooting):
        if discretization_method.with_cholesky:
            raise ValueError("Cholesky decomposition is not compatible with DirectMultipleShooting transcription.")
        if discretization_method.with_helper_matrix:
            raise ValueError("Helper matrix is not compatible with DirectMultipleShooting transcription.")


def prepare_ocp(
    ocp_example: ExampleAbstract,
    dynamics_transcription: TranscriptionAbstract,
    discretization_method: DiscretizationAbstract,
):

    check_the_configuration(ocp_example, dynamics_transcription, discretization_method)

    nb_collocation_points = dynamics_transcription.nb_collocation_points

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
        collocation_points_initial_guesses,
    ) = ocp_example.get_bounds_and_init(
        ocp_example.n_shooting,
        nb_collocation_points,
    )
    motor_noise_magnitude, sensory_noise_magnitude = ocp_example.get_noises_magnitude()

    variables_vector = discretization_method.declare_variables(
        ocp_example=ocp_example,
        states_lower_bounds=states_lower_bounds,
        controls_lower_bounds=controls_lower_bounds,
    )
    noises_numerical, noises_single = discretization_method.declare_noises(
        ocp_example.model,
        ocp_example.n_shooting,
        ocp_example.nb_random,
        motor_noise_magnitude,
        sensory_noise_magnitude,
        seed=ocp_example.seed,
    )

    # Start with an empty NLP
    j = 0
    constraints = Constraints(ocp_example.n_shooting)

    # Add dynamics constraints (continuity and others)
    dynamics_transcription.initialize_dynamics_integrator(
        ocp_example=ocp_example,
        discretization_method=discretization_method,
        variables_vector=variables_vector,
        noises_single=noises_single,
    )
    dynamics_transcription.set_dynamics_constraints(
        ocp_example,
        discretization_method,
        variables_vector,
        noises_single,
        noises_numerical,
        constraints,
        n_threads=ocp_example.n_threads,
    )

    # Add constraints specific to this problem
    ocp_example.set_specific_constraints(
        ocp_example.model,
        discretization_method,
        dynamics_transcription,
        variables_vector,
        noises_single,
        noises_numerical,
        constraints,
    )

    # Add objectives specific to this problem
    j_example = ocp_example.get_specific_objectives(
        ocp_example.model,
        discretization_method,
        dynamics_transcription,
        variables_vector,
        noises_single,
        noises_numerical,
    )

    j += j_example

    # Redeclare the bounds if they have changed
    lb_vector, ub_vector, w0_vector = discretization_method.declare_bounds_and_init(
        ocp_example=ocp_example,
        states_lower_bounds=states_lower_bounds,
        states_upper_bounds=states_upper_bounds,
        states_initial_guesses=states_initial_guesses,
        controls_lower_bounds=controls_lower_bounds,
        controls_upper_bounds=controls_upper_bounds,
        controls_initial_guesses=controls_initial_guesses,
        collocation_points_initial_guesses=collocation_points_initial_guesses,
    )

    # Modify the initial guess if needed
    discretization_method.modify_init(ocp_example, w0_vector)

    g, lbg, ubg, g_names = constraints.to_list()

    ocp = {
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
        "w": variables_vector.get_full_vector(keep_only_symbolic=True),
        "variable_init": w0_vector,
        "variable_lb": lb_vector,
        "variable_ub": ub_vector,
        "w0": w0_vector.get_full_vector(keep_only_symbolic=True),
        "lbw": lb_vector.get_full_vector(keep_only_symbolic=True),
        "ubw": ub_vector.get_full_vector(keep_only_symbolic=True),
        "j": j,
        "g": g,
        "lbg": lbg,
        "ubg": ubg,
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
    save_path_suffix: str = "",
) -> tuple[np.ndarray, dict[str, any], cas.Function, cas.Function, str]:
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
    if g.shape[0] != lbg.shape[0]:
        raise ValueError(
            f"The number of constraints in g ({g.shape[0]}) must be equal to the length of lbg ({lbg.shape[0]})."
        )
    if g.shape[0] != ubg.shape[0]:
        raise ValueError(
            f"The number of constraints in g ({g.shape[0]}) must be equal to the length of ubg ({ubg.shape[0]})."
        )

    if w0.shape[0] != w.shape[0]:
        raise ValueError(f"The initial guess w0 must have shape ({w.shape[0]}, 1), but has shape {w0.shape}.")
    if lbw.shape[0] != w.shape[0]:
        raise ValueError(f"The lower bounds lbw must have shape ({w.shape[0]}, 1), but has shape {lbw.shape}.")
    if ubw.shape[0] != w.shape[0]:
        raise ValueError(f"The upper bounds ubw must have shape ({w.shape[0]}, 1), but has shape {ubw.shape}.")

    if j.shape != (1, 1):
        raise ValueError(f"The objective j must be a scalar of shape (1, 1), but has shape {j.shape}.")

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

    print_constraints_at_init(g, lbg, ubg, g_names, w, w0)
    if pre_optim_plot:
        plot_jacobian(g, w)

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

    # Print the constraints
    print_constraints_at_init(g, lbg, ubg, g_names, w, w_opt)

    # Plot the solution
    time_vector = np.linspace(0, w_opt[0], ocp["n_shooting"] + 1)
    states_fig, states_plots, states_axes, controls_fig, controls_plots, controls_axes = create_variable_plot_out(
        ocp,
        time_vector,
    )
    update_variable_plot_out(
        ocp,
        time_vector,
        states_plots,
        controls_plots,
        w_opt,
    )
    save_path = get_the_save_path(
        solver,
        ocp_example.tol,
        ocp_example,
        ocp["dynamics_transcription"],
        ocp["discretization_method"],
    ).replace(
        ".pkl", f"_{save_path_suffix}.pkl"
    )

    states_fig.savefig(save_path.replace(".pkl", "_states_opt.png"))
    controls_fig.savefig(save_path.replace(".pkl", "_controls_opt.png"))

    return w_opt, solver, grad_f_func, grad_g_func, save_path
