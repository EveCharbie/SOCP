import numpy as np
import casadi as cas
from datetime import datetime

from socp import (
    ArmReaching,
    ObstacleAvoidance,
    DirectMultipleShooting,
    DirectCollocationTrapezoidal,
    DirectCollocationPolynomial,
    NoiseDiscretization,
    MeanAndCovariance,
    prepare_ocp,
    solve_ocp,
    save_results,
)


def main():
    tol = 1e-6
    n_simulations = 100

    # # Arm Reaching - DirectMultipleShooting - NoiseDiscretization  -> Does not converge, but runs
    # ocp_example = ArmReaching()
    # dynamics_transcription = DirectMultipleShooting()
    # discretization_method = NoiseDiscretization()

    # # Arm Reaching - DirectMultipleShooting - NoiseDiscretization  -> ? converge, but runs
    # ocp_example = ArmReaching()
    # dynamics_transcription = DirectMultipleShooting()
    # discretization_method = MeanAndCovariance(with_cholesky=False)

    # # Arm Reaching - DirectCollocationTrapezoidal - MeanAndCovariance  -> ? converge, but runs
    # ocp_example = ArmReaching()
    # dynamics_transcription = DirectCollocationTrapezoidal()
    # discretization_method = MeanAndCovariance(dynamics_transcription, with_cholesky=False, with_helper_matrix=True)

    # Obstacle Avoidance - DirectMultipleShooting - MeanAndCovariance  -> ? converge, but runs
    ocp_example = ObstacleAvoidance()
    dynamics_transcription = DirectCollocationPolynomial()
    discretization_method = MeanAndCovariance(dynamics_transcription, with_cholesky=False, with_helper_matrix=True)

    # Prepare the problem
    ocp = prepare_ocp(
        ocp_example=ocp_example,
        dynamics_transcription=dynamics_transcription,
        discretization_method=discretization_method,
    )

    # Solve the problem
    w_opt, solver, grad_f_func, grad_g_func = solve_ocp(
        ocp,
        ocp_example=ocp_example,
        hessian_approximation="exact",  # or "limited-memory",
        linear_solver="ma57",  # TODO change to "ma97" if available
        pre_optim_plot=False,
        show_online_optim=False,
    )

    # Save the results
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    status = "CVG" if solver.stats()["success"] else "DVG"
    print_tol = "{:1.1e}".format(tol).replace(".", "p")
    save_path = f"results/{ocp_example.name()}_{dynamics_transcription.name()}_{discretization_method.name()}_{status}_{print_tol}_{current_time}.pkl"

    data_saved = save_results(w_opt, ocp, save_path, n_simulations, solver, grad_f_func, grad_g_func)
    print(f"Results saved in {save_path}")

    ocp_example.specific_plot_results(ocp, data_saved, save_path.replace(".pkl", "_specific.png"))


if __name__ == "__main__":
    main()
