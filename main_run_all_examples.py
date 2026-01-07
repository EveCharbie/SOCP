import numpy as np
import casadi as cas
from datetime import datetime

from socp import (
    ArmReaching,
    DirectMultipleShooting,
    NoiseDiscretization,
    prepare_ocp,
    solve_ocp,
)

tol = 1e-6
n_simulations = 100

# Arm Reaching
ocp_example = ArmReaching()
dynamics_transcription = DirectMultipleShooting()
discretization_method = NoiseDiscretization()

# Prepare the problem
ocp = prepare_ocp(
    ocp_example=ocp_example,
    dynamics_transcription=dynamics_transcription,
    discretization_method=discretization_method,
)

# Solve the problem
w_opt, solver = solve_ocp(
    ocp,
    ocp_example=ocp_example,
    hessian_approximation="exact",  # or "limited-memory",
    pre_optim_plot=False,
    show_online_optim=True,
)

# Save the results
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
status = "CVG" if solver.stats()["success"] else "DVG"
print_tol = "{:1.1e}".format(tol).replace(".", "p")
save_path = f"results/{ocp_example.name()}_{dynamics_transcription.name()}_{discretization_method.name()}_{status}_{print_tol}_{current_time}.pkl"

variable_data = save_results(w_opt, ocp, save_path, n_simulations, solver)
