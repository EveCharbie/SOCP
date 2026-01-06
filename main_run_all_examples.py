import numpy as np
import casadi as cas

from socp import (
    ArmReaching,
    DirectMultipleShooting,
    NoiseDiscretization,
    prepare_ocp,
    solve_ocp,
)

# Arm Reaching
ocp_example = ArmReaching()
dynamics_transcription = DirectMultipleShooting()
discretization_method = NoiseDiscretization()

ocp = prepare_ocp(
    ocp_example=ocp_example,
    dynamics_transcription=dynamics_transcription,
    discretization_method=discretization_method,
)
w_opt, solver = solve_ocp(
    ocp,
    ocp_example=ocp_example,
    hessian_approximation="exact",  # or "limited-memory",
    pre_optim_plot=False,
    # show_online_optim = True, # TODO
)
