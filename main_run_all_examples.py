import numpy as np
import casadi as cas


# Arm Reaching
ocp_example=ArmReaching()
dynamics_transcription = DirectMultipleShooting(),

ocp = prepare_ocp(
    ocp_example=ocp_example,
    dynamics_transcription=dynamics_transcription,
)
w_opt, solver = solve_ocp(
    ocp,
    pre_optim_plot=False,
)
