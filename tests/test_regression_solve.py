"""
The one "did the actual optimized answer change" regression test in this suite. Full IPOPT
solves are slow for stochastic (NS/SDA) formulations (17s-360s per the project's own paper), so
this is deliberately limited to two fast, Deterministic, small-scale solves; NS/SDA correctness
is instead covered by the fast structural/property tests in test_noise_magnitude.py and
test_problem_construction.py.

Golden values (cost, first/last 5 decision variables) were captured by running the current code
once; a future change that alters the optimized answer will be caught here.

DirectMultipleShooting is excluded: its initialize_dynamics_integrator currently requires an
`is_brownian` argument that prepare_ocp's call site does not pass (in-progress work).
"""
import numpy as np
import numpy.testing as npt
import casadi as cas

from socp import VertebrateArm, prepare_ocp, solve_ocp, DirectCollocationTrapezoidal, Variational, Deterministic

N_SHOOTING = 3


def _solve(dynamics_transcription_cls):
    ocp_example = VertebrateArm(nb_random=1)
    ocp_example.n_shooting = N_SHOOTING
    dynamics_transcription = dynamics_transcription_cls()
    discretization_method = Deterministic(dynamics_transcription)

    ocp = prepare_ocp(ocp_example, dynamics_transcription, discretization_method)
    w_opt, solver, grad_f_func, grad_g_func, save_path, g_without_bounds_at_init = solve_ocp(
        ocp,
        ocp_example,
        linear_solver="mumps",
        show_online_optim=False,
        plot_solution=False,
    )

    assert solver.stats()["success"]

    cost_func = cas.Function("cost", [ocp["w"]], [ocp["j"]])
    cost = float(cost_func(w_opt))
    return cost, w_opt


def test_regression_direct_collocation_trapezoidal_deterministic():
    cost, w_opt = _solve(DirectCollocationTrapezoidal)

    npt.assert_almost_equal(cost, 0.01382773868545999)
    npt.assert_array_almost_equal(w_opt[:5], [1.0, 0.4284424584623453, 2.284707736426437, 0.0, 0.0])
    npt.assert_array_almost_equal(
        w_opt[-5:], [1.2903720522834312, 0.9415315963358117, -1.9839921416565942, 0.014090592610862913, -0.022533417973768097]
    )


def test_regression_variational_deterministic():
    cost, w_opt = _solve(Variational)

    npt.assert_almost_equal(cost, 0.01109645806515947)
    npt.assert_array_almost_equal(w_opt[:5], [1.0, 0.4284424584623453, 2.284707736426437, 0.0, 0.0])
    npt.assert_array_almost_equal(
        w_opt[-5:], [1.2903720522834312, 0.8570193012893859, -1.810565687100966, 0.0, 0.0]
    )
